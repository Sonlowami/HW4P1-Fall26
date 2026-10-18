import copy
import pytest
import torch
from torch import nn


def model():
    return nn.ModuleDict({'q_proj': nn.Linear(4, 4, bias=False), 'v_proj': nn.Linear(4, 2), 'head': nn.Linear(4, 3)})


def test_initial_equivalence_and_known_update():
    from hw4lib.model.lora import LoRALinear
    torch.manual_seed(5)
    base = nn.Linear(4, 2)
    layer = LoRALinear(base, 2, 4, 0)
    x = torch.ones(3, 4)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    with torch.no_grad():
        layer.lora_A.fill_(1)
        layer.lora_B.fill_(1)
    torch.testing.assert_close(layer(x), base(x) + 16)


def test_freezing_and_gradients():
    from hw4lib.model.lora import inject_lora
    m = model()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    assert sorted(inject_lora(m, rank=2, alpha=4, dropout=0)) == ['q_proj', 'v_proj']
    optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        m['q_proj'](torch.ones(2, 4)).sum().backward()
        optimizer.step()
    assert m['q_proj'].lora_A.grad.abs().sum() > 0
    assert m['q_proj'].lora_B.grad.abs().sum() > 0
    for name, p in m.named_parameters():
        assert p.requires_grad == (name.endswith('lora_A') or name.endswith('lora_B'))
    for name, expected in before.items():
        actual = m.state_dict()[name.replace('.weight', '.base.weight').replace('.bias', '.base.bias') if not name.startswith('head.') else name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_roundtrip_and_mismatch(tmp_path):
    from hw4lib.model.lora import inject_lora, save_adapter, load_adapter
    m = model()
    other = copy.deepcopy(m)
    wrong = copy.deepcopy(m)
    for obj in (m, other): inject_lora(obj, rank=2, alpha=4, dropout=0)
    with torch.no_grad(): m['q_proj'].lora_B.fill_(.2)
    file = tmp_path / 'adapter.pth'
    save_adapter(m, file, {'base_checkpoint': 'base.pth'})
    load_adapter(other, file, base_checkpoint='base.pth')
    x = torch.randn(2, 4)
    torch.testing.assert_close(m['q_proj'](x), other['q_proj'](x))
    assert all('lora_' in k for k in torch.load(file, weights_only=True)['state_dict'])
    inject_lora(wrong, rank=1, alpha=4, dropout=0)
    with pytest.raises(ValueError): load_adapter(wrong, file)
    with pytest.raises(FileExistsError): save_adapter(m, file, {})


def test_missing_targets_and_repeated_injection():
    from hw4lib.model.lora import inject_lora
    m = model()
    with pytest.raises(ValueError): inject_lora(m, rank=2, alpha=4, dropout=0, targets=('missing',))
    inject_lora(m, rank=2, alpha=4, dropout=0)
    with pytest.raises(ValueError): inject_lora(m, rank=2, alpha=4, dropout=0)

def test_partial_target_typo_rejected_before_mutation():
    from hw4lib.model.lora import inject_lora, LoRALinear
    m = model()
    with pytest.raises(ValueError, match="v_projj"):
        inject_lora(m, rank=2, alpha=4, dropout=0, targets=('q_proj', 'v_projj'))
    assert not any(isinstance(module, LoRALinear) for module in m.modules())
    assert all(p.requires_grad for p in m.parameters())


def test_adapter_checks_base_attention_geometry_and_checkpoint(tmp_path):
    from hw4lib.model import DecoderOnlyTransformer
    from hw4lib.model.lora import inject_lora, save_adapter, load_adapter
    cfg = dict(num_layers=1, d_model=16, num_heads=2, num_kv_heads=1,
               d_ff=32, dropout=0, max_len=16, num_classes=16)
    a = DecoderOnlyTransformer(**cfg)
    b = DecoderOnlyTransformer(**dict(cfg, num_heads=4, num_kv_heads=2))
    for m in (a, b): inject_lora(m, rank=2, alpha=4, dropout=0)
    path = tmp_path / 'adapter.pth'
    save_adapter(a, path, {'model': cfg, 'base_checkpoint': 'base.pth'})
    with pytest.raises(ValueError):
        load_adapter(b, path)
    with pytest.raises(ValueError):
        load_adapter(a, path, base_checkpoint='another-base.pth')
    save_adapter(a, path, {'model': cfg, 'base_checkpoint': 'base.pth'}, reuse=True)
    with torch.no_grad(): a.dec_layers[0].self_attn.mha.q_proj.lora_B.add_(1)
    with pytest.raises(ValueError):
        save_adapter(a, path, {'model': cfg, 'base_checkpoint': 'base.pth'}, reuse=True)
