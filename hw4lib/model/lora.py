"""Low-rank adaptation of P1 attention projections."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..utils.sft_experiment import validate_model_config


class LoRALinear(nn.Module):
    """Frozen linear projection plus (alpha / rank) B A dropout(x).

    A has shape (rank, in_features), B has shape (out_features, rank).
    Initialize B to zero so inserting LoRA leaves predictions unchanged.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Require positive alpha and 0 <= dropout < 1")
        self.base, self.rank, self.alpha = base, rank, float(alpha)
        self.dropout = nn.Dropout(dropout)
        # TODO: Create trainable A and B on the base weight's device and dtype.
        # Initialize A, zero B, and freeze the original projection.
        raise NotImplementedError("Complete LoRALinear.__init__")

    def forward(self, x):
        # TODO: Add the scaled low-rank update to the original linear output.
        raise NotImplementedError("Complete LoRALinear.forward")


def inject_lora(model, *, rank, alpha, dropout, targets=("q_proj", "v_proj")):
    """Replace named projection leaves, then freeze all non-adapter parameters.

    Return the replaced module paths. Validate before mutating the model.
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("LoRA is already installed")
    # TODO: Find nn.Linear leaves whose names match targets; reject any unmatched target.
    # Build replacements before modifying the model. Freeze base parameters.
    raise NotImplementedError("Complete inject_lora")


def _adapter_config(model):
    return {name: {"rank": module.rank, "alpha": module.alpha,
                   "dropout": module.dropout.p,
                   "in_features": module.base.in_features,
                   "out_features": module.base.out_features}
            for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def save_adapter(model, path: Path, metadata: dict, *, reuse=False) -> None:
    """Provided: save adapter tensors/config only; retain the original base separately."""
    config = _adapter_config(model)
    if not config:
        raise ValueError("Model has no LoRA adapters")
    validate_model_config(model, metadata.get("model", {}))
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.endswith((".lora_A", ".lora_B"))}
    if Path(path).exists() and reuse:
        old = torch.load(path, map_location="cpu", weights_only=True)
        if (old.get("config") != config or old.get("metadata") != metadata
                or set(old.get("state_dict", {})) != set(state)
                or any(not torch.equal(old["state_dict"][name], value) for name, value in state.items())):
            raise ValueError("Existing adapter differs from the selected checkpoint; use a new run name")
        return
    with Path(path).open("xb") as f:
        torch.save({"config": config, "state_dict": state, "metadata": metadata}, f)


def load_adapter(model, path: Path, *, base_checkpoint=None) -> dict:
    """Provided: restore onto a matching base architecture and checkpoint reference.

    Load the original base with load_base_weights first. For a relocated base,
    pass its recorded checkpoint reference explicitly after verifying its origin.
    Paths/configuration do not prove that a weight file's contents are unchanged.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    validate_model_config(model, metadata.get("model", {}))
    recorded_base = metadata.get("base_checkpoint")
    if recorded_base:
        actual_base = base_checkpoint or getattr(model, "_hw4p2_weight_source", {}).get("path")
        if actual_base is None or Path(actual_base).expanduser().resolve() != Path(recorded_base).expanduser().resolve():
            raise ValueError("Adapter base checkpoint does not match; load the recorded P1 base first")
    if payload.get("config") != _adapter_config(model) or not payload["config"]:
        raise ValueError("Adapter architecture/configuration does not match")
    expected = {n: p for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))}
    state = payload["state_dict"]
    if set(expected) != set(state) or any(state[n].shape != p.shape for n, p in expected.items()):
        raise ValueError("Adapter keys/shapes do not match")
    with torch.no_grad():
        for name, param in expected.items():
            param.copy_(state[name].to(param))
    return metadata
