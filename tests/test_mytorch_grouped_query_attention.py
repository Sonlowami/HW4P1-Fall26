from collections.abc import Callable
from contextlib import contextmanager
import warnings

import numpy as np
import torch
import torch.nn.functional as torch_functional

from mytorch.nn import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)


@contextmanager
def ignore_macos_matmul_warnings():
    """Ignore false-positive Accelerate matmul warnings in this test process."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        yield


def run_with_macos_matmul_warnings_ignored(
    test_function: Callable[[], None],
) -> None:
    """Run one legacy NumPy test with the narrow macOS warning filter.

    Args:
        test_function: Zero-argument test function to run.

    Returns:
        None.
    """
    with ignore_macos_matmul_warnings():
        test_function()


def _make_weights(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Create deterministic projection parameters for one attention module.

    Args:
        embed_dim: Size of the input and output embeddings.
        num_heads: Number of query heads.
        num_kv_heads: Number of key and value heads.
        seed: Seed used by the random number generator.

    Returns:
        Query, key, value, and output weights and biases in ``init_weights``
        argument order.
    """
    generator = np.random.default_rng(seed)
    head_dim = embed_dim // num_heads
    kv_dim = num_kv_heads * head_dim

    Wq = generator.normal(scale=0.2, size=(embed_dim, embed_dim)).astype(np.float32)
    bq = generator.normal(scale=0.2, size=(embed_dim,)).astype(np.float32)
    Wk = generator.normal(scale=0.2, size=(kv_dim, embed_dim)).astype(np.float32)
    bk = generator.normal(scale=0.2, size=(kv_dim,)).astype(np.float32)
    Wv = generator.normal(scale=0.2, size=(kv_dim, embed_dim)).astype(np.float32)
    bv = generator.normal(scale=0.2, size=(kv_dim,)).astype(np.float32)
    Wo = generator.normal(scale=0.2, size=(embed_dim, embed_dim)).astype(np.float32)
    bo = generator.normal(scale=0.2, size=(embed_dim,)).astype(np.float32)

    weights = (Wq, bq, Wk, bk, Wv, bv, Wo, bo)
    return weights


def _torch_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    weights: tuple[np.ndarray, ...],
    num_heads: int,
    num_kv_heads: int,
    key_padding_mask: np.ndarray,
    attn_mask: np.ndarray,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Build the same grouped-query calculation with PyTorch autograd.

    Args:
        query: Query input with shape ``(N, L, embed_dim)``.
        key: Key input with shape ``(N, S, embed_dim)``.
        value: Value input with shape ``(N, S, embed_dim)``.
        weights: Projection parameters in ``init_weights`` argument order.
        num_heads: Number of query heads.
        num_kv_heads: Number of key and value heads.
        key_padding_mask: Boolean padding mask with shape ``(N, S)``.
        attn_mask: Boolean attention mask with shape ``(L, S)``.

    Returns:
        The reference output, differentiable input tensors, and differentiable
        projection parameter tensors.
    """
    query_tensor = torch.tensor(query, dtype=torch.float32, requires_grad=True)
    key_tensor = torch.tensor(key, dtype=torch.float32, requires_grad=True)
    value_tensor = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    parameter_tensors = tuple(
        torch.tensor(parameter, dtype=torch.float32, requires_grad=True)
        for parameter in weights
    )
    Wq, bq, Wk, bk, Wv, bv, Wo, bo = parameter_tensors

    batch_size, query_length, embed_dim = query_tensor.shape
    key_length = key_tensor.shape[1]
    head_dim = embed_dim // num_heads
    num_kv_groups = num_heads // num_kv_heads

    projected_query = torch_functional.linear(query_tensor, Wq, bq)
    projected_key = torch_functional.linear(key_tensor, Wk, bk)
    projected_value = torch_functional.linear(value_tensor, Wv, bv)

    query_heads = projected_query.reshape(
        batch_size,
        query_length,
        num_heads,
        head_dim,
    ).transpose(1, 2)
    key_heads = projected_key.reshape(
        batch_size,
        key_length,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)
    value_heads = projected_value.reshape(
        batch_size,
        key_length,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    repeated_key_heads = key_heads.repeat_interleave(num_kv_groups, dim=1)
    repeated_value_heads = value_heads.repeat_interleave(num_kv_groups, dim=1)

    scores = torch.matmul(query_heads, repeated_key_heads.transpose(-2, -1))
    scores = scores / np.sqrt(head_dim)

    key_mask = torch.tensor(key_padding_mask, dtype=torch.bool)
    key_mask = key_mask[:, None, None, :].expand(
        batch_size,
        num_heads,
        query_length,
        key_length,
    )
    attention_mask = torch.tensor(attn_mask, dtype=torch.bool)
    attention_mask = attention_mask[None, None, :, :].expand(
        batch_size,
        num_heads,
        query_length,
        key_length,
    )
    combined_mask = torch.logical_or(key_mask, attention_mask)
    masked_scores = scores.masked_fill(combined_mask, -1e10)

    probabilities = torch.softmax(masked_scores, dim=-1)
    context = torch.matmul(probabilities, repeated_value_heads)
    merged_context = context.transpose(1, 2).reshape(
        batch_size,
        query_length,
        embed_dim,
    )
    output = torch_functional.linear(merged_context, Wo, bo)

    input_tensors = (query_tensor, key_tensor, value_tensor)
    reference = (output, input_tensors, parameter_tensors)
    return reference


def test_grouped_query_attention_forward_and_backward() -> None:
    """Compare MHA, GQA, and MQA outputs and gradients with PyTorch.

    Args:
        None.

    Returns:
        None.
    """
    generator = np.random.default_rng(11785)
    batch_size = 2
    query_length = 3
    key_length = 4
    embed_dim = 8
    num_heads = 4

    query = generator.normal(size=(batch_size, query_length, embed_dim)).astype(np.float32)
    key = generator.normal(size=(batch_size, key_length, embed_dim)).astype(np.float32)
    value = generator.normal(size=(batch_size, key_length, embed_dim)).astype(np.float32)
    d_output = generator.normal(
        size=(batch_size, query_length, embed_dim)
    ).astype(np.float32)

    key_padding_mask = np.array(
        [[False, False, False, True], [False, False, True, False]]
    )
    attn_mask = np.array(
        [
            [False, False, True, False],
            [False, False, False, True],
            [False, True, False, False],
        ]
    )

    for num_kv_heads in (4, 2, 1):
        weights = _make_weights(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seed=100 + num_kv_heads,
        )
        attention = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        attention.init_weights(*weights)

        with ignore_macos_matmul_warnings():
            output = attention.forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            d_query, d_key, d_value = attention.backward(d_output)
        assert np.isfinite(output).all()
        assert np.isfinite(d_query).all()
        assert np.isfinite(d_key).all()
        assert np.isfinite(d_value).all()

        reference_output, reference_inputs, reference_parameters = _torch_reference(
            query=query,
            key=key,
            value=value,
            weights=weights,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        reference_output.backward(torch.tensor(d_output, dtype=torch.float32))

        np.testing.assert_allclose(
            output,
            reference_output.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        for actual, reference_input in zip(
            (d_query, d_key, d_value),
            reference_inputs,
        ):
            np.testing.assert_allclose(
                actual,
                reference_input.grad.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

        actual_parameter_gradients = (
            attention.q_proj.dLdW,
            attention.q_proj.dLdb,
            attention.k_proj.dLdW,
            attention.k_proj.dLdb,
            attention.v_proj.dLdW,
            attention.v_proj.dLdb,
            attention.out_proj.dLdW,
            attention.out_proj.dLdb,
        )
        for actual, reference_parameter in zip(
            actual_parameter_gradients,
            reference_parameters,
        ):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(
                actual,
                reference_parameter.grad.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )


def test_attention_specializations_and_optional_masks() -> None:
    """Check the MHA/MQA wrappers and each optional-mask path.

    Args:
        None.

    Returns:
        None.
    """
    generator = np.random.default_rng(7)
    embed_dim = 8
    num_heads = 4
    inputs = generator.normal(size=(2, 3, embed_dim)).astype(np.float32)
    key_padding_mask = np.array(
        [[False, False, True], [False, True, False]]
    )
    attn_mask = np.triu(np.ones((3, 3), dtype=bool), k=1)

    specializations = (
        (MultiHeadAttention(embed_dim, num_heads), num_heads),
        (MultiQueryAttention(embed_dim, num_heads), 1),
    )
    for specialized_attention, num_kv_heads in specializations:
        generic_attention = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        weights = _make_weights(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seed=num_kv_heads,
        )
        specialized_attention.init_weights(*weights)
        generic_attention.init_weights(*weights)

        with ignore_macos_matmul_warnings():
            specialized_output = specialized_attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            generic_output = generic_attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        np.testing.assert_allclose(specialized_output, generic_output)

        with ignore_macos_matmul_warnings():
            no_mask_output = specialized_attention.forward(inputs, inputs, inputs)
            padding_only_output = specialized_attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
            )
            attention_only_output = specialized_attention.forward(
                inputs,
                inputs,
                inputs,
                attn_mask=attn_mask,
            )
        assert np.isfinite(no_mask_output).all()
        assert np.isfinite(padding_only_output).all()
        assert np.isfinite(attention_only_output).all()


def test_grouped_query_attention_validation_and_shapes() -> None:
    """Check head-count validation and compact projection dimensions.

    Args:
        None.

    Returns:
        None.
    """
    invalid_arguments = (
        (0, 4, 2),
        (8, 0, 1),
        (8, 4, 0),
        (10, 4, 2),
        (8, 4, 3),
    )
    for arguments in invalid_arguments:
        raised_error = False
        try:
            GroupedQueryAttention(*arguments)
        except ValueError:
            raised_error = True
        assert raised_error

    attention = GroupedQueryAttention(
        embed_dim=16,
        num_heads=8,
        num_kv_heads=2,
    )
    assert attention.q_proj.W.shape == (16, 16)
    assert attention.k_proj.W.shape == (4, 16)
    assert attention.v_proj.W.shape == (4, 16)
    assert attention.out_proj.W.shape == (16, 16)

    compact_heads = np.arange(8).reshape(1, 2, 2, 2)
    repeated_heads = attention._repeat_kv(compact_heads)
    expected_repeated = np.repeat(compact_heads, 4, axis=1)
    np.testing.assert_array_equal(repeated_heads, expected_repeated)

    repeated_gradients = np.ones((1, 8, 2, 2))
    compact_gradients = attention._sum_kv_gradients(repeated_gradients)
    expected_gradients = np.full((1, 2, 2, 2), 4.0)
    np.testing.assert_array_equal(compact_gradients, expected_gradients)


def test_fully_masked_rows_have_zero_probabilities() -> None:
    """Check that MHA, GQA, and MQA do not attend through an empty row.

    Args:
        None.

    Returns:
        None.
    """
    generator = np.random.default_rng(19)
    inputs = generator.normal(size=(2, 3, 8)).astype(np.float32)
    key_padding_mask = np.ones((2, 3), dtype=bool)

    attention_modules = (
        MultiHeadAttention(embed_dim=8, num_heads=4),
        GroupedQueryAttention(embed_dim=8, num_heads=4, num_kv_heads=2),
        MultiQueryAttention(embed_dim=8, num_heads=4),
    )
    for index, attention in enumerate(attention_modules):
        weights = _make_weights(
            embed_dim=8,
            num_heads=4,
            num_kv_heads=attention.num_kv_heads
            if hasattr(attention, "num_kv_heads")
            else 4,
            seed=200 + index,
        )
        attention.init_weights(*weights)
        with ignore_macos_matmul_warnings():
            output = attention.forward(
                inputs,
                inputs,
                inputs,
                key_padding_mask=key_padding_mask,
            )
        probabilities = attention.attention.attention_scores
        assert np.isfinite(output).all()
        assert np.count_nonzero(probabilities) == 0


if __name__ == "__main__":
    test_grouped_query_attention_forward_and_backward()
    test_attention_specializations_and_optional_masks()
    test_grouped_query_attention_validation_and_shapes()
    test_fully_masked_rows_have_zero_probabilities()
    print("All grouped-query attention tests passed.")
