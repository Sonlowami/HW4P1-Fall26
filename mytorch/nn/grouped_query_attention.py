from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np
from typing import Optional

class GroupedQueryAttention:
    """
    Grouped Query Attention
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of key/value heads shared across query groups
        """
        # DO NOT MODIFY
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.num_kv_heads  = num_kv_heads
        self.head_dim      = embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # TODO: Implement the attention layer and the projections

        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()

        # Initialize your linear layers
        # Query is projected to embed_dim, but key and value are compact:
        # they carry num_kv_heads heads instead of num_heads.
        # What output size does that give the key and value projections?
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim // self.num_kv_heads)
        self.v_proj   = Linear(embed_dim, embed_dim // self.num_kv_heads)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(
        self,
        Wq: np.ndarray,
        bq: np.ndarray,
        Wk: np.ndarray,
        bk: np.ndarray,
        Wv: np.ndarray,
        bv: np.ndarray,
        Wo: np.ndarray,
        bo: np.ndarray,
    ) -> None:
        """
        Initialize the weights and biases with the given values.
        Note that Wk and Wv are shaped for the compact key/value heads.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        key_padding_mask: Optional[np.ndarray] = None,
        attn_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """

        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # Project inputs
        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        # Reshape for multiple heads
        # Query and key/value do not use the same head count here
        q = self._split_heads(q)
        k = self._split_heads(k, num_heads=self.num_kv_heads)
        v = self._split_heads(v, num_heads=self.num_kv_heads)

        # Repeat key/value heads for their query groups
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Combine padding and causal masks
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply attention
        attn_outputs = self.attention.forward(q, k, v, mask)

        # Merge heads
        attn_output = self._concat_heads(attn_outputs)

        # Final projection
        output = self.out_proj.forward(attn_output)

        return output

    def backward(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for grouped query attention.
        """

        # TODO: Implement backward pass

        # Backpropagate through output projection
        d_attn_output = self.out_proj.backward(d_output)

        # Undo head splitting
        d_attn_outputs = self._split_heads(d_attn_output)

        # Backpropagate through attention
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # Reduce the repeated key/value gradients back to compact heads
        d_k = d_k.reshape(
            self.N, self.num_kv_heads, self.num_kv_groups, self.S, self.head_dim).sum(axis=2)
        d_v = d_v.reshape(
            self.N, self.num_kv_heads, self.num_kv_groups, self.S, self.head_dim).sum(axis=2)

        # Merge head gradients
        d_q = self._concat_heads(d_q)
        d_k = np.swapaxes(d_k, 1, 2).reshape(
            self.N, self.S, self.num_kv_heads * self.head_dim
        )
        d_v = np.swapaxes(d_v, 1, 2).reshape(
            self.N, self.S, self.num_kv_heads * self.head_dim
        )

        # Backpropagate through input projections
        d_q = self.q_proj.backward(d_q)
        d_k = self.k_proj.backward(d_k)
        d_v = self.v_proj.backward(d_v)

        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge two mask types into a single mask.
        """
        if key_padding_mask is None and attn_mask is None:
            return None

        key_mask = None
        attention_mask = None

        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]

        if attn_mask is not None:
            attention_mask = attn_mask[None, None, :, :]

        if key_mask is None:
            combined_mask = attention_mask
        elif attention_mask is None:
            combined_mask = key_mask
        else:
            combined_mask = np.logical_or(key_mask, attention_mask)

        return np.broadcast_to(
            combined_mask,
            (self.N, self.num_heads, self.L, self.S),
        )

    def _split_heads(
        self,
        x: np.ndarray,
        num_heads: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reshape tensor for multi-head attention.
        Query is split into num_heads, key and value into num_kv_heads.
        :param x: (N, T, num_heads * head_dim)
        :return: (N, num_heads, T, head_dim)
        """
        # Select the head count for this call (DO NOT MODIFY)
        num_heads = self.num_heads if num_heads is None else num_heads

        # TODO: Implement head splitting
        N, T, E = x.shape
        x = x.reshape((N, T, num_heads, E // num_heads))

        # Reshape and transpose for heads
        x = np.swapaxes(x, 1, 2)
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # Transpose and reshape
        x = np.swapaxes(x, 1, 2)
        x = x.reshape((self.N, self.L, self.E))
        return x

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Repeat each key/value head for the query heads that share it.
        :param x: (N, num_kv_heads, S, head_dim)
        :return: (N, num_heads, S, head_dim)
        """
        # TODO: Implement key/value head repetition

        # Each head is needed num_kv_groups times
        # Think about where the copies of one head have to sit so that a
        # query head lines up with the key/value head of its own group
        x = np.repeat(x, self.num_kv_groups, axis=1)

        return x
