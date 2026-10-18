import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        pe = torch.zeros((max_len, d_model), dtype=torch.float32)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer to save with model state
        self.register_buffer('pe', pe.unsqueeze(0))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype)


class RotaryPositionalEncoding(nn.Module):
    """Apply rotary position information to attention queries and keys."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0):
        """
        Initialize the RotaryPositionalEncoding.
        Args:
            head_dim (int): The dimension of each attention head.
            max_len (int): The maximum context length.
            base (float): The base used for the rotary frequencies.
        """
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary encoding")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if base <= 0:
            raise ValueError("base must be positive")

        self.head_dim = head_dim
        self.max_len = max_len

        # Create one inverse frequency for each neighboring feature pair.
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequencies = base ** (-dimensions / head_dim)

        # Build angle tables with one value per neighboring feature pair.
        positions = torch.arange(max_len, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)[None, None, :, :]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)[None, None, :, :]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass for the RotaryPositionalEncoding.
        Args:
            x (torch.Tensor): Split attention heads. shape: (B, H, T, head_dim)
            start_pos (int): Absolute position of the first token in x.
        Returns:
            torch.Tensor: Rotated attention heads with the same shape as x.
        """
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, heads, sequence, head_dim)")
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f"Expected head_dim {self.head_dim}, but received {x.size(-1)}"
            )
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError("start_pos must be an integer")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        # DO NOT MODIFY: input and context validation.
        seq_len = x.size(-2)
        end_pos = start_pos + seq_len
        if end_pos > self.max_len:
            raise ValueError(
                f"Positions up to {end_pos - 1} exceed max_len {self.max_len}"
            )

        # Select the requested positions and cast them to the input dtype.
        cos = self.cos[:, :, start_pos:end_pos, :].to(dtype=x.dtype)
        sin = self.sin[:, :, start_pos:end_pos, :].to(dtype=x.dtype)

        # Rotate every neighboring even/odd feature pair and interleave them.
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos[..., 0::2] - odd * sin[..., 0::2]
        rotated_odd = even * sin[..., 0::2] + odd * cos[..., 0::2]
        output = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
        return output
