import torch
import torch.nn as nn
from typing import Tuple

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    
    Args:
        d_model (int): The dimension of the model.
        max_seq_len (int): Maximum sequence length.
        base (int): Base for the frequency calculation.
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create rotary inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache cos/sin
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the cos and sin embeddings for the given sequence length.
        
        Args:
            x: Input tensor (used for device check).
            seq_len: Current sequence length.
        Returns:
            Tuple[cos, sin] tensors.
        """
        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype)
        )

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to an input tensor."""
    # x shape: [batch, seq, head, dim] or [batch, head, seq, dim] depending on implementation
    # Assuming [batch, seq, head, dim] for compatibility with this RoPE impl
    
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    
    # Reshape cos/sin for broadcasting
    # cos shape in cache: [seq, dim] -> needs [1, seq, 1, dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    return (x * cos) + (rotated_x * sin)