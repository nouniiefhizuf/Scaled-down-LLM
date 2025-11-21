import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .embeddings import apply_rope

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Flash Attention v2 and RoPE.
    
    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = dropout
        
        # Flash Attention check
        self.flash_attn_available = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_cache: Optional[tuple] = None
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [Batch, Seq, Dim]
            cos, sin: RoPE embeddings.
            mask: Attention mask.
            use_cache: Whether to return KV cache.
            kv_cache: Previous KV states.
            
        Returns:
            Output tensor and (optionally) new KV cache.
        """
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE (Rotate Q and K)
        # Note: apply_rope expects [Batch, Seq, Head, Dim]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # KV Cache Logic
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
            
        current_kv_cache = (k, v) if use_cache else None
        
        # Transpose for Attention: [Batch, Head, Seq, Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash Attention or Manual Fallback
        if self.flash_attn_available:
            # SDPA expects [Batch, Head, Seq, Dim]
            # Is_causal=True handles the lower triangular mask automatically
            # Only use is_causal=True if we are training (full seq). 
            # During inference with cache, we use manual masking or rely on shape.
            is_causal = mask is None and T > 1
            
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
            )
        else:
            # Manual implementation (fallback)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            elif T > 1:
                # Causal mask
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        
        return y, current_kv_cache