import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .embeddings import RotaryPositionalEmbedding
import torch.utils.checkpoint as checkpoint

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(
            config.d_model, 
            config.n_heads, 
            config.max_seq_len, 
            config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout, config.activation)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm architecture
        attn_out, _ = self.attn(self.ln1(x), cos, sin, mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> dict:
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Seq len {T} exceeds max {self.config.max_seq_len}"

        x = self.token_embedding(idx)
        x = self.dropout(x)
        
        # RoPE
        cos, sin = self.rope(x, T)
        
        for block in self.blocks:
            if self.training:
                # Activation Checkpointing
                x = checkpoint.checkpoint(block, x, cos, sin, None, use_reentrant=False)
            else:
                x = block(x, cos, sin, None)
                
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return {"logits": logits, "loss": loss}
        else:
            logits = self.lm_head(x[:, [-1], :]) # only last token
            return {"logits": logits}