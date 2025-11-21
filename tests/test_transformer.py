import pytest
import torch
from src.model.transformer import TransformerLM
from src.utils.config import ModelConfig

@pytest.fixture
def config():
    return ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        vocab_size=100,
        max_seq_len=32
    )

def test_forward_pass(config):
    model = TransformerLM(config)
    x = torch.randint(0, 100, (2, 16)) # Batch 2, Seq 16
    out = model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 100) # Last token only for inference default

def test_training_loss(config):
    model = TransformerLM(config)
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    out = model(x, targets=y)
    assert "loss" in out
    assert not torch.isnan(out["loss"])