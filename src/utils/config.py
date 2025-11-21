import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class ModelConfig(BaseModel):
    """Configuration for the Transformer Model."""
    d_model: int = Field(..., gt=0)
    n_layers: int = Field(..., gt=0)
    n_heads: int = Field(..., gt=0)
    d_ff: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    max_seq_len: int = Field(1024, gt=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    activation: str = "gelu"

class TrainingConfig(BaseModel):
    """Configuration for the Training Loop."""
    batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    learning_rate: float = Field(..., gt=0.0)
    weight_decay: float = 0.01
    max_steps: int = Field(..., gt=0)
    warmup_steps: int = 0
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    mixed_precision: bool = True
    compile_model: bool = False
    output_dir: str = "./checkpoints"

class DataConfig(BaseModel):
    """Configuration for Data Loading."""
    train_path: str
    val_path: Optional[str] = None
    tokenizer_name: str = "gpt2"
    num_workers: int = 4
    prefetch_factor: int = 2

class AppConfig(BaseModel):
    """Master Configuration Object."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)