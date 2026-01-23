import argparse
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from src.utils.config import load_config, AppConfig
from src.utils.logging import Logger
from src.utils.distributed import setup_distributed, cleanup_distributed
from src.model.transformer import TransformerLM
from src.training.trainer import Trainer
from data.processors import get_dataloader
import os

def main():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Distributed Setup
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Load Config
    raw_config = load_config(args.config)
    
    # In a real setup, we merge configs here. Simplification:
    # We assume the file structure matches the AppConfig exactly or we construct it manually
    # For the prompt's sake, let's manually map the YAML to our schema structure
    # Assuming the config file passed contains keys for all sections
    
    # Mocking the loading for the split file structure provided in requirements
    # Real implementation would load sub-configs.
    # We create a default config object based on what's likely in the files
    
    config = AppConfig(
        model=raw_config.get('model', {}),
        training=raw_config.get('training', {}),
        data=raw_config.get('data', {})
    )
    
    # Create Output Dir
    if rank == 0:
        os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Logger
    logger = Logger(raw_config, use_wandb=False) if rank == 0 else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = TransformerLM(config.model)
    model.to(device)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )

    # Data
    train_loader = get_dataloader(config.training, tokenizer) # Passing training config for batch size

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config.training,
        logger=logger,
        device=device
    )

    try:
        if rank == 0:
            logger.console.info("Starting training...")
        trainer.train()
    except KeyboardInterrupt:
        if rank == 0:
            logger.console.info("Training interrupted.")
    finally:
        if rank == 0:
            logger.close()
        cleanup_distributed()

if __name__ == "__main__":
    main()
