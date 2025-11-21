import logging
import os
from typing import Dict, Any, Optional
import wandb
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """Unified logging interface for W&B, TensorBoard, and Console."""

    def __init__(self, config: Dict[str, Any], use_wandb: bool = False, log_dir: str = "logs"):
        self.use_wandb = use_wandb
        self.step = 0
        
        # Console Logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.console = logging.getLogger("LLM-Systems")

        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        # W&B
        if self.use_wandb:
            wandb.init(project="llm-systems-project", config=config)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Logs metrics to all active trackers."""
        self.step = step
        
        # Console (subset)
        if step % 100 == 0:
            msg = f"Step {step} | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.console.info(msg)

        # TensorBoard
        for k, v in metrics.items():
            self.tb_writer.add_scalar(k, v, step)

        # W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def close(self) -> None:
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()