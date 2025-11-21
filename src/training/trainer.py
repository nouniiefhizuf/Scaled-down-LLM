import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from ..utils.logging import Logger
from ..utils.distributed import is_main_process
from typing import Optional

class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        config,
        logger: Logger,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.logger = logger
        self.device = device
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        self.scheduler = self._get_scheduler()
        
        if config.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

    def _get_scheduler(self):
        return OneCycleLR(
            self.optimizer, 
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_steps / self.config.max_steps,
            anneal_strategy='cos'
        )

    def train(self):
        self.model.train()
        step = 0
        data_iter = iter(self.train_loader)
        accum_steps = self.config.gradient_accumulation_steps
        
        start_time = time.time()
        tokens_processed = 0

        while step < self.config.max_steps:
            t0_step = time.time()
            optimizer_step_performed = False
            
            # Gradient Accumulation Loop
            for micro_step in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                inputs = batch["input_ids"].to(self.device)
                targets = batch["labels"].to(self.device)
                
                # Mixed Precision Forward
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(inputs, targets)
                    loss = outputs["loss"] / accum_steps

                # Backward
                self.scaler.scale(loss).backward()
                tokens_processed += inputs.numel()

            # Optimizer Step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            step += 1
            dt_step = time.time() - t0_step

            # Logging
            if step % self.config.log_interval == 0 and is_main_process():
                loss_val = loss.item() * accum_steps
                tps = tokens_processed / (time.time() - start_time)
                
                metrics = {
                    "train/loss": loss_val,
                    "train/perplexity": torch.exp(torch.tensor(loss_val)).item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "perf/step_time": dt_step,
                    "perf/tokens_per_sec": tps
                }
                self.logger.log_metrics(metrics, step)

            # Checkpointing (simplified for brevity)
            if step % self.config.save_interval == 0 and is_main_process():
                self._save_checkpoint(step)

    def _save_checkpoint(self, step):
        path = f"{self.config.output_dir}/ckpt_{step}.pt"
        # Unwrap DDP if necessary
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        torch.save({
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        self.logger.console.info(f"Saved checkpoint to {path}")