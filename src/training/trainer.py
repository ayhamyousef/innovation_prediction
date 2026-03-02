"""
Training loop for the Transformer patent classifier.

Features:
  - Mixed precision (FP16) training
  - Learning rate scheduling (cosine, linear, constant)
  - Early stopping on validation metric
  - Gradient clipping
  - Periodic evaluation and checkpointing
  - WandB logging (optional)
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.models.baselines import compute_metrics

logger = logging.getLogger("innovation_prediction.trainer")


class Trainer:
    """
    Handles training, validation, and testing of the Transformer model.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, test_loader: Optional[DataLoader],
                 cfg: Dict, device: torch.device,
                 class_weights: Optional[torch.Tensor] = None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg["training"]
        self.device = device

        # Loss with optional class weights
        if class_weights is not None and cfg["training"]["use_class_weights"]:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )

        # Scheduler
        num_training_steps = (
            len(train_loader) * self.cfg["epochs"]
            // self.cfg.get("gradient_accumulation_steps", 1)
        )
        num_warmup_steps = int(num_training_steps * self.cfg["warmup_ratio"])

        self.scheduler = get_scheduler(
            name=self.cfg["scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed precision
        self.use_fp16 = self.cfg["fp16"] and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_fp16)

        # Early stopping
        self.patience = self.cfg["early_stopping_patience"]
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.best_state = None

        # Output
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulators
        self.global_step = 0
        self.train_losses = []
        self.val_metrics_history = []

    def train(self) -> Dict:
        """Full training loop with validation and early stopping."""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"  Epochs: {self.cfg['epochs']}")
        logger.info(f"  Batch size: {self.cfg['batch_size']}")
        logger.info(f"  LR: {self.cfg['learning_rate']}")
        logger.info(f"  FP16: {self.use_fp16}")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 60)

        accum_steps = self.cfg.get("gradient_accumulation_steps", 1)

        for epoch in range(1, self.cfg["epochs"] + 1):
            epoch_start = time.time()
            self.model.train()

            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast(enabled=self.use_fp16):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids"),
                        metadata=batch.get("metadata"),
                        labels=batch["labels"],
                    )

                    # Use custom loss if class weights provided
                    loss = self.criterion(outputs["logits"], batch["labels"])
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg["max_grad_norm"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                epoch_loss += loss.item() * accum_steps
                epoch_steps += 1

                # Periodic logging
                if self.global_step % self.cfg["log_every_n_steps"] == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Step {self.global_step} | "
                        f"Loss: {loss.item() * accum_steps:.4f} | "
                        f"LR: {lr:.2e}"
                    )

            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.train_losses.append(avg_loss)
            elapsed = time.time() - epoch_start

            # Validate
            val_metrics = self.evaluate(self.val_loader)
            self.val_metrics_history.append(val_metrics)

            logger.info(
                f"Epoch {epoch}/{self.cfg['epochs']} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Val AUC: {val_metrics.get('roc_auc', 0):.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping
            current_metric = val_metrics[self.cfg["early_stopping_metric"]]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                if self.cfg["save_best_model"]:
                    self._save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info(f"Restored best model (metric={self.best_metric:.4f})")

        # Final test evaluation
        results = {"train_losses": self.train_losses,
                    "val_metrics": self.val_metrics_history}
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            results["test_metrics"] = test_metrics
            logger.info("=" * 60)
            logger.info("TEST RESULTS:")
            logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
            logger.info(f"  ROC-AUC:   {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  CrossEnt:  {test_metrics.get('cross_entropy', 0):.4f}")
            logger.info("=" * 60)

        return results

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on a dataloader."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.use_fp16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    metadata=batch.get("metadata"),
                    labels=batch["labels"],
                )

            logits = outputs["logits"]
            loss = self.criterion(logits, batch["labels"])
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["eval_loss"] = total_loss / max(n_batches, 1)

        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        path = self.output_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
