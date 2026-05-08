from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, ProjectConfiguration
from torch.utils.data import DataLoader

from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import lfm2_collator
from liquid_audio.model.lfm2_audio import LFM2AudioModelOutput

if TYPE_CHECKING:
    from liquid_audio.data.dataloader import LFM2DataLoader
    from liquid_audio.data.types import LFM2AudioModelInput


class Trainer:
    def __init__(
        self,
        model_id: str = "LiquidAI/LFM2.5-Audio-1.5B",
        train_data: LFM2DataLoader | None = None,
        val_data: LFM2DataLoader | None = None,
        lr: float = 3e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        min_ratio: float = 0.1,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        logging_interval: int = 10,
        save_interval: int = 500,
        val_interval: int = 100,
        output_dir: str = "tmp",
    ) -> None:
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.logging_interval = logging_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.output_dir = output_dir

        self.accelerator = Accelerator(
            mixed_precision="bf16",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=False,
            ),
            project_config=ProjectConfiguration(
                project_dir=self.output_dir,
                automatic_checkpoint_naming=True,
                total_limit=30,
            ),
        )

        # Model
        self.model = LFM2AudioModel.from_pretrained(
            model_id,
            device=self.accelerator.device,
            dtype=torch.bfloat16,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=1e-8,
            weight_decay=weight_decay,
            fused=True,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=max(1, max_steps - warmup_steps),
            eta_min=lr * min_ratio,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Data
        if train_data is None:
            raise ValueError("train_data is required")
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lfm2_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            persistent_workers=dataloader_num_workers > 0,
            prefetch_factor=2 if dataloader_num_workers > 0 else None,
        )
        self.val_loader = None
        if val_data is not None:
            self.val_loader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lfm2_collator,
                num_workers=dataloader_num_workers,
                pin_memory=True,
                persistent_workers=dataloader_num_workers > 0,
                prefetch_factor=2 if dataloader_num_workers > 0 else None,
            )

        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        )

        self.optimizer.zero_grad()
        self.step = 0
        self.epoch = 0
        self.time = 0.0

    def train(self) -> None:
        self.time = time.monotonic()
        total = int(time.monotonic() - self.time)
        mins, secs = divmod(total, 60)
        self.accelerator.print(f"[{mins:02d}:{secs:02d}] Start training")
        train_iter = iter(self.train_loader)

        while self.step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            out = self.train_step(batch)
            self.step += 1
            self.log(out)

            if self.step % self.save_interval == 0 and self.step > 0:
                self.accelerator.save_state()

            if self.val_loader is not None and self.step % self.val_interval == 0 and self.step > 0:
                self.model.eval()
                self.validate()
                self.model.train()

        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(
            self.accelerator.unwrap_model(self.model),
            f"{self.output_dir}/final",
            max_shard_size="5GB",
            safe_serialization=True,
        )
        self.accelerator.end_training()
        total = int(time.monotonic() - self.time)
        mins, secs = divmod(total, 60)
        self.accelerator.print(f"[{mins:02d}:{secs:02d}] Training complete at step {self.step}")

    def train_step(self, batch: LFM2AudioModelInput) -> LFM2AudioModelOutput:
        self.optimizer.zero_grad()

        batch = batch.to(self.accelerator.device)

        with self.accelerator.autocast():
            out = self.model(batch)

        self.accelerator.backward(out.loss)
        self.optimizer.step()
        self.scheduler.step()
        return out

    @torch.no_grad()
    def validate(self) -> None:
        if self.val_loader is None:
            return

        loss_sum = torch.zeros(1, device=self.accelerator.device)
        loss_count = torch.zeros(1, device=self.accelerator.device)

        for batch in self.val_loader:
            batch = batch.to(self.accelerator.device)
            with self.accelerator.autocast():
                out = self.model(batch)
            loss_sum += out.loss.detach()
            loss_count += 1

        global_loss_sum = self.accelerator.reduce(loss_sum, reduction="sum")
        global_loss_count = self.accelerator.reduce(loss_count, reduction="sum")
        mean_val_loss = (global_loss_sum / global_loss_count.clamp_min(1)).item()

        total = int(time.monotonic() - self.time)
        mins, secs = divmod(total, 60)
        self.accelerator.print(
            f"[{mins:02d}:{secs:02d}] VALIDATION: epoch={self.epoch} step={self.step}/{self.max_steps} val_loss={mean_val_loss:.4f}"
        )

    def log(self, model_output: LFM2AudioModelOutput) -> None:
        if self.step > 0 and self.step % self.logging_interval == 0:
            train_loss = self.accelerator.reduce(model_output.loss.detach(), reduction="mean").item()
            lr = self.optimizer.param_groups[0]["lr"]
            total = int(time.monotonic() - self.time)
            mins, secs = divmod(total, 60)
            self.accelerator.print(
                f"[{mins:02d}:{secs:02d}] TRAIN: epoch={self.epoch} step={self.step}/{self.max_steps} "
                f"loss={train_loss:.4f} lr={lr:.3e}"
            )
