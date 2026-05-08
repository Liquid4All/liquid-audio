from __future__ import annotations

import argparse
from pathlib import Path

from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--data", default="data/jenny_tts/train")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="tmp")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at {dataset_path}. Run preprocessing before training.")

    train_data = LFM2DataLoader(dataset_path=str(dataset_path), context_length=args.context_length)

    trainer = Trainer(
        model_id=args.model_id,
        train_data=train_data,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=args.num_workers,
        logging_interval=10,
        save_interval=500,
        val_interval=100,
        output_dir=args.output_dir,
    )
    trainer.train()
