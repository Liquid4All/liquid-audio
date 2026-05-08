from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from liquid_audio.data.types import LFM2AudioModelInput, LFM2AudioRow
from liquid_audio.utils import LFMModality


class LFM2DataLoader(TorchDataset[LFM2AudioRow]):
    def __init__(
        self,
        dataset_path: str,
        context_length: int = 4096,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.context_length = context_length
        self.dataset: Dataset = load_from_disk(self.dataset_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> LFM2AudioRow:
        row = self.dataset[idx]

        text = torch.as_tensor(row["text"], dtype=torch.long)
        audio_in = torch.as_tensor(row["audio_in"], dtype=torch.float32)
        audio_in_lens = torch.as_tensor(row["audio_in_lens"], dtype=torch.long)
        audio_out = torch.as_tensor(row["audio_out"], dtype=torch.long)
        modality = torch.as_tensor(row["modality_flag"], dtype=torch.long)
        supervision = torch.as_tensor(row["supervision_mask"], dtype=torch.bool)

        pad_len = self.context_length - int(modality.shape[1])
        if pad_len < 0:
            raise ValueError(
                f"sample at index {idx} has {modality.shape[1]} tokens, "
                f"which is longer than context_length={self.context_length}"
            )

        text = F.pad(text, (0, pad_len))
        modality = F.pad(modality, (0, pad_len), value=int(LFMModality.TEXT))
        supervision = F.pad(supervision, (0, pad_len), value=False)

        return LFM2AudioRow(
            text=text,
            audio_in=audio_in,
            audio_in_lens=audio_in_lens,
            audio_out=audio_out,
            modality_flag=modality,
            supervision_mask=supervision,
        )


def lfm2_collator(batch: list[LFM2AudioRow]) -> LFM2AudioModelInput:
    audio_in = torch.cat([row.audio_in for row in batch], dim=1)
    audio_in_lens = torch.cat([row.audio_in_lens for row in batch], dim=0)

    text = torch.cat([row.text for row in batch], dim=1)
    audio_out = torch.cat([row.audio_out for row in batch], dim=1)

    modality_flag = torch.cat([row.modality_flag for row in batch], dim=0)
    supervision_mask = torch.cat([row.supervision_mask for row in batch], dim=0)

    return LFM2AudioModelInput(
        text=text,
        audio_in=audio_in,
        audio_in_lens=audio_in_lens,
        audio_out=audio_out,
        modality_flag=modality_flag,
        supervision_mask=supervision_mask,
    )
