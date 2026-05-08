from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class TextSegment:
    kind: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True, slots=True)
class AudioSegment:
    kind: Literal["audio"] = "audio"
    audio: bytes = b""


@dataclass(frozen=True, slots=True)
class InterleavedSegment:
    kind: Literal["interleaved"] = "interleaved"
    text: str = ""
    audio: bytes = b""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatMessage:
    role: Literal["user", "system", "assistant"]
    content: list[ChatContentSegment]


ChatContentSegment = TextSegment | AudioSegment | InterleavedSegment


@dataclass(slots=True, kw_only=True)
class LFM2AudioTrainingSample:
    # Pre-packed data item
    text: torch.Tensor
    audio_in: torch.Tensor
    audio_in_lens: torch.Tensor
    audio_out: torch.Tensor
    modality_flag: torch.Tensor
    supervision_mask: torch.Tensor


@dataclass(slots=True, kw_only=True)
class LFM2AudioRow:
    # Single padded row from dataset
    text: torch.Tensor
    audio_in: torch.Tensor
    audio_in_lens: torch.Tensor
    audio_out: torch.Tensor
    modality_flag: torch.Tensor
    supervision_mask: torch.Tensor


@dataclass(slots=True, kw_only=True)
class LFM2AudioModelInput:
    # Batched model input assembled by collate
    text: torch.Tensor
    audio_in: torch.Tensor
    audio_in_lens: torch.Tensor
    audio_out: torch.Tensor
    modality_flag: torch.Tensor
    supervision_mask: torch.Tensor

    def to(self, device: torch.device | str) -> LFM2AudioModelInput:
        return LFM2AudioModelInput(
            text=self.text.to(device),
            audio_in=self.audio_in.to(device),
            audio_in_lens=self.audio_in_lens.to(device),
            audio_out=self.audio_out.to(device),
            modality_flag=self.modality_flag.to(device),
            supervision_mask=self.supervision_mask.to(device),
        )
