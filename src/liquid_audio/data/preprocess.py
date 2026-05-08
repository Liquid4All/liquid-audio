from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import datasets
from datasets import Features, Sequence, Value

from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.types import ChatMessage


def preprocess_dataset(
    data: Iterable[list[ChatMessage]],
    output_path: str | Path,
    mapper: LFM2AudioChatMapper,
    max_context_length: int = -1,
) -> None:
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=False)

    features = Features(
        {
            "text": Sequence(Sequence(Value("int64"))),
            "audio_in": Sequence(Sequence(Value("float32"))),
            "audio_in_lens": Sequence(Value("int64")),
            "audio_out": Sequence(Sequence(Value("int64"))),
            "modality_flag": Sequence(Sequence(Value("int64"))),
            "supervision_mask": Sequence(Sequence(Value("bool"))),
        }
    )

    def generator():
        for i, messages in enumerate(data):
            sample = mapper(messages)
            sample_len = int(sample.modality_flag.shape[-1])
            if 0 <= max_context_length < sample_len:
                print(f"WARNING: skipping sample {i} with {sample_len} tokens (max_context_length={max_context_length})")
                continue
            yield {
                "text": sample.text.tolist(),
                "audio_in": sample.audio_in.tolist(),
                "audio_in_lens": sample.audio_in_lens.tolist(),
                "audio_out": sample.audio_out.tolist(),
                "modality_flag": sample.modality_flag.tolist(),
                "supervision_mask": sample.supervision_mask.tolist(),
            }

    preprocessed = datasets.Dataset.from_generator(generator, features=features)
    preprocessed.save_to_disk(out_dir)
