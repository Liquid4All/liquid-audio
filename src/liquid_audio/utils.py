from enum import IntEnum, auto
from functools import cache
from pathlib import Path

import torch
from loguru import logger
from huggingface_hub import snapshot_download


class LFMModality(IntEnum):
    TEXT = auto()
    AUDIO_IN = auto()
    AUDIO_OUT = auto()


def mel2emb_len[T: (int, torch.Tensor)](l: T) -> T:
    """Convert log-mel feature length to final LFM embedding length

    This is just floor division.
    Note: smallest mel-length for encoder is 9
    """
    return -(l // -8)


def emb2mel_len[T: (int, torch.Tensor)](l: T) -> T:
    """Convert LFM embedding length to log-mel feature length

    Note: this is an upper bound.
    """
    return l * 8


def module_exists(name: str) -> bool:
    import importlib.util

    spec = importlib.util.find_spec(name)

    return spec is not None


@cache
def get_model_dir(
    repo_id: str | Path,
    *,
    revision: str | None = None,
) -> Path:
    cache_path: Path
    if isinstance(repo_id, str):
        cache_path = Path(snapshot_download(repo_id, revision=revision))
    else:
        if revision is not None:
            logger.error("cannot use `revision` kwarg when given a path")
            raise RuntimeError("cannot use `revision` kwarg when given a path")
        cache_path = repo_id

    return cache_path
