import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, assert_never

import torch
import torchaudio
from transformers import AutoTokenizer, Lfm2Config, PreTrainedTokenizer

from liquid_audio import moshi
from liquid_audio.detokenizer import LFM2AudioDetokenizer
from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor
from liquid_audio.moshi.models.compression import MimiModel
from liquid_audio.utils import LFMModality, get_model_dir, mel2emb_len


@dataclass(kw_only=True)
class PreprocessorConfig:
    sample_rate: int
    normalize: str
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int
    log: bool
    frame_splicing: int
    dither: float
    pad_to: int
    pad_value: float


class LFM2AudioProcessor:
    """Container for LFM2-Audio text and audio processors"""

    def __init__(
        self,
        text_tokenizer_path: str,
        audio_processor_config: PreprocessorConfig,
        mimi_weights_path: str | None = None,
        detokenizer_path: str | None = None,
        name: str | None = None,
    ) -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        self.audio_processor = AudioToMelSpectrogramPreprocessor(**asdict(audio_processor_config)).eval()
        self.mimi_weights_path = mimi_weights_path
        self.detokenizer_path = detokenizer_path

        self.name = name

        self._mimi: MimiModel | None = None
        self._audio_detokenizer: LFM2AudioDetokenizer | None = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cuda",
    ) -> Self:
        cache_path = get_model_dir(repo_id, revision=revision)
        with (cache_path / "config.json").open() as f:
            config = json.load(f)

        mimi_ckpt = cache_path / "tokenizer-e351c8d8-checkpoint125.safetensors"
        mimi_weights_path = str(mimi_ckpt) if mimi_ckpt.exists() else None

        detok_ckpt = cache_path / "audio_detokenizer"
        detokenizer_weights_path = str(detok_ckpt) if detok_ckpt.exists() else None

        return cls(
            text_tokenizer_path=str(cache_path),
            audio_processor_config=PreprocessorConfig(**config["preprocessor"]),
            mimi_weights_path=mimi_weights_path,
            detokenizer_path=detokenizer_weights_path,
            name=str(repo_id),
        ).to(device)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> Self:
        self.audio_processor.to(device=device, dtype=dtype)
        return self

    def eval(self) -> Self:
        self.audio_processor.eval()
        return self

    def train(self) -> Self:
        self.audio_processor.train()
        return self

    @property
    def text(self) -> PreTrainedTokenizer:
        return self.text_tokenizer

    @property
    def audio(self) -> AudioToMelSpectrogramPreprocessor:
        return self.audio_processor

    @property
    def mimi(self) -> MimiModel:
        if self.mimi_weights_path is None:
            if self.name is None:
                msg = "expected `mimi_weights_path` to be specified."
            else:
                msg = f"model {self.name} does not provide Mimi weights, use {type(self).__name__}.decode instead."
            raise AttributeError(msg)

        if self._mimi is None:
            from safetensors.torch import load_file

            mimi_model = moshi.models.loaders.get_mimi(None, device=self.device)
            mimi_weights = load_file(self.mimi_weights_path, device=str(self.device))
            mimi_model.load_state_dict(mimi_weights, strict=True)

            self._mimi = mimi_model

        return self._mimi

    @property
    def audio_detokenizer(self) -> LFM2AudioDetokenizer:
        if self.detokenizer_path is None:
            if self.name is None:
                msg = "expected `detokenizer_weights_path` to be specified."
            else:
                msg = (
                    f"model {self.name} does not provide LFM based audio detokenizer, use {type(self).__name__}.mimi instead."
                )
            raise AttributeError(msg)

        if self._audio_detokenizer is None:
            detok_config_path = Path(self.detokenizer_path) / "config.json"
            detok_config = Lfm2Config.from_pretrained(detok_config_path)

            # Make llama.cpp config compatible with transformers Lfm2Model
            def rename_layer(
                layer: Literal["conv", "sliding_attention", "full_attention"],
            ) -> Literal["conv", "full_attention"]:
                match layer:
                    case "conv" | "full_attention":
                        return layer
                    case "sliding_attention":
                        return "full_attention"
                    case _:
                        assert_never(layer)

            assert isinstance(detok_config.layer_types, list)
            detok_config.layer_types = [rename_layer(layer) for layer in detok_config.layer_types]  # type: ignore[arg-type]

            detok = LFM2AudioDetokenizer(detok_config).eval().cuda()

            detok_weights_path = Path(self.detokenizer_path) / "model.safetensors"
            from safetensors.torch import load_file

            detok_weights = load_file(detok_weights_path)
            detok.load_state_dict(detok_weights)

            detok.eval()

            self._audio_detokenizer = detok

        return self._audio_detokenizer

    @torch.no_grad()
    def decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Detokenize audio codes into waveform with LFM2 based detokenizer

        Args:
            audio_codes: (1, 8, T) shaped integer tensor, with values in [0, 2047]
        Returns:
            waveform: (1, T') float tensor, 24kHz mono waveform
        """
        if torch.any(audio_codes >= 2048) or torch.any(audio_codes < 0):
            raise RuntimeError("expected audio codes in range ")

        return self.audio_detokenizer(audio_codes)

    @property
    def device(self) -> torch.device:
        return next(self.audio.buffers()).device


class ChatState(Mapping):
    model_inputs: ClassVar[list[str]] = ["text", "audio_in", "audio_in_lens", "audio_out", "modality_flag"]

    def __init__(self, processor: LFM2AudioProcessor, *, codebooks: int = 8, dtype: torch.dtype = torch.bfloat16) -> None:
        self.proc = processor
        self.codebooks = codebooks
        self.dtype = dtype

        start = "<|startoftext|>"

        self.text = self.proc.text.encode(start, add_special_tokens=False, return_tensors="pt").to(self.device)
        self.audio_in = torch.empty((128, 0), device=self.device, dtype=self.dtype)
        self.audio_in_lens = torch.empty((0,), device=self.device, dtype=torch.long)
        self.audio_out = self.text.new_empty((self.codebooks, 0))

        self.modality_flag = torch.full_like(self.text, LFMModality.TEXT)

    def __repr__(self) -> str:
        return f"ChatState(text_tok: {self.text.shape[1]}, audio_in: {self.audio_in.shape[1]}, audio_out: {self.audio_out.shape[1]})"

    # Mapping abstract method implementations
    def __getitem__(self, name: str) -> Any:
        if name not in self.model_inputs:
            raise KeyError(f"expected one of {self.model_inputs}, got {name}.")
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.model_inputs)

    def __len__(self) -> int:
        return len(self.model_inputs)

    @property
    def device(self) -> torch.device:
        return self.proc.device

    def add_text(self, text: str) -> None:
        new_text = self.proc.text.encode(text, add_special_tokens=False, return_tensors="pt").to(self.device)
        new_mod = self.modality_flag.new_full(new_text.shape, LFMModality.TEXT)
        self.text = torch.cat([self.text, new_text], 1)
        self.modality_flag = torch.cat([self.modality_flag, new_mod], 1)

    def add_audio(self, wave: torch.Tensor, sampling_rate: int) -> None:
        assert len(wave.shape) == 2
        assert wave.shape[0] == 1

        device = next(self.proc.audio.buffers()).device

        wave = wave.to(device=device)
        wave = torchaudio.functional.resample(wave, sampling_rate, 16_000)
        length = torch.tensor([wave.shape[1]], dtype=torch.long, device=wave.device)

        mel, _ = self.proc.audio(wave, length)

        new_audio_in = mel[0].to(self.dtype)
        new_mod = self.modality_flag.new_tensor(
            [
                [
                    *[LFMModality.AUDIO_IN] * mel2emb_len(new_audio_in.shape[1]),
                ]
            ]
        )
        new_audio_in_lens = self.audio_in_lens.new_tensor([new_audio_in.shape[1]])

        self.audio_in = torch.cat([self.audio_in, new_audio_in], 1)
        self.modality_flag = torch.cat([self.modality_flag, new_mod], 1)
        self.audio_in_lens = torch.cat([self.audio_in_lens, new_audio_in_lens])

    def end_turn(self) -> None:
        self.add_text("<|im_end|>\n")

    def new_turn(self, role: Literal["system", "user", "assistant"]) -> None:
        self.add_text(f"<|im_start|>{role}\n")

    def append(self, text: torch.Tensor, audio_out: torch.Tensor, modality_flag: torch.Tensor):
        if len(modality_flag.shape) == 1:
            modality_flag = modality_flag.unsqueeze(0)

        assert len(text) == 1
        assert len(audio_out) == self.codebooks
        assert len(modality_flag) == 1
        assert modality_flag.shape[1] == text.shape[1] + audio_out.shape[1]

        self.text = torch.cat([self.text, text.to(self.device)], 1)
        self.audio_out = torch.cat([self.audio_out, audio_out.to(self.device)], 1)
        self.modality_flag = torch.cat([self.modality_flag, modality_flag.to(self.device)], 1)
