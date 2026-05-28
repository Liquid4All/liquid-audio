from __future__ import annotations

import io

import torch
import torchaudio

from liquid_audio.data.types import AudioSegment, ChatMessage, InterleavedSegment, LFM2AudioTrainingSample, TextSegment
from liquid_audio.processor import LFM2AudioProcessor
from liquid_audio.utils import LFMModality, mel2emb_len


class LFM2AudioChatMapper:
    """Map a chat into an LFM2 training sample."""

    def __init__(
        self,
        processor: LFM2AudioProcessor,
        *,
        codebooks: int = 8,
        interleaved_text_tokens: int = 6,
        interleaved_audio_tokens: int = 12,
    ) -> None:
        self.processor = processor
        self.codebooks = codebooks
        self.interleaved_text_tokens = interleaved_text_tokens
        self.interleaved_audio_tokens = interleaved_audio_tokens

    def __call__(self, messages: list[ChatMessage]) -> LFM2AudioTrainingSample:
        text_parts: list[torch.Tensor] = []
        mel_parts: list[torch.Tensor] = []
        audio_out_parts: list[torch.Tensor] = []
        audio_in_lens: list[int] = []
        modality_seq: list[int] = []
        supervision_seq: list[bool] = []

        self._append_text(
            "<|startoftext|>",
            supervised=False,
            text_parts=text_parts,
            modality_seq=modality_seq,
            supervision_seq=supervision_seq,
        )

        for msg in messages:
            self._append_text(
                f"<|im_start|>{msg.role}\n",
                supervised=False,
                text_parts=text_parts,
                modality_seq=modality_seq,
                supervision_seq=supervision_seq,
            )

            for segment in msg.content:
                if isinstance(segment, InterleavedSegment):
                    if msg.role != "assistant":
                        raise ValueError("InterleavedSegment is only supported for assistant messages")
                    self._append_interleaved_out(
                        text=segment.text,
                        audio=segment.audio,
                        text_parts=text_parts,
                        audio_out_parts=audio_out_parts,
                        modality_seq=modality_seq,
                        supervision_seq=supervision_seq,
                    )
                elif isinstance(segment, TextSegment):
                    self._append_text(
                        segment.text,
                        supervised=(msg.role == "assistant"),
                        text_parts=text_parts,
                        modality_seq=modality_seq,
                        supervision_seq=supervision_seq,
                    )
                elif isinstance(segment, AudioSegment):
                    wav, sampling_rate = self._load_audio_bytes(segment.audio)
                    if msg.role == "assistant":
                        self._append_text(
                            "<|audio_start|>",
                            supervised=True,
                            text_parts=text_parts,
                            modality_seq=modality_seq,
                            supervision_seq=supervision_seq,
                        )
                        self._append_audio_out(
                            wav=wav,
                            sampling_rate=sampling_rate,
                            audio_out_parts=audio_out_parts,
                            modality_seq=modality_seq,
                            supervision_seq=supervision_seq,
                        )
                    else:
                        self._append_audio_in(
                            wav=wav,
                            sampling_rate=sampling_rate,
                            mel_parts=mel_parts,
                            audio_in_lens=audio_in_lens,
                            modality_seq=modality_seq,
                            supervision_seq=supervision_seq,
                        )

            self._append_text(
                "<|im_end|>\n",
                supervised=(msg.role == "assistant"),
                text_parts=text_parts,
                modality_seq=modality_seq,
                supervision_seq=supervision_seq,
            )

        text = torch.cat(text_parts, dim=0).unsqueeze(0).to(dtype=torch.long)
        audio_in = torch.cat(mel_parts, dim=1) if mel_parts else torch.empty((128, 0), dtype=torch.float32)
        audio_in_lens_t = torch.tensor(audio_in_lens, dtype=torch.long)
        audio_out = (
            torch.cat(audio_out_parts, dim=1).to(dtype=torch.long)
            if audio_out_parts
            else torch.empty((self.codebooks, 0), dtype=torch.long)
        )
        modality_flag = torch.tensor(modality_seq, dtype=torch.long).unsqueeze(0)
        supervision_mask = torch.tensor(supervision_seq, dtype=torch.bool).unsqueeze(0)

        return LFM2AudioTrainingSample(
            text=text,
            audio_in=audio_in,
            audio_in_lens=audio_in_lens_t,
            audio_out=audio_out,
            modality_flag=modality_flag,
            supervision_mask=supervision_mask,
        )

    def _append_interleaved_out(
        self,
        *,
        text: str,
        audio: bytes,
        text_parts: list[torch.Tensor],
        audio_out_parts: list[torch.Tensor],
        modality_seq: list[int],
        supervision_seq: list[bool],
    ) -> None:
        text_tokens = self.processor.text.encode(f"{text}<|text_end|>", add_special_tokens=False, return_tensors="pt").squeeze(
            0
        )
        text_parts.append(text_tokens)

        wav, sampling_rate = self._load_audio_bytes(audio)
        audio_out = self._encode_audio_out(wav=wav, sampling_rate=sampling_rate)
        audio_out_parts.append(audio_out)

        n_text = self.interleaved_text_tokens
        n_audio = self.interleaved_audio_tokens
        text_left = int(text_tokens.shape[0])
        audio_left = int(audio_out.shape[1])
        while text_left > 0 or audio_left > 0:
            take_text = min(n_text, text_left)
            if take_text > 0:
                modality_seq.extend([int(LFMModality.TEXT)] * take_text)
                supervision_seq.extend([True] * take_text)
                text_left -= take_text

            take_audio = min(n_audio, audio_left)
            if take_audio > 0:
                modality_seq.extend([int(LFMModality.AUDIO_OUT)] * take_audio)
                supervision_seq.extend([True] * take_audio)
                audio_left -= take_audio

    def _append_text(
        self,
        text: str,
        *,
        supervised: bool,
        text_parts: list[torch.Tensor],
        modality_seq: list[int],
        supervision_seq: list[bool],
    ) -> None:
        text_tokens = self.processor.text.encode(text, add_special_tokens=False, return_tensors="pt").squeeze(0)
        text_parts.append(text_tokens)
        n = int(text_tokens.shape[0])
        modality_seq.extend([int(LFMModality.TEXT)] * n)
        supervision_seq.extend([supervised] * n)

    def _append_audio_in(
        self,
        *,
        wav: torch.Tensor,
        sampling_rate: int,
        mel_parts: list[torch.Tensor],
        audio_in_lens: list[int],
        modality_seq: list[int],
        supervision_seq: list[bool],
    ) -> None:
        wav = wav.to(device=self.processor.device, dtype=torch.float32)
        if sampling_rate != 16_000:
            wav = torchaudio.functional.resample(wav, sampling_rate, 16_000)

        wav_len = torch.tensor([wav.shape[-1]], device=wav.device, dtype=torch.long)
        mel, mel_len = self.processor.audio(wav, wav_len)
        cur_len = int(mel_len[0].item())
        cur_mel = mel[0, :, :cur_len].to(dtype=torch.float32).cpu()

        mel_parts.append(cur_mel)
        audio_in_lens.append(cur_len)

        n_emb = int(mel2emb_len(cur_len))
        modality_seq.extend([int(LFMModality.AUDIO_IN)] * n_emb)
        supervision_seq.extend([False] * n_emb)

    def _append_audio_out(
        self,
        *,
        wav: torch.Tensor,
        sampling_rate: int,
        audio_out_parts: list[torch.Tensor],
        modality_seq: list[int],
        supervision_seq: list[bool],
    ) -> None:
        codes = self._encode_audio_out(wav=wav, sampling_rate=sampling_rate)

        audio_out_parts.append(codes)
        n = int(codes.shape[1])
        modality_seq.extend([int(LFMModality.AUDIO_OUT)] * n)
        supervision_seq.extend([True] * n)

    def _encode_audio_out(self, *, wav: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        wav = wav.to(device=self.processor.device, dtype=torch.float32)
        mimi_sample_rate = int(self.processor.mimi.sample_rate)
        if sampling_rate != mimi_sample_rate:
            wav = torchaudio.functional.resample(wav, sampling_rate, mimi_sample_rate)

        codes = self.processor.mimi.encode(wav.unsqueeze(0))[0].cpu()
        codes = codes[: self.codebooks].to(dtype=torch.long)
        end_of_audio = torch.full((self.codebooks, 1), 2048, dtype=torch.long)
        return torch.cat([codes, end_of_audio], dim=1)

    @staticmethod
    def _load_audio_bytes(audio: bytes) -> tuple[torch.Tensor, int]:
        with io.BytesIO(audio) as stream:
            wav, sampling_rate = torchaudio.load(stream)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sampling_rate
