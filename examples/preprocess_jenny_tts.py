from __future__ import annotations

from collections.abc import Iterator

from datasets import Audio, load_dataset

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment


class JennyTTSIterator:
    def __init__(
        self,
        split: str = "train",
        system_prompt: str = "Perform TTS. Use the Irish female voice.",
    ) -> None:
        self.split = split
        self.system_prompt = system_prompt

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        ds = load_dataset("reach-vb/jenny_tts_dataset", split=self.split)
        ds = ds.cast_column("audio", Audio(decode=False))

        for row in ds:
            text = row["transcription"]
            audio = row["audio"]["bytes"]
            yield [
                ChatMessage(role="system", content=[TextSegment(text=self.system_prompt)]),
                ChatMessage(role="user", content=[TextSegment(text=text)]),
                ChatMessage(role="assistant", content=[AudioSegment(audio=audio)]),
            ]


if __name__ == "__main__":
    processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B", device="cuda").eval()
    mapper = LFM2AudioChatMapper(processor)
    data = JennyTTSIterator()
    preprocess_dataset(
        data=data,
        output_path="data/jenny_tts/train",
        mapper=mapper,
        max_context_length=256,  # skips 163 JennyTTS samples
    )
