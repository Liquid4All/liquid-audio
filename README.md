# Liquid Audio - Speech-to-Speech models

We present LFM2-Audio-1.5B, [Liquid AI](https://www.liquid.ai/)'s first end-to-end audio foundation model. Built with low-latency in mind, the lightweight [LFM2](https://huggingface.co/LiquidAI/LFM2-1.2B) backbone enables real time speech-to-speech conversations without sacrificing quality.

LFM2-Audio supports two generation modes, interleaved and sequential, to maximize performance and quality across different tasks. Interleaved generation outputs text and audio tokens in a fixed interleaved pattern. This approach minimizes time to first audio output and number of tokens generated, making it ideal for naturally flowing real-time speech-to-speech interactions on resource constrained devices. Sequential generation mode, where the model decides when to switch modalities via special tokens, is suitable for non-conversational tasks, such as speech-to-text (ASR) or text-to-speech (TTS).

### Updates
- [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) is released! This model is based on the stronger LFM2.5-1.2B base, and comes with a lightning fast LFM2 based audio detokenizer, stronger ASR, and better TTS voices. To use the new detokenizer, simply use `processor.decode`, see the examples below for more details. For the improved TTS voices, see the [TTS](#tts) section.

## Installation
The package can be installed via `pip`
```bash
pip install liquid-audio
pip install "liquid-audio [demo]" # optional, to install demo dependencies
pip install flash-attn --no-build-isolation  # optional, to use flash attention 2. Will fallback to torch SDPA if not installed
```

## Usage
Generation is handled by two generation modes, interleaved and sequential, accessible from the methods `LFM2AudioModel.generate_interleaved` and `LFM2AudioModel.generate_sequential` respectively. Both are generators that yield `torch.Tensor`s. Text tokens are represented by tensors with 1 entry, and audio tokens are tensors with 8 entries, corresponding to 8 [Mimi](https://huggingface.co/docs/transformers/en/model_doc/mimi) codebooks.

The `LFM2AudioModel` class operates on tokens only. The `LFM2AudioProcessor` class is used convert between tokens and data. For text, this means the conversion from string to tokens and back. For audio inputs, this handles the conversion of waveforms to log-mel features, and for audio outputs, this handles the detokenization of audio tokens to waveform.

To facilitate the creation of inputs for the generation methods and to apply the correct chat templates, use the `ChatState` helper class. See examples below for usage instructions.

### Gradio demo
To use the demo interface, make sure to install the extra dependencies in the `[demo]` group, e.g.
```bash
pip install "liquid-audio [demo]"
```
To launch the demo, use the command `liquid-audio-demo` on the terminal. The demo interface will be available via the url http://localhost:7860.

### Multi-turn, multi-modal chat
For multi-turn chat with text and audio output, we use interleaved generation. The system prompt should be set to `Respond with interleaved text and audio.`. Here we use audio as the first user turn, and text as the second one

<details>

<summary>Conversation transcript</summary>

**User**

https://github.com/user-attachments/assets/e2ffb8c3-c84c-4460-9cb8-f95a13b6eec6

**Assistant**

Sure! How about "Handcrafted Woodworking, Precision Made for You"? Another option could be "Quality Woodworking, Quality Results." If you want something more personal, you might try "Your Woodworking Needs, Our Expertise."

https://github.com/user-attachments/assets/019664b5-3480-4801-b05a-bd62ddcb8d3e

**User**

My business specialized in chairs, can you give me something related to that?

**Assistant**

Sure thing! How about “Comfortable Chairs, Crafted with Care” or “Elegant Seats, Handcrafted for You”? Let me know if you’d like a few more options.

https://github.com/user-attachments/assets/d0d054b2-6d1d-49fb-94df-4aa0b6641990

</details>

```python
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()

chat.new_turn("user")
wav, sampling_rate = torchaudio.load("assets/question.wav")
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# Generate text and audio tokens.
text_out: list[torch.Tensor] = []
audio_out: list[torch.Tensor] = []
modality_out: list[LFMModality] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
        text_out.append(t)
        modality_out.append(LFMModality.TEXT)
    else:
        audio_out.append(t)
        modality_out.append(LFMModality.AUDIO_OUT)

# output: Sure! How about "Handcrafted Woodworking, Precision Made for You"? Another option could be "Quality Woodworking, Quality Results." If you want something more personal, you might try "Your Woodworking Needs, Our Expertise."

# Detokenize audio, removing the last "end-of-audio" codes
# Mimi returns audio at 24kHz
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
torchaudio.save("answer1.wav", waveform.cpu(), 24_000)

# Append newly generated tokens to chat history
chat.append(
    text = torch.stack(text_out, 1),
    audio_out = torch.stack(audio_out, 1),
    modality_flag = torch.tensor(modality_out),
)
chat.end_turn()

# Start new turn
chat.new_turn("user")
chat.add_text("My business specialized in chairs, can you give me something related to that?")
chat.end_turn()

chat.new_turn("assistant")

# Generate second turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
    else:
        audio_out.append(t)

# output: Sure thing! How about “Comfortable Chairs, Crafted with Care” or “Elegant Seats, Handcrafted for You”? Let me know if you’d like a few more options.

# Detokenize second turn audio, removing the last "end-of-audio" codes
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
torchaudio.save("answer2.wav", waveform.cpu(), 24_000)
```


### ASR
For ASR, we use sequential generation, with the fixed system prompt `Perform ASR.`. The output is capitalized and punctuated.

<details>

<summary>Input audio snippet</summary>

https://github.com/user-attachments/assets/b3cc017f-363d-49f3-8e7d-f6db9556900e

**Model output**: The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun.

</details>

```python
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Perform ASR.")
chat.end_turn()

chat.new_turn("user")
wav, sampling_rate = torchaudio.load("assets/asr.wav")
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# Generate text
for t in model.generate_sequential(**chat, max_new_tokens=512):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)

# Output: The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun.
```

### TTS
For TTS, we also use sequential generation. We support four pre-defined voices, which can be selected by choosing one of the four system prompts below
```
Perform TTS. Use the US male voice.
Perform TTS. Use the US female voice.
Perform TTS. Use the UK male voice.
Perform TTS. Use the UK female voice.
```

<details>

<summary>TTS Sample</summary>

**System prompt**: Perform TTS. Use the UK male voice.

**Input sentence**: What is this obsession people have with books? They put them in their houses—like they're trophies. What do you need it for after you read it?

**Output audio**

https://github.com/user-attachments/assets/8d57c184-b92e-4e1a-983b-d1f9d16d0d92

</details>

```python
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Perform TTS. Use the UK male voice.")
chat.end_turn()

chat.new_turn("user")
chat.add_text("What is this obsession people have with books? They put them in their houses—like they're trophies. What do you need it for after you read it?")
chat.end_turn()

chat.new_turn("assistant")

# Generate text
audio_out: list[torch.Tensor] = []
for t in model.generate_sequential(**chat, max_new_tokens=512, audio_temperature = 0.8, audio_top_k=64):
    if t.numel() > 1:
        audio_out.append(t)

# Detokenize audio
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
torchaudio.save("tts.wav", waveform.cpu(), 24_000)
```


## License
The code in this repository and associated weights are licensed under the [LFM Open License v1.0](LICENSE).

The code for the audio encoder is based on [Nvidia NeMo](https://github.com/NVIDIA-NeMo/NeMo/tree/main), licensed under [Apache 2.0](https://github.com/NVIDIA-NeMo/NeMo/blob/294ddff187f68c055d87ffe9400e65975b38693d/LICENSE), and the [canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash) checkpoint, licensed under [CC-BY 4.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-4.0.md). To simplify dependency resolution, we also ship the Python code of [Kyutai Mimi](https://github.com/kyutai-labs/moshi), licensed under the [MIT License](https://github.com/kyutai-labs/moshi/blob/aee53fc0fc0119e4d7343e5ea4dd6ddafd7f09c4/LICENSE-MIT).
