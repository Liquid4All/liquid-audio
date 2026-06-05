# Liquid Audio - Japanese Speech-to-Speech models

### Finetuning

To fine-tune our Japanese model on your own data in interleaved generation mode, instantiate the `LFM2AudioChatMapper` class with `interleaved_text_tokens=6` and `interleaved_audio_tokens=9`. These values reflect the predefined Japanese interleaving ratio of 6 text tokens to 9 audio tokens, based on tokenization statistics.

### Multi-turn, multi-modal chat

<details>

<summary>Conversation transcript</summary>

**User**

https://github.com/user-attachments/assets/3c8a2f21-d8e5-4b52-9d8e-9604c5cbb1e6

**Assistant**

こんにちは。私はリキッドリリーと申します。質問に答えたり、アドバイスを提供したりするためのAIボイスアシスタントです。リアルタイムでさまざまな言語タスクをお手伝いするよう設計されています。

https://github.com/user-attachments/assets/345d0837-4f90-4e78-910e-39cd95321cdd

**User**

富士山の高さは何メートルですか。

**Assistant**

富士山の高さは約3,776メートルです。

https://github.com/user-attachments/assets/f0efb661-e01f-4bd2-83fb-77e9e0d9f8d7

</details>

```python
import torch
import soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-JP"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()

chat.new_turn("user")
wav, sampling_rate = sf.read("assets/question_jp.wav", dtype="float32")
wav = torch.from_numpy(wav).unsqueeze(0)
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

# output: こんにちは。私はリキッドリリーと申します。質問に答えたり、アドバイスを提供したりするためのAIボイスアシスタントです。リアルタイムでさまざまな言語タスクをお手伝いするよう設計されています。

# Detokenize audio, removing the last "end-of-audio" codes
# Mimi returns audio at 24kHz
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
sf.write("answer_jp1.wav", waveform.cpu()[0], 24_000)

# Append newly generated tokens to chat history
chat.append(
    text = torch.stack(text_out, 1),
    audio_out = torch.stack(audio_out, 1),
    modality_flag = torch.tensor(modality_out),
)
chat.end_turn()

# Start new turn
chat.new_turn("user")
chat.add_text("富士山の高さは何メートルですか。")
chat.end_turn()

chat.new_turn("assistant")

# Generate second turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
    else:
        audio_out.append(t)

# output: 富士山の高さは約3,776メートルです。

# Detokenize second turn audio, removing the last "end-of-audio" codes
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
sf.write("answer_jp2.wav", waveform.cpu()[0], 24_000)
```


### ASR

For ASR, use the fixed system prompt `Perform ASR in japanese.` instead.

<details>

<summary>Input audio snippet</summary>

https://github.com/user-attachments/assets/be2438d9-b7c7-41ab-bbc9-cd7c505083e4

**Model output**: この度は弊社の確認不足により多大なご迷惑をおかけしましたことを深くお詫び申し上げます。今後はこのようなことが二度と起こらないよう社内のチェック体制を徹底してまいります。

</details>

```python
import torch
import soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-JP"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Perform ASR in japanese.")
chat.end_turn()

chat.new_turn("user")
wav, sampling_rate = sf.read("assets/asr_jp.wav", dtype="float32")
wav = torch.from_numpy(wav).unsqueeze(0)
chat.add_audio(wav, sampling_rate)
chat.end_turn()

chat.new_turn("assistant")

# Generate text
for t in model.generate_sequential(**chat, max_new_tokens=512):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)

# Output: この度は弊社の確認不足により多大なご迷惑をおかけしましたことを深くお詫び申し上げます。今後はこのようなことが二度と起こらないよう社内のチェック体制を徹底してまいります。
```

### TTS

For TTS, use the fixed system prompt `Perform TTS in japanese.` instead.

<details>

<summary>TTS Sample</summary>

**System prompt**: Perform TTS in japanese.

**Input sentence**: 先週ご相談いただいた新しいプロジェクトの件ですが、社内で検討した結果、ぜひ前向きに進めさせていただきたいと考えております。つきましては、具体的なスケジュールについて一度お打ち合わせの機会をいただけますでしょうか。

**Output audio**

https://github.com/user-attachments/assets/ee95f4db-141d-44a5-b592-eb2b6d77a45c

</details>

```python
import torch
import soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Load models
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-JP"

processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# Set up inputs for the model
chat = ChatState(processor)

chat.new_turn("system")
chat.add_text("Perform TTS in japanese.")
chat.end_turn()

chat.new_turn("user")
chat.add_text("先週ご相談いただいた新しいプロジェクトの件ですが、社内で検討した結果、ぜひ前向きに進めさせていただきたいと考えております。つきましては、具体的なスケジュールについて一度お打ち合わせの機会をいただけますでしょうか。")
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
sf.write("tts_jp.wav", waveform.cpu()[0], 24_000)
```
