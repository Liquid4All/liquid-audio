"""Text/proactive prefixes must bypass the speech encoder; real speech still uses it."""

from types import SimpleNamespace

import torch
from torch import nn

from liquid_audio.model.lfm2_audio import LFM2AudioModel


class Encoder(nn.Module):
    def forward(self, signal, lengths):
        if signal.shape[0] == 0:
            raise RuntimeError("empty speech batch")
        return signal.new_zeros((1, 128, 1)), torch.ones_like(lengths)


def model():
    result = LFM2AudioModel.__new__(LFM2AudioModel)
    nn.Module.__init__(result)
    result.lfm = nn.Module()
    result.lfm.config = SimpleNamespace(hidden_size=4)
    result.lfm.embed_tokens = nn.Embedding(16, 4)
    result.codebooks = 8
    result.audio_vocab_size = 2049
    result.register_buffer("codebook_offsets", torch.arange(8) * 2049)
    result.audio_embedding = nn.Embedding(8 * 2049, 4)
    result.conformer = Encoder()
    result.audio_adapter = nn.Linear(128, 4, bias=False)
    return result


def test_text_only_prefill_uses_text_embeddings_without_empty_encoder_batch():
    instance = model()
    text = torch.tensor([[1, 2]])
    result = instance._prefill(
        text=text,
        audio_in=torch.empty(128, 0),
        audio_in_lens=torch.empty(0, dtype=torch.long),
        audio_out=torch.empty(8, 0, dtype=torch.long),
        modality_flag=torch.ones(1, 2, dtype=torch.long),
    )
    assert torch.equal(result[0], instance.lfm.embed_tokens(text[0]))


def test_speech_prefill_retains_the_encoded_audio_position():
    instance = model()
    text = torch.tensor([[1]])
    result = instance._prefill(
        text=text,
        audio_in=torch.ones(128, 8),
        audio_in_lens=torch.tensor([8]),
        audio_out=torch.empty(8, 0, dtype=torch.long),
        modality_flag=torch.tensor([[1, 2]]),
    )
    assert torch.equal(result[0, 0], instance.lfm.embed_tokens(text[0])[0])
    assert torch.equal(result[0, 1], torch.zeros(4))
