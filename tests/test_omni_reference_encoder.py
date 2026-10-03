"""The native encoder's fixed CPU cache must survive a direct-device load."""

import pytest
import torch
from torch import nn

from liquid_audio.model.conformer.mha import RelPositionalEncoding
from liquid_audio.model.lfm2_audio import LFM2AudioModel


def test_reference_alignment_restores_cpu_float32_positions_without_changing_weights():
    encoder = nn.Module()
    encoder.self_attention_model = "rel_pos"
    encoder.max_audio_length = 8
    encoder.register_parameter("weight", nn.Parameter(torch.randn(4, dtype=torch.bfloat16)))
    encoder.pos_enc = RelPositionalEncoding(4, 0.0)
    encoder.pos_enc.extend_pe(8, "cpu", torch.float32)
    expected = encoder.pos_enc.pe.to(torch.bfloat16).clone()
    encoder.pos_enc.pe = torch.zeros_like(expected)
    encoder.use_pytorch_sdpa = False
    encoder.attention = nn.Module()
    encoder.attention.use_pytorch_sdpa = False
    encoder.sync_max_audio_length = True
    weights = encoder.weight.detach().clone()
    model = LFM2AudioModel.__new__(LFM2AudioModel)
    nn.Module.__init__(model)
    model.conformer = encoder
    model.configure_omni_reference_encoder()
    assert torch.equal(encoder.pos_enc.pe, expected)
    assert torch.equal(encoder.weight, weights)
    assert encoder.pos_enc.pe.dtype == weights.dtype
    assert encoder.use_pytorch_sdpa and encoder.attention.use_pytorch_sdpa
    assert not encoder.sync_max_audio_length


def test_reference_alignment_rejects_other_position_protocols():
    model = LFM2AudioModel.__new__(LFM2AudioModel)
    nn.Module.__init__(model)
    model.conformer = nn.Module()
    model.conformer.self_attention_model = "abs_pos"
    with pytest.raises(ValueError, match="relative positional"):
        model.configure_omni_reference_encoder()
