"""A checkpoint written by the Trainer must load back with `LFM2AudioModel.from_pretrained`."""

import json
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import save_model
from transformers import Lfm2Config

from liquid_audio import LFM2AudioModel, LFMModality
from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.model.conformer.encoder import ConformerEncoderConfig
from liquid_audio.model.lfm2_audio import DepthformerConfig, LFM2AudioConfig
from liquid_audio.processor import PreprocessorConfig
from liquid_audio.trainer import Trainer
from liquid_audio.utils import get_model_dir

# A tiny random model: no download, runs on CPU in seconds.
CONFIG = {
    "architectures": ["LFM2AudioModel"],
    "codebooks": 8,
    "tie_audio_embeddings": False,
    "semantic_codebook_factor": 100,
    "codebook_weight": "log",
    "interleaved_n_text": 6,
    "interleaved_n_audio": 12,
    "preprocessor": {
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "features": 128,
        "n_fft": 512,
        "log": True,
        "frame_splicing": 1,
        "dither": 1e-05,
        "pad_to": 0,
        "pad_value": 0.0,
    },
    "encoder": {
        "feat_in": 128,
        "feat_out": -1,
        "n_layers": 1,
        "d_model": 32,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 16,
        "causal_downsampling": False,
        "reduction": None,
        "reduction_position": None,
        "reduction_factor": 1,
        "ff_expansion_factor": 2,
        "self_attention_model": "rel_pos",
        "n_heads": 2,
        "att_context_size": [-1, -1],
        "xscaling": False,
        "untie_biases": True,
        "pos_emb_max_len": 512,
        "conv_kernel_size": 9,
        "conv_norm_type": "batch_norm",
        "conv_context_size": None,
        "dropout": 0.0,
        "dropout_pre_encoder": 0.0,
        "dropout_emb": 0,
        "dropout_att": 0.0,
    },
    "lfm": {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 2048,
        "conv_L_cache": 3,
        "eos_token_id": 7,
        "block_auto_adjust_ff_dim": False,
    },
    "depthformer": {"layers": 1, "dim": 64, "tie": True},
}


def tiny_model() -> LFM2AudioModel:
    config = json.loads(json.dumps(CONFIG))
    conf = LFM2AudioConfig(
        lfm=Lfm2Config(**config.pop("lfm")),
        encoder=ConformerEncoderConfig(**config.pop("encoder")),
        depthformer=DepthformerConfig(**config.pop("depthformer")),
        preprocessor=PreprocessorConfig(**config.pop("preprocessor")),
        **config,
    )
    return LFM2AudioModel(conf)


def write_base_model(path: Path) -> None:
    """A model folder laid out like the Hugging Face repos: config.json + model.safetensors."""
    torch.manual_seed(0)
    path.mkdir()
    save_model(tiny_model(), str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(CONFIG))


def write_dataset(path: Path) -> None:
    """One preprocessed sample: a user turn with speech, an assistant turn with speech out."""
    text, audio, out, eos = LFMModality.TEXT, LFMModality.AUDIO_IN, LFMModality.AUDIO_OUT, 2048
    modality = [text] * 3 + [audio] * 2 + [text] + [out] * 3 + [text]
    supervised = [False] * 5 + [True] * 5
    Dataset.from_dict(
        {
            "text": [[[1, 2, 3, 128, 7]]],
            "audio_in": [torch.randn(128, 16).tolist()],
            "audio_in_lens": [[16]],
            "audio_out": [[[5, 6, eos]] * 8],
            "modality_flag": [[[int(m) for m in modality]]],
            "supervision_mask": [[supervised]],
        }
    ).save_to_disk(str(path))


def test_get_model_dir_accepts_a_local_directory_given_as_a_string(tmp_path: Path) -> None:
    assert get_model_dir(str(tmp_path)) == tmp_path


def test_trainer_final_checkpoint_loads_with_from_pretrained(tmp_path: Path) -> None:
    base, data, out = tmp_path / "base", tmp_path / "data", tmp_path / "out"
    write_base_model(base)
    write_dataset(data)

    trainer = Trainer(
        model_id=str(base),
        train_data=LFM2DataLoader(str(data), context_length=16),
        lr=1e-3,
        max_steps=2,
        warmup_steps=1,
        batch_size=1,
        save_interval=1000,
        output_dir=str(out),
    )
    trainer.train()
    trained = {name: value.cpu() for name, value in trainer.accelerator.unwrap_model(trainer.model).state_dict().items()}
    initial = LFM2AudioModel.from_pretrained(base, device="cpu").state_dict()
    assert any(not torch.equal(initial[name], trained[name]) for name in trained)

    for model_dir in (str(out / "final"), out / "final"):
        loaded = LFM2AudioModel.from_pretrained(model_dir, device="cpu").state_dict()
        assert loaded.keys() == trained.keys()
        assert all(torch.equal(loaded[name], trained[name]) for name in trained)
