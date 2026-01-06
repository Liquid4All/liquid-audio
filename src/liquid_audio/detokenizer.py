import torch
from torch import nn
from transformers import Lfm2Config, Lfm2Model


class FusedEmbedding(nn.Module):
    """Turn codes into embeddings"""

    def __init__(
        self,
        dim: int,
        codeboooks: int = 8,
        vocab_size: int = 2048,
    ):
        super().__init__()
        self.emb = nn.Embedding(codeboooks * vocab_size, dim)

        self.codeboooks = codeboooks
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.codeboooks, device=x.device) * self.vocab_size  # TODO: buffer?
        offset_x = offsets[:, None] + x
        return self.emb(offset_x).mean(1)  # B L D


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Adapted from Vocos: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L7
    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.
        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.
        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,  # type: ignore[arg-type]
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        _B, _N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]  # type: ignore[index]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)  # type: ignore[operator]
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class LFM2AudioDetokenizer(nn.Module):
    def __init__(self, backbone_config: Lfm2Config):
        super().__init__()
        self.emb = FusedEmbedding(512)
        self.lfm = Lfm2Model(backbone_config)
        self.lin = nn.Linear(512, 1282)  # half are log-magnitude, half are angle

        self.istft = ISTFT(1280, 320, 1280, padding="same")
        self.sliding_window_size = getattr(backbone_config, "sliding_window", 30)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        upsample_size = 6 * x.shape[1]
        x = nn.functional.interpolate(x.mT, upsample_size, mode="nearest-exact").mT

        # Set attn mask
        idx = torch.arange(x.shape[1], device=x.device)
        d_idx = idx - idx[:, None]
        mask = torch.logical_and(d_idx <= 0, d_idx > -self.sliding_window_size)[None, None, ...]

        x = self.lfm(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        x = self.lin(x)

        log_abs, angle = torch.chunk(x.mT.contiguous(), 2, 1)
        y = torch.polar(log_abs.exp(), angle)

        return self.istft(y)
