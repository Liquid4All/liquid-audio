"""Initialize models"""

import logging

import torch

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

logger = logging.getLogger(__name__)

__all__ = ["lfm2_audio", "mimi", "proc"]

HF_DIR = "LiquidAI/LFM2.5-Audio-1.5B"

logging.info("Loading processor")
proc = LFM2AudioProcessor.from_pretrained(HF_DIR).eval()
logging.info("Loading model")
lfm2_audio = LFM2AudioModel.from_pretrained(HF_DIR).eval()
logging.info("Loading tokenizer")
mimi = proc.mimi.eval()

logging.info("Warmup tokenizer")
with mimi.streaming(1), torch.no_grad():
    for _ in range(5):
        x = torch.randint(2048, (1, 8, 1), device="cuda")
        mimi.decode(x)
