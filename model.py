"""
SPECT neural-network reconstruction model.

Provides the NeuralNetwork architecture, model loading, and single-sinogram
inference.  Import this module from both infer_sinogram_directory.py and
visualizer.py.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.backends.cudnn.benchmark = True

# Default sinogram input dimensions: (depth, angles, width) == (128, 120, 128)
DEFAULT_INPUT_DIMS: tuple[int, int, int] = (128, 120, 128)


# --- model architecture ------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, filters: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.query_conv = nn.Conv3d(filters, filters, kernel_size=1, padding=0)
        self.key_conv = nn.Conv3d(filters, filters, kernel_size=1, padding=0)
        self.value_conv = nn.Conv3d(filters, filters, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h, d = x.size()
        q = self.query_conv(x).view(b, -1, c)
        k = self.key_conv(x).view(b, -1, c)
        v = self.value_conv(x).view(b, -1, c)
        s = torch.bmm(q.permute(0, 2, 1), k)
        scores = torch.bmm(v, torch.nn.functional.softmax(s, dim=-1))
        return x + self.gamma * scores.view(b, c, w, h, d)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 2, 3, padding=1, stride=2),
            nn.BatchNorm3d(2, momentum=0.8),
            nn.SiLU(),
            nn.Conv3d(2, 4, 3, padding=1, stride=2),
            nn.BatchNorm3d(4, momentum=0.8),
            nn.SiLU(),
            nn.Conv3d(4, 8, 3, padding=1, stride=2),
            nn.BatchNorm3d(8, momentum=0.8),
            nn.SiLU(),
            nn.Conv3d(8, 16, 3, padding=1, stride=2),
            nn.BatchNorm3d(16, momentum=0.8),
            nn.SiLU(),
            Attention(16),
            nn.Conv3d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm3d(32, momentum=0.8),
            nn.SiLU(),
            Attention(32),
            nn.Conv3d(32, 64, 3, padding=1, stride=(2, 1, 2)),
            nn.BatchNorm3d(64, momentum=0.8),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(-1, 1, 128, 120, 128))


_DECODER_BASE = 32


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose3d(1, _DECODER_BASE, 2, 2, padding=0, output_padding=0),
            nn.Conv3d(_DECODER_BASE, _DECODER_BASE, 3, stride=1, padding=1),
            nn.BatchNorm3d(_DECODER_BASE, momentum=0.8),
            nn.SiLU(),
            Attention(_DECODER_BASE),
            nn.ConvTranspose3d(_DECODER_BASE, _DECODER_BASE // 2, 2, 2, padding=0, output_padding=0),
            nn.Conv3d(_DECODER_BASE // 2, _DECODER_BASE // 2, 3, stride=1, padding=1),
            nn.BatchNorm3d(_DECODER_BASE // 2, momentum=0.8),
            nn.SiLU(),
            Attention(_DECODER_BASE // 2),
            nn.ConvTranspose3d(_DECODER_BASE // 2, 1, 2, 2, padding=0, output_padding=0),
            nn.Conv3d(1, 1, 3, stride=1, padding=1),
            nn.BatchNorm3d(1, momentum=0.8),
            nn.SiLU(),
            Attention(1),
            nn.ConvTranspose3d(1, 1, 2, (2, 2, 2), padding=0, output_padding=0),
            nn.Conv3d(1, 1, 3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(1, momentum=0.8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(-1, 1, 8, 8, 8))


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(Encoder(), Decoder())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# --- model loading -----------------------------------------------------------

def clean_state_dict_keys(state_dict: dict) -> dict:
    """Strip ``_orig_mod.`` and ``module.`` prefixes from compiled/DDP checkpoints."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_model(model_path: str, device: str, use_compile: bool = False) -> NeuralNetwork:
    """Load a NeuralNetwork from a .pth checkpoint and set it to eval mode."""
    mdl = NeuralNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Unsupported checkpoint format: expected a state_dict or a dict with model_state_dict"
        )

    try:
        mdl.load_state_dict(state_dict)
    except RuntimeError:
        mdl.load_state_dict(clean_state_dict_keys(state_dict))

    if use_compile:
        mdl = torch.compile(mdl)

    mdl.eval()
    return mdl


# --- single-file inference ---------------------------------------------------

def reconstruct_sinogram(
    mdl: NeuralNetwork,
    path: str,
    device: str,
    input_dims: tuple[int, int, int] = DEFAULT_INPUT_DIMS,
    input_dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """Reconstruct one sinogram .raw file into a (64, 128, 128) float32 volume.

    Parameters
    ----------
    mdl:
        A loaded, eval-mode NeuralNetwork.
    path:
        Path to the sinogram .raw file (flat float32 array of *input_dims* values).
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    input_dims:
        Expected sinogram dimensions ``(D, H, W)``; defaults to ``(128, 120, 128)``.
    input_dtype:
        NumPy dtype of the raw file; defaults to ``float32``.
    """
    dtype = np.dtype(input_dtype)
    arr = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(input_dims))
    if arr.size != expected:
        raise ValueError(
            f"'{Path(path).name}' has {arr.size} values; "
            f"expected {expected} for dims {input_dims}"
        )
    # reshape to (D, H, W) — same convention as infer_sinogram_directory
    sino = arr.reshape(input_dims[::-1])

    x = torch.from_numpy(sino[np.newaxis]).to(device, non_blocking=True)
    with torch.inference_mode():
        pred = mdl(x)

    # squeeze batch + channel dims; convert to float32 numpy
    return pred.squeeze(1).detach().to(torch.float32).cpu().numpy()[0]
