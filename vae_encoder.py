"""MLX YuE2-VAE encoder (upstream only ships decoder).

The hum-to-song prosody adapter conditions the acoustic decoder on VAE latents of a synthetic carrier,
so we need encode. Architecture mirrors yue2/modeling_vae.py::OobleckEncoder:
Conv1d(2->64, k7) then six EncoderBlock(s) (three ResidualUnit + SnakeBeta + strided conv)
then SnakeBeta + Conv1d(2048->128, k3). The 128 output channels are (mean, scale);
encode returns the posterior mean (first 64).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from lyra.vae import ResidualUnit, activation
from safetensors import safe_open

SAMPLE_RATE = 48000


class EncoderBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_snake):
        super().__init__()
        # Match the YuE2 studio encoder structure:
        # layers.0-2 = ResidualUnit (three residual blocks)
        # layers.3 = SnakeBeta (activation after ResidualUnits)
        # layers.4 = Conv1d (strided conv to reduce resolution)
        self.layers = [
            *[ResidualUnit(cin, d, use_snake) for d in (1, 3, 9)],  # layers.0-2
            activation(cin, use_snake),  # layers.3
            nn.Conv1d(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),  # layers.4
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckEncoder(nn.Module):
    """Channels-last MLX encoder: [B, S, 2] float32 -> [B, T, latent_dim] (mean + scale)."""

    def __init__(self, config):
        super().__init__()
        if config.get("antialias_activation", False):
            raise ValueError("Unsupported encoder architecture")
        c, mults = config["channels"], [1, *config["c_mults"]]
        strides, snake = config["strides"], config.get("use_snake", False)
        self.strides = tuple(strides)
        self.ratio = math.prod(strides)
        self.latent_dim = config["latent_dim"]
        self.mean_dim = self.latent_dim // 2
        # layers.0 = initial Conv1d
        # layers.1-6 = EncoderBlock (strided conv + 3 ResidualUnit)
        # layers.7 = final SnakeBeta
        # layers.8 = final Conv1d (to latent_dim)
        self.layers = [
            nn.Conv1d(config["in_channels"], mults[0] * c, 7, padding=3),  # layers.0
            *[EncoderBlock(mults[i] * c, mults[i + 1] * c, strides[i], snake) for i in range(len(mults) - 1)],  # layers.1-6
            activation(mults[-1] * c, snake),  # layers.7
            nn.Conv1d(mults[-1] * c, self.latent_dim, 3, padding=1),  # layers.8
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def encoder_weights(weights) -> dict[str, mx.array]:
    """Fold official weight_norm (dim=0) and move kernels to MLX's [out, k, in] layout."""
    result = {}
    for full_key, value in weights.items():
        # Strip "encoder." prefix if present (it's just a namespace)
        key = full_key
        if key.startswith("encoder."):
            key = key.removeprefix("encoder.")
        
        value = np.asarray(value, dtype=np.float32)
        if key.endswith("weight_v"):
            gain = np.asarray(weights.get(full_key[:-1] + "g", weights.get(key[:-1] + "g", 1.0)), dtype=np.float32)
            norm = np.sqrt(np.sum(value * value, axis=(1, 2), keepdims=True))
            if np.any(norm == 0):
                raise ValueError("Zero weight-normalization denominator")
            value = (value * (gain / norm)).transpose(0, 2, 1)
            key = key.removesuffix("_v")
        result[key] = mx.array(value)
    return result


def load_encoder(directory) -> OobleckEncoder:
    directory = Path(directory)
    
    # Look for vae_config.json in variant subdirectories (8bit/, 4bit/, bf16/)
    # The encoder architecture is the same across all variants
    config = None
    config_path = None
    for subdir in ["8bit", "4bit", "bf16"]:
        candidate = directory / subdir / "vae_config.json"
        if candidate.exists():
            config = json.loads(candidate.read_text())
            config_path = candidate
            break
    
    if config is None:
        # Fallback: check for config.json in variant subdirs
        for subdir in ["8bit", "4bit", "bf16"]:
            candidate = directory / subdir / "config.json"
            if candidate.exists():
                c = json.loads(candidate.read_text())
                if "encoder_config" in c:
                    config = c
                    config_path = candidate
                    break
    
    if config is None:
        raise FileNotFoundError(f"No vae_config.json or config.json with encoder_config found in {directory}/{{8bit,4bit,bf16}}/")

    model = OobleckEncoder(config["encoder_config"])

    # Load encoder weights from parent directory
    encoder_path = directory / "encoder.safetensors"
    if not encoder_path.exists():
        raise FileNotFoundError(f"No encoder.safetensors found in {directory}")
    weights_path = encoder_path

    with safe_open(weights_path, framework="numpy") as reader:
        raw_keys = reader.keys()
        weights = {k: reader.get_tensor(k) for k in raw_keys if k.startswith("encoder.")}

    mx_weights = encoder_weights(weights)
    
    # Load all weights at once with strict=False
    # The keys are flat like "layers.0.weight", "layers.1.layers.0.layers.1.weight", etc.
    # which matches the nested model structure
    model.load_weights(list(mx_weights.items()), strict=False)
    model.freeze()
    mx.eval(model.parameters())
    return model


def encode(model: OobleckEncoder, audio: np.ndarray, *, chunk_seconds: int = 30, cancelled=None,
           on_progress=None) -> np.ndarray:
    """FP32 stereo [S, 2] at 48 kHz -> posterior-mean latents [T, 64] (25 Hz), CPU float32.

    Encoded in chunk_seconds pieces to bound activations; a trailing piece shorter than
    one latent frame (1920 samples) is dropped.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < model.ratio:
        raise ValueError("Expected stereo audio [S, 2] with at least one latent frame")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite values")
    chunk = int(chunk_seconds * SAMPLE_RATE)
    starts = [s for s in range(0, len(audio), chunk) if len(audio) - s >= model.ratio]
    pieces = []
    for index, start in enumerate(starts, 1):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during VAE encoding")
        latent = model(mx.array(audio[None, start:start + chunk]))[0, :, : model.mean_dim]
        mx.eval(latent)
        pieces.append(np.array(latent, dtype=np.float32))
        mx.clear_cache()
        if on_progress is not None:
            on_progress(index, len(starts))
    latents = np.concatenate(pieces)
    if latents.ndim != 2 or latents.shape[1] != model.mean_dim or not np.isfinite(latents).all():
        raise ValueError("VAE encoder produced non-finite or misshaped latents")
    return latents
