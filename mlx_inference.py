#!/usr/bin/env python3
"""MLX inference wrapper for YuE2-3B.

Wraps the native MLX inference from https://huggingface.co/ahmadw/YuE2-3B-MLX
into a simple callable pipeline compatible with the Gradio UI.

Expected directory structure after install:
  models/YuE2-3B-MLX/
    yue2_model.py           (model architecture)
    yue2_vae.py             (VAE decoder)
    bf16/
      model.safetensors
      vae.safetensors
      config.json
      vae_config.json
      qwen.tiktoken
      yue2_generation_config.json
    8bit/
      ...
    4bit/
      ...

Usage as library:
    from mlx_inference import Yue2PipelineMLX, ModelVariant
    pipe = Yue2PipelineMLX("models/YuE2-3B-MLX", variant=ModelVariant.EIGHT_BIT)
    result = pipe(style="...", lyrics="...", cot="full", seed=831001)
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
import math
import os
import psutil
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mlx.core as mx
import numpy as np

# Ensure the model directory is on the path for yue2_model.py / yue2_vae.py
# These are downloaded into models/YuE2-3B-MLX/ by huggingface-cli
_MODEL_ROOT = Path(__file__).parent / "models" / "YuE2-3B-MLX"
if _MODEL_ROOT.exists():
    sys.path.insert(0, str(_MODEL_ROOT))

# ---------------------------------------------------------------------------
# Enum for model variants
# ---------------------------------------------------------------------------

class ModelVariant(Enum):
    BF16 = "bf16"
    EIGHT_BIT = "8bit"
    FOUR_BIT = "4bit"


# ---------------------------------------------------------------------------
# Tokenizer (from generate.py)
# ---------------------------------------------------------------------------

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
VOCAB, CONTEXT = 184704, 24576
SAMPLE_RATE = 48000

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


class Tokenizer:
    """The frozen Qwen text/ABC BPE (not the audio codec)."""

    def __init__(self, merge_file: Path):
        import tiktoken
        ranks = {base64.b64decode(t): int(r) for t, r in
                 (line.split() for line in Path(merge_file).read_bytes().splitlines() if line)}
        specials = ["</s>", "\n", "\n\n", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += [f"<extra_{i}>" for i in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self._enc = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks,
                                      special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})

    def encode(self, text):
        return self._enc.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, ids):
        return self._enc.decode([int(i) for i in ids if 0 <= i < self._enc.n_vocab], errors="replace")


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000


# ---------------------------------------------------------------------------
# Prefix helpers
# ---------------------------------------------------------------------------

def request_text(style, lyrics, cot):
    return f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"


def token_prefix(tok, style, lyrics, cot, abc_ids=None):
    base = [EOD] + tok.encode(request_text(style, lyrics, cot))
    if cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        return base + [ABC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def negative_prefix(tok, cot, abc_ids):
    base = [EOD] + tok.encode(INSTRUCTIONS[cot])
    if cot == "off":
        return base + [MUSIC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


# ---------------------------------------------------------------------------
# AR sampling
# ---------------------------------------------------------------------------

def allowed_mask(phase, dtype):
    idx = mx.arange(VOCAB)
    end = ABC_END if phase == "abc" else MUSIC_END
    ok = (idx < EOD) if phase == "abc" else ((idx >= CODEC_OFFSET) & (idx < CODEC_OFFSET + CODEC_SIZE))
    return mx.where(ok | (idx == end), 0.0, -mx.inf).astype(dtype)


def distribution(logits, s: Sampling, history, step, phase, allowed, legacy_off=False):
    scores = logits if legacy_off else logits.astype(mx.float32)
    end = ABC_END if phase == "abc" else MUSIC_END
    scores = scores + allowed
    if step < s.min_tokens:
        scores[end] = -mx.inf
    recent = history[-s.penalty_window:]
    if s.repetition_penalty != 1.0 and recent:
        freq = mx.zeros((VOCAB,), scores.dtype).at[mx.array(recent)].add(1)
        alpha = mx.power(mx.array(s.repetition_penalty, scores.dtype), freq)
        scores = mx.where(scores < 0, scores * alpha, scores / alpha)
    if s.temperature == 0:
        return scores
    if s.temperature != 1:
        scores = scores / s.temperature
    kth = VOCAB - min(s.top_k, VOCAB)
    threshold = mx.partition(scores, kth)[kth]
    scores = mx.where(scores < threshold, -mx.inf, scores)
    if s.top_p < 1:
        order = mx.argsort(-scores)
        values = scores[order]
        probs = mx.softmax(values.astype(mx.float32), axis=-1)
        removed = (mx.cumsum(probs) - probs) > s.top_p
        removed = removed & (mx.arange(VOCAB) >= (3 if legacy_off else 1))
        scores = mx.put_along_axis(scores, order, mx.where(removed, -mx.inf, values), axis=0)
    return scores


def generate_tokens(model, prefix, s: Sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, on_token=None):
    """Return (ids, truncated). Two KV caches when CFG is active (cond / uncond)."""
    if len(prefix) + s.max_tokens > CONTEXT or (negative and len(negative) + s.max_tokens > CONTEXT):
        raise ValueError("Prefix + generation budget exceeds the 24576 context")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")

    n_layers = len(model.model.layers)
    caches = [[None for _ in range(n_layers)] for _ in range(2 if cfg_scale != 1 else 1)]
    # Initialize KV caches
    from yue2_model import KVCache
    for i in range(len(caches)):
        caches[i] = [KVCache() for _ in range(n_layers)]

    cond = model.ar_step(mx.array([prefix]), caches[0])
    uncond = model.ar_step(mx.array([negative]), caches[1]) if cfg_scale != 1 else None

    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = allowed_mask(phase, cond.dtype if legacy_off else mx.float32)
    key = mx.random.key(seed)
    history, eos = [], False

    for step in range(s.max_tokens):
        if cfg_scale == 1:
            logits = cond
        else:
            logits = uncond + cfg_scale * (cond - uncond)
        scores = distribution(logits[0], s, history, step, phase, allowed, legacy_off)
        if s.temperature == 0:
            token = int(mx.argmax(scores).item())
        else:
            key, sub = mx.random.split(key)
            token = int(mx.random.categorical(scores.astype(mx.float32), key=sub).item())

        if on_token is not None:
            on_token(phase, token)

        if token == end:
            eos = True
            break

        history.append(token)
        if step + 1 < s.max_tokens:
            nxt = mx.array([[token]])
            cond = model.ar_step(nxt, caches[0])
            if uncond is not None:
                uncond = model.ar_step(nxt, caches[1])

    return history, not eos


# ---------------------------------------------------------------------------
# NAR flow matching
# ---------------------------------------------------------------------------

def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    size = min((context - prefix_tokens - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("Empty codec or prefix leaves no acoustic context")
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]


def _logit(t):
    return max(-20.0, min(20.0, math.log(t / (1 - t)))) if 0 < t < 1 else (20.0 if t >= 1 else -20.0)


def synthesize(model, prefix, codec, seed, steps=32, noise=None, on_progress=None, tile_size=4096, tile_overlap=128):
    """Return latents [frames,64] float32 via midpoint ODE from t=1 → 0.

    Args:
        tile_size: Process this many frames per tile. Larger = faster.
        tile_overlap: Extra frames processed as overlap at tile boundaries.
                      Smooths transitions, reduces boundary artifacts.
    """
    if noise is None:
        noise = mx.random.normal((len(codec), 64), key=mx.random.key(seed))
    dt = 1.0 / steps
    out = []

    for a, b, core_a, core_b in _tile_ranges(len(codec), tile_size, tile_overlap):
        ar_tokens = prefix + [c + CODEC_OFFSET for c in codec[a:b]] + [MUSIC_END]
        ar_cache = model.nar_prefill(ar_tokens)

        state = noise[a:b].astype(mx.bfloat16)
        for step in range(steps):
            t = 1.0 - step * dt
            v1 = model.nar_velocity(state, _logit(t), ar_cache, len(ar_tokens))
            mid = state - v1 * (dt / 2)
            state = state - model.nar_velocity(mid, _logit(t - dt / 2), ar_cache, len(ar_tokens)) * dt
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, steps)
        # Keep only the core region (discard overlap)
        out.append(state[core_a - a: core_b - a].astype(mx.float32))

    # Convert to numpy incrementally to avoid mx.concatenate() spike
    tiles_np = [np.array(t) for t in out]
    del out
    return np.concatenate(tiles_np, axis=0)


def _tile_ranges(n, tile_size, overlap=0):
    """Split [0, n) into tiles with optional overlap.

    Each tile processes [start - overlap, end + overlap] frames,
    but only the core [start, end) is kept.
    """
    tiles = []
    start = 0
    while start < n:
        end = min(start + tile_size, n)
        actual_start = max(0, start - overlap)
        actual_end = min(n, end + overlap)
        tiles.append((actual_start, actual_end, start, end))
        start = end
    return tiles


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

class Yue2PipelineMLX:
    """MLX-powered YuE2 pipeline with a simple callable interface."""

    HUB_REPO = "ahmadw/YuE2-3B-MLX"

    def __init__(self, model_root: str | Path, variant: ModelVariant = ModelVariant.EIGHT_BIT, log=print):
        self.model_root = Path(model_root)
        self.variant = variant
        self.variant_dir = self.model_root / variant.value
        self.log = log

        # Ensure model files exist — download on-demand if missing
        self._ensure_model_files()

        # Load generation config
        gen_path = self.variant_dir / "yue2_generation_config.json"
        if not gen_path.exists():
            gen_path = self.model_root / "yue2_generation_config.json"
        gen = json.loads(gen_path.read_text())
        self.abc_sampling = Sampling(**gen["abc"])
        self.semantic_sampling = Sampling(**gen["semantic"])
        self.ode_steps = gen.get("ode_steps", 32)

        # Load tokenizer
        tiktoken_path = self.variant_dir / "qwen.tiktoken"
        if not tiktoken_path.exists():
            tiktoken_path = self.model_root / "qwen.tiktoken"
        self.log("[tokenizer] loading")
        self.tokenizer = Tokenizer(tiktoken_path)

        # Load VAE
        self.log("[vae] loading")
        from yue2_vae import load_vae
        self.vae = load_vae(self.variant_dir)

        # Load model
        self.log("[model] loading")
        from yue2_model import load_model
        self.model = load_model(self.variant_dir)
        self.log(f"[model] loaded variant={variant.value}")

    def _ensure_model_files(self):
        """Download model files from HuggingFace if they don't exist locally."""
        variant = self.variant.value
        variant_dir = self.model_root / variant
        model_check = variant_dir / "model.safetensors"

        if model_check.exists():
            self.log("[model] files already exist, skipping download")
            return

        self.log(f"[model] downloading '{variant}' variant from HuggingFace (~{self._variant_size(variant)})")
        try:
            import subprocess
            import shutil
            import os

            # Create variant directory
            variant_dir.mkdir(parents=True, exist_ok=True)

            # List of files to download for this variant
            variant_files = [
                "model.safetensors",
                "vae.safetensors",
                "config.json",
                "vae_config.json",
                "qwen.tiktoken",
                "yue2_generation_config.json",
            ]

            # Download each file using curl (bypasses broken Python httpx)
            base_url = f"https://huggingface.co/{self.HUB_REPO}/resolve/main"
            for fname in variant_files:
                url = f"{base_url}/{variant}/{fname}"
                dest = variant_dir / fname
                if dest.exists() and dest.stat().st_size > 0:
                    self.log(f"[model] {fname} already exists, skipping")
                    continue
                self.log(f"[model] downloading {fname}...")
                result = subprocess.run(
                    ["curl", "-fsSL", url, "-o", str(dest)],
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour per file
                )
                if result.returncode != 0:
                    self.log(f"[model] warning: failed to download {fname}: {result.stderr[:200]}")
                    # Don't raise - continue downloading other files

            # Download root files (yue2_model.py, yue2_vae.py) if not already present
            root_files = ["yue2_model.py", "yue2_vae.py"]
            for fname in root_files:
                url = f"{base_url}/{fname}"
                dest = Path(__file__).parent / fname
                if not dest.exists():
                    self.log(f"[model] downloading {fname}...")
                    result = subprocess.run(
                        ["curl", "-fsSL", url, "-o", str(dest)],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode != 0:
                        self.log(f"[model] failed to download {fname}: {result.stderr}")
                        # Non-critical, continue

            # Refresh sys.path
            if _MODEL_ROOT.exists():
                sys.path.insert(0, str(_MODEL_ROOT))

            self.log(f"[model] download complete ({variant} variant)")
        except Exception as exc:
            self.log(f"[model] download failed: {exc}")
            raise

    def _variant_size(self, variant):
        """Return estimated download size for a variant."""
        sizes = {"bf16": "~7 GB", "8bit": "~4.2 GB", "4bit": "~3.4 GB"}
        return sizes.get(variant, "~unknown")

    def release_models(self):
        """Drop resident AR/NAR/VAE weights (released for cover song transcription)."""
        import mlx.core as mx
        self.log("[pipeline] releasing models")
        del self.model
        del self.vae
        self.model = None
        self.vae = None
        mx.clear_cache()
        import gc
        gc.collect()

    def __call__(self, style: str, lyrics: str, cot: str = "full", seed: int = 831001,
                 abc: str | None = None, cfg_scale: float | None = None,
                 steps: int | None = None, tile_size: int = 4096, vae_tile: int = 256,
                 on_token=None, on_nar=None, on_vae=None, on_progress=None):
        """Generate audio from style + lyrics. Returns dict with 'audio' (numpy) and 'abc' (str)."""
        if cot not in INSTRUCTIONS:
            raise ValueError("cot must be off, melody or full")
        if abc is not None and cot == "off":
            raise ValueError("External ABC requires cot=melody/full")

        # Set MLX memory limits at the START so they apply to all phases
        # (model loading, NAR synthesis, VAE decode)
        # Cache limit: 128MB — aggressively evicts unused tensors
        # Memory limit: (total_RAM - 8GB) — reserves 8GB for OS
        _total_ram_gib = psutil.virtual_memory().total / (1024**3)
        mx.set_cache_limit(128 * 1024 * 1024)  # 128 MB cache
        mx.set_memory_limit(int((_total_ram_gib - 8) * 1024 * 1024 * 1024))

        tok = self.tokenizer
        log = self.log

        # 1. Symbolic plan (ABC score)
        abc_ids, abc_text = [], None
        if cot != "off":
            if abc is not None:
                abc_ids = tok.encode(abc)
                abc_text = abc
            else:
                log("[plan] generating ABC score")
                abc_ids, truncated = generate_tokens(
                    self.model,
                    token_prefix(tok, style, lyrics, cot),
                    self.abc_sampling, seed, "abc",
                    on_token=on_token,
                )
                abc_text = tok.decode(abc_ids)
                if truncated:
                    log("[plan] ABC hit max_tokens")

        prefix = token_prefix(tok, style, lyrics, cot, abc_ids)

        # 2. Semantic codec tokens
        guidance = (1.01 if cot == "off" else 1.0) if cfg_scale is None else cfg_scale
        negative = negative_prefix(tok, cot, abc_ids) if guidance != 1 else None
        log(f"[semantic] prefix {len(prefix)} tokens, cfg {guidance}")
        ids, truncated = generate_tokens(
            self.model, prefix, self.semantic_sampling, seed,
            "semantic", negative, guidance, legacy_off=cot == "off",
            on_token=on_token,
        )
        if truncated:
            log("[semantic] hit max_tokens")
        codec = [t - CODEC_OFFSET for t in ids]
        if not codec:
            raise RuntimeError("Semantic stage produced no codec tokens")

        # 3. Acoustic latents (NAR flow matching)
        n_steps = steps or self.ode_steps
        log(f"[nar] {len(codec)} frames ({len(codec) * 1920 / SAMPLE_RATE:.1f}s), {n_steps} midpoint steps")

        def nar_cb(done, total):
            if on_nar is not None:
                on_nar(done, total)
            if on_progress is not None:
                on_progress("nar", done, total)

        latents = synthesize(self.model, prefix, codec, seed, n_steps, tile_size=tile_size, tile_overlap=128, on_progress=nar_cb)

        # 5. VAE decode to waveform
        # Convert to bfloat16 to halve latent tensor memory (negligible quality loss)
        latents_bf16 = latents.astype(mx.bfloat16) if latents.dtype == mx.float32 else latents
        del latents
        gc.collect()
        mx.clear_cache()

        log("[vae] decoding")
        audio_np = np.array(self.vae.decode_tiled(latents_bf16, core=vae_tile))
        audio = np.clip(audio_np, -1, 1)
        return {
            "audio": audio,
            "abc": abc_text,
            "codec": codec,
            "latents": latents_bf16,
            "prefix": prefix,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YuE2 MLX inference CLI")
    parser.add_argument("--model", type=Path, required=True, help="MLX model directory (e.g., models/YuE2-3B-MLX)")
    parser.add_argument("--variant", default="8bit", choices=["bf16", "8bit", "4bit"], help="Model variant")
    parser.add_argument("--style", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lyrics")
    group.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--cot", default="full", choices=list(INSTRUCTIONS))
    parser.add_argument("--abc-file", type=Path)
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--steps", type=int, help="NAR midpoint steps (default from config)")
    parser.add_argument("--out", type=Path, default=Path("yue2.wav"))
    args = parser.parse_args()

    lyrics = args.lyrics if args.lyrics else Path(args.lyrics_file).read_text()
    abc = Path(args.abc_file).read_text() if args.abc_file else None
    log = lambda msg: print(msg, file=sys.stderr, flush=True)

    variant = ModelVariant(args.variant.upper().replace("BF16", "BF16").replace("EIGHTBIT", "EIGHT_BIT").replace("FOURBIT", "FOUR_BIT"))
    variant_map = {"bf16": ModelVariant.BF16, "8bit": ModelVariant.EIGHT_BIT, "4bit": ModelVariant.FOUR_BIT}
    variant = variant_map.get(args.variant.lower(), ModelVariant.EIGHT_BIT)

    pipe = Yue2PipelineMLX(args.model, variant=variant, log=log)
    start = time.perf_counter()
    result = pipe(
        style=args.style, lyrics=lyrics, cot=args.cot, seed=args.seed,
        abc=abc, cfg_scale=args.cfg_scale, steps=args.steps,
    )

    # Write WAV
    audio = result["audio"]
    if len(audio.shape) == 1:
        audio = audio.reshape(-1, 1)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(args.out), "wb") as f:
        f.setnchannels(1 if len(pcm.shape) == 1 else pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())

    if result["abc"] is not None:
        args.out.with_suffix(".abc").write_text(result["abc"])

    duration = len(audio) / SAMPLE_RATE
    log(f"[done] {args.out} {duration:.1f}s in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
