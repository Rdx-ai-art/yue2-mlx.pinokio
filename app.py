#!/usr/bin/env python3
"""YUE2 // MLX — Gradio web UI for YuE2 on Apple Silicon (MLX).

A streamlined, Mac-optimized Gradio interface that wraps the native MLX
inference from https://huggingface.co/Dirdx/YuE2-3B-MLX-with-Hum-encoder.

Features:
  - Original song generation (style + lyrics → 48 kHz stereo)
  - LLM Writing Room for lyric/style composition (OpenAI-compatible API)
  - Cover mode with ABC score input
  - All sampling parameters exposed
  - Progress tracking through generation stages
  - Audio playback and download

Requires: Apple Silicon Mac (M1/M2/M3/M4), 16GB+ RAM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import wave
from pathlib import Path

# Set HuggingFace cache to our models folder so all downloads go to one place
HF_CACHE = Path(__file__).parent / "models" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)

import gradio as gr
import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# MLX inference backend
# ---------------------------------------------------------------------------

# We inline the MLX pipeline so there's no dependency on the upstream yue2
# package.  The generate.py / yue2_model.py / yue2_vae.py files from the
# MLX repo are copied into this folder during install.
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mlx_inference import Yue2PipelineMLX, ModelVariant
    from lora import discover_loras, list_hum_adapters
except ImportError as exc:
    sys.exit(
        "MLX inference files not found. Run Install first, or check that "
        "mlx_inference.py, yue2_model.py, yue2_vae.py are in the app folder.\n"
        f"ImportError: {exc}"
    )

# Variant display string → ModelVariant enum
_VARIANT_MAP = {
    "BF16 (highest quality, 7 GB)": ModelVariant.BF16,
    "8-bit (recommended, 4.2 GB)": ModelVariant.EIGHT_BIT,
    "4-bit (fastest, 3.4 GB)": ModelVariant.FOUR_BIT,
}


# ---------------------------------------------------------------------------
# Global pipeline singleton
# ---------------------------------------------------------------------------

_PIPE: Yue2PipelineMLX | None = None
_PIPE_KEY: tuple | None = None
_LOCK = __import__("threading").Lock()


def _load_pipeline(model_dir: str, variant: str, lora_adapters: list = None, lora_scale: float = 1.0, progress=None) -> Yue2PipelineMLX:
    """Load (or reuse) the MLX pipeline. Reuses existing one when settings unchanged."""
    global _PIPE, _PIPE_KEY

    mv = _VARIANT_MAP.get(variant, ModelVariant.EIGHT_BIT)
    key = (model_dir, variant, tuple(lora_adapters or []), lora_scale)

    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    if _PIPE is not None:
        del _PIPE
        _PIPE = None
        _PIPE_KEY = None

    if progress is not None:
        progress(0.0, desc="Loading MLX model…")
    
    # Build LoRA adapter specs
    lora_specs = []
    if lora_adapters:
        lora_dir = Path(model_dir).parent / "loras"
        if lora_dir.exists():
            from lora import discover_loras
            all_loras = discover_loras(lora_dir)
            # Filter to only selected adapters
            lora_specs = [l for l in all_loras if l["name"] in lora_adapters]
    
    if lora_specs:
        _PIPE = Yue2PipelineMLX(model_dir, variant=mv, lora_adapters=lora_specs, lora_scale=lora_scale)
    else:
        _PIPE = Yue2PipelineMLX(model_dir, variant=mv)
    _PIPE_KEY = key
    return _PIPE


def unload_pipeline():
    global _PIPE, _PIPE_DIR
    with _LOCK:
        _PIPE = None
        _PIPE_DIR = None


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _generate_song(
    style, lyrics, cot, seed, cfg_scale, steps, variant,
    tile_size, vae_tile, save_format,
    lora_adapters, lora_scale,
    abc_text=None,
    progress=gr.Progress(),
):
    """Generate a song using the MLX pipeline."""
    model_dir = "./models/YuE2-3B-MLX"
    mp3_bitrate = "320k"
    t0 = time.perf_counter()

    pipe = _load_pipeline(model_dir, variant, lora_adapters, lora_scale, progress)

    if not style.strip():
        raise gr.Error("Style is required (e.g. 'indie pop, warm vocal')")
    if not lyrics.strip():
        raise gr.Error("Lyrics are required")

    _CLEARED.clear()
    # Generate random seed if -1
    if seed == -1:
        import secrets
        seed = secrets.randbelow(2**32)

    def _on_token(phase, token):
        if phase == "abc":
            frac = 0.15 * min(1.0, token / 2000)
        else:
            frac = 0.15 + 0.40 * min(1.0, token / 5000)
        if progress is not None:
            progress(min(0.55, frac), desc=f"Generating {phase}: {token} tokens")

    def _on_nar(done, total):
        frac = 0.55 + 0.40 * min(1.0, done / max(1, total))
        if progress is not None:
            progress(frac, desc=f"Synthesizing audio: step {done}/{total}")

    callbacks = {
        "on_token": _on_token,
        "on_nar": _on_nar,
    }

    # Check for cancel request
    if _CLEARED.is_set():
        raise InterruptedError("Generation cancelled")

    # Clean up ABC text (dedent common indentation)
    cleaned_abc = None
    if abc_text and abc_text.strip():
        if cot == "off":
            raise gr.Error("An ABC score requires cot=full or cot=melody")
        cleaned_abc = textwrap.dedent(abc_text).strip()

    try:
        result = pipe(
            style=style.strip(),
            lyrics=lyrics.strip(),
            cot=cot,
            seed=seed,
            cfg_scale=float(cfg_scale) if cfg_scale else None,
            steps=int(steps),
            tile_size=int(tile_size),
            vae_tile=int(vae_tile),
            abc=cleaned_abc if cleaned_abc else None,
            **callbacks,
        )
    except InterruptedError:
        raise
    except Exception as exc:
        raise gr.Error(f"Generation failed: {type(exc).__name__}: {exc}") from exc

    elapsed = time.perf_counter() - t0
    audio = result["audio"]  # MLX array → numpy
    abc = result.get("abc", "")
    duration_s = len(audio) / 48000.0

    # Calculate ABC (COT) time
    abc_time = result.get("abc_time", None)

    # Normalize audio to prevent clipping/distortion
    if isinstance(audio, np.ndarray) and audio.dtype != object:
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Normalize to [-1, 1] range with headroom
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # First clip to prevent extreme values
            audio = np.clip(audio, -1.5, 1.5)
            # Scale to near full range with 10% headroom
            scale = 0.9 / max(max_val, 0.001)
            audio = audio * scale

    # Keep normalized float32 for saving, convert to int16 only for Gradio display
    audio_for_save = audio.copy() if isinstance(audio, np.ndarray) else audio
    if isinstance(audio, np.ndarray) and audio.dtype != object:
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        audio = (48000, audio_int16)

    info = (
        f"Done: {duration_s:.1f}s audio in {elapsed:.0f}s\n"
        f"seed={seed}  cot={cot}  steps={steps}\n"
        f"elapsed={elapsed:.0f}s"
    )

    # Save to outputs folder — each song gets its own directory
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    song_name = f"song_{timestamp}_s{seed}"
    song_dir = outputs_dir / song_name
    song_dir.mkdir(exist_ok=True)

    save_format = save_format or "WAV"
    is_mp3 = save_format == "MP3"
    mp3_bitrate = mp3_bitrate or "192k"
    saved_filepath = None

    if is_mp3:
        # Save as MP3 using ffmpeg (pre-installed with Pinokio AI bundle)
        wav_path = song_dir / f"{song_name}.wav"
        mp3_path = song_dir / f"{song_name}.mp3"
        try:
            # Write temp WAV from normalized float32 audio
            if isinstance(audio_for_save, tuple):
                _, audio_arr = audio_for_save
            else:
                audio_arr = audio_for_save
            if len(audio_arr.shape) == 1:
                audio_arr = audio_arr.reshape(-1, 1)
            pcm = (np.clip(audio_arr, -1, 1) * 32767).astype("<i2")
            import wave
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1 if len(pcm.shape) == 1 else pcm.shape[1])
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm.tobytes())
            # Convert to MP3 with ffmpeg
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", mp3_bitrate, "-ar", "48000", str(mp3_path)],
                capture_output=True, timeout=120, check=True
            )
            # Clean up temp WAV
            wav_path.unlink(missing_ok=True)
            filename = f"{song_name}.mp3"
            saved_filepath = str(mp3_path)
            info += f"\nSaved: {filename}"
        except Exception as e:
            info += f"\nSave error: {e}"
    else:
        # Save as WAV (default)
        filename = f"{song_name}.wav"
        filepath = song_dir / filename
        try:
            # Write from normalized float32 audio
            if isinstance(audio_for_save, tuple):
                _, audio_arr = audio_for_save
            else:
                audio_arr = audio_for_save
            if len(audio_arr.shape) == 1:
                audio_arr = audio_arr.reshape(-1, 1)
            pcm = (np.clip(audio_arr, -1, 1) * 32767).astype("<i2")
            import wave
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1 if len(pcm.shape) == 1 else pcm.shape[1])
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm.tobytes())
            saved_filepath = str(filepath)
            info += f"\nSaved: {filename}"
        except Exception as e:
            info += f"\nSave error: {e}"

    # Save metadata JSON
    metadata = {
        "song_name": song_name,
        "timestamp": timestamp,
        "duration_s": round(duration_s, 2),
        "gen_time_s": round(elapsed, 1),
        "abc_time_s": round(abc_time, 1) if abc_time else None,
        "seed": seed,
        "cot": cot,
        "steps": int(steps),
        "cfg_scale": float(cfg_scale) if cfg_scale else None,
        "nar_tile": int(tile_size),
        "vae_tile": int(vae_tile),
        "style": style.strip(),
        "lyrics": lyrics.strip(),
        "save_format": save_format,
        "mp3_bitrate": mp3_bitrate if is_mp3 else None,
        "filename": filename,
        "filepath": saved_filepath,
        "lora_adapters": lora_adapters or [],
        "lora_scale": lora_scale,
    }
    if abc and abc.strip():
        metadata["abc"] = abc
    try:
        metadata_path = song_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        info += f"\nMetadata: metadata.json"
    except Exception as e:
        info += f"\nMetadata save error: {e}"

    # Return audio tuple for playback, filepath for download
    # If MP3 saved, return file path so download button saves MP3
    if saved_filepath and is_mp3:
        return saved_filepath, abc, info
    return audio, abc, info


# ---------------------------------------------------------------------------
# LLM Writing Room
# ---------------------------------------------------------------------------

def _llm_chat(messages, api_url, model, max_tokens, temperature, api_key):
    """Call an OpenAI-compatible LLM API (e.g. LM Studio)."""
    import urllib.request
    import ssl

    if not api_url.strip():
        raise gr.Error("API URL is required (e.g. http://127.0.0.1:1234/v1/chat/completions)")

    system_prompt = (
        "You are a creative songwriting assistant. Help the user write lyrics, "
        "compose songs from ideas, suggest styles, and format lyrics with section "
        "tags like [Verse], [Chorus], [Bridge], [Pre-Chorus]. Keep lyrics original "
        "and creative. Always respond in the same language as the user.\n\n"
        "When generating lyrics, wrap them between #STARTLYRICS and #ENDLYRICS markers. "
        "Style/production notes go AFTER #ENDLYRICS."
    )

    messages_with_system = [{"role": "system", "content": system_prompt}] + messages

    data = json.dumps({
        "model": model or "default",
        "messages": messages_with_system,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=3600) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    content = result["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": content})
    return messages, content


def _generate_lyrics_from_idea(idea, api_url, model, max_tokens, temperature, api_key):
    """Generate full lyrics from a short idea/description."""
    prompt = (
        f"Create a complete song based on this idea: {idea}\n\n"
        f"Write full lyrics with section tags ([Verse], [Chorus], [Bridge], etc.).\n"
        f"Also suggest a musical style description (genre, mood, instrumentation).\n"
        f"Keep it original and creative."
    )
    messages = [{"role": "user", "content": prompt}]
    return _llm_chat(messages, api_url, model, max_tokens, temperature, api_key)


def _generate_style_from_idea(idea, api_url, model, max_tokens, temperature, api_key):
    """Generate a style prompt from a short idea."""
    prompt = (
        f"Describe the musical style for a song with this concept: {idea}\n\n"
        f"Include genre, mood, instrumentation, vocal style, and any other relevant "
        f"musical elements. Format as a comma-separated list suitable for a music AI model."
    )
    messages = [{"role": "user", "content": prompt}]
    return _llm_chat(messages, api_url, model, max_tokens, temperature, api_key)


def _expand_lyrics(current_lyrics, direction, api_url, model, max_tokens, temperature, api_key):
    """Add more verses or a bridge to existing lyrics."""
    prompt = (
        f"Here are existing lyrics:\n\n{current_lyrics}\n\n"
        f"Please {direction} by adding more content. Keep the style consistent.\n"
        f"Return the COMPLETE updated lyrics with all sections."
    )
    messages = [{"role": "user", "content": prompt}]
    return _llm_chat(messages, api_url, model, max_tokens, temperature, api_key)


def _parse_lyrics_from_response(content):
    """Extract lyrics from LLM response using #STARTLYRICS/#ENDLYRICS markers.
    Returns (lyrics_text, style_text)."""
    if not content or not content.strip():
        return "", ""

    import re
    # Extract lyrics between markers
    m = re.search(r'#STARTLYRICS\s*\n(.*?)\s*#ENDLYRICS', content, re.DOTALL)
    if m:
        lyrics_text = m.group(1).strip()
        # Style/production notes come after #ENDLYRICS
        remaining = content[m.end():].strip()
        style_text = remaining if remaining else ""
        return lyrics_text, style_text

    # Fallback: if content has section tags but no markers, treat as lyrics
    if '[' in content and ('Verse' in content or 'Chorus' in content):
        return content.strip(), ""

    return content.strip(), ""


_CLEARED = __import__("threading").Event()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _update_task_desc(task):
    """Update the task description markdown based on selected task."""
    descriptions = {
        "full": "**Full** — Transcribe the whole arrangement and follow it closely (closest to the original)",
        "melody-full": "**Melody-Full** — Transcribe the melody, let YuE2 write the full arrangement. Recommended for covers.",
        "melody-vocal": "**Melody-Vocal** — Transcribe the melody and follow it with the vocal line only.",
    }
    return descriptions.get(task, "")


def _update_melody_desc(melody):
    """Update the melody mode description markdown based on selected mode."""
    descriptions = {
        "continue": "**Continue** — The hum's notes open the score; YuE2 writes the rest of the song around them.",
        "hum_only": "**Hum Only** — Hum is the complete melody, like a cover; the song is as long as the hum.",
        "ignore": "**Ignore** — YuE2 writes the melody, hum shapes phrasing via lora adapter.",
    }
    return descriptions.get(melody, "")


# ── Lazy encoder download (only once, on first use) ──────────────────
_ENCODER_DOWNLOADED = False


def _ensure_encoder_downloaded():
    """Download encoder.safetensors lazily on first generation, only once."""
    global _ENCODER_DOWNLOADED
    if _ENCODER_DOWNLOADED:
        return
    _ENCODER_DOWNLOADED = True
    encoder_path = Path(__file__).parent / "models" / "YuE2-3B-MLX" / "encoder.safetensors"
    if encoder_path.exists():
        return
    import subprocess
    log("[hum] downloading encoder.safetensors (first use)...")
    try:
        result = subprocess.run(
            ["hf", "download", "Dirdx/YuE2-3B-MLX-with-Hum-encoder", "--include", "encoder.safetensors", "--local-dir", "models/YuE2-3B-MLX"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            log("[hum] encoder.safetensors downloaded successfully")
        else:
            log(f"[hum] encoder download failed: {result.stderr[:200]}")
    except Exception as e:
        log(f"[hum] encoder download error: {e}")


def _check_models():
    """Check if SheetSage2 + MERT models are downloaded. Returns status string."""
    sheetsage_dir = HF_CACHE / "models" / "m-a-p-SheetSage2"
    mert_dir = HF_CACHE / "models" / "m-a-p-MERT-v2-FullSong"

    def _is_downloaded(dir_path, required_files):
        if not dir_path.exists():
            return False
        for f in required_files:
            if not (dir_path / f).exists():
                return False
        return True

    # SheetSage2 is a code-heavy repo — check for config + key Python files
    # MERT has actual model weights in safetensors format
    sheetsage_ok = _is_downloaded(sheetsage_dir, ["config.json", "infer.py", "configuration_sheetsage2.py"])
    mert_ok = _is_downloaded(mert_dir, ["config.json", "model.safetensors"])

    if sheetsage_ok and mert_ok:
        return "✅ All models downloaded"
    return "❌ Models not downloaded — click 'Download Models' below"


def build_ui():
    """Build and return the Gradio Block UI."""

    with gr.Blocks(
        title="YUE2 // MLX",
    ) as demo:
        gr.Markdown(
            "# 🎵 YUE2 // MLX — Mac-Optimized Music Generation\n"
            "Generate complete songs (melody, chords, vocals, accompaniment) from text prompts "
            "using the YuE2-3B model on Apple Silicon via MLX. No PyTorch needed."
        )

        with gr.Tabs():
            # ── TAB 1: Generate ──────────────────────────────────────
            with gr.Tab("01 // GENERATE"):
                gr.Markdown("Enter Style+Lyrics (or use llm writing room) → Select model+settings → Generate song")
                gr.Markdown("> 📦 **Required models will be downloaded automatically during first run**")
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Song Parameters")
                        style_input = gr.Textbox(
                            label="Style",
                            placeholder="indie pop, bright acoustic guitar, warm vocal",
                            lines=3,
                        )
                        lyrics_input = gr.Textbox(
                            label="Lyrics",
                            placeholder="[Verse]\nSoft morning light...",
                            lines=7,
                        )
                        with gr.Row():
                            cot_input = gr.Radio(
                                choices=["full", "melody", "off"],
                                value="off",
                                label="ABC Plan",
                            )
                            seed_input = gr.Number(
                                value=-1,
                                label="Seed (-1 = random)",
                                precision=0,
                            )
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.0,
                                step=0.01, label="CFG",
                            )
                            steps_input = gr.Slider(
                                minimum=8, maximum=64, value=8,
                                step=1, label="NAR Steps",
                            )
                        with gr.Row():
                            tile_size_input = gr.Dropdown(
                                choices=[1024, 2048, 3072, 4096],
                                value=2048,
                                label="NAR Tile",
                            )
                            vae_tile_input = gr.Dropdown(
                                choices=[64, 128, 256, 512, 1024],
                                value=64,
                                label="VAE Tile",
                            )

                    with gr.Column(scale=2):
                        gr.Markdown("### Model Settings")
                        variant_input = gr.Radio(
                            choices=["BF16 (highest quality, 7 GB)", "8-bit (recommended, 4.2 GB)", "4-bit (fastest, 3.4 GB)"],
                            value="8-bit (recommended, 4.2 GB)",
                            label="Variant",
                        )
                        save_format_input = gr.Radio(
                            choices=["WAV", "MP3"],
                            value="WAV",
                            label="Format",
                        )

                        gr.Markdown("### Advanced Sampling")
                        with gr.Accordion("ABC Phase", open=False):
                            with gr.Row():
                                abc_temp = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temp")
                                abc_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                            with gr.Row():
                                abc_k = gr.Slider(1, 100, value=30, step=1, label="Top K")
                                abc_rep = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Rep Pen")
                        with gr.Accordion("Semantic Phase", open=False):
                            with gr.Row():
                                sem_temp = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temp")
                                sem_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top P")
                            with gr.Row():
                                sem_k = gr.Slider(1, 200, value=100, step=1, label="Top K")
                                sem_rep = gr.Slider(0.5, 2.0, value=1.2, step=0.01, label="Rep Pen")

                        gr.Markdown("### LoRA Adapters")
                        # Discover adapters at startup
                        lora_dir = Path(__file__).parent / "models" / "loras"
                        _lora_adapters = discover_loras(lora_dir) if lora_dir.exists() else []
                        lora_choices = [(a["name"], a["name"]) for a in _lora_adapters]
                        lora_adapters_input = gr.Dropdown(
                            choices=lora_choices,
                            value=[],
                            label="Select Adapters (stack multiple with Ctrl/Cmd)",
                            multiselect=True,
                            interactive=True,
                        )
                        lora_scale_input = gr.Slider(
                            minimum=0.0, maximum=4.0, value=1.0, step=0.1,
                            label="LoRA Scale (1.0 = as trained)",
                        )

                        generate_btn = gr.Button("🎵 Generate Song", variant="primary", size="lg", elem_classes="generate-btn")

                with gr.Accordion("Or enter an ABC score manually", open=False):
                    abc_score_input = gr.Textbox(
                        label="ABC Score",
                        placeholder="X:1\nT:My Song\nM:4/4\nL:1/8\nK:C\nV:Vocal\nC D E F G A B c |",
                        lines=6,
                        info="Paste an ABC notation score (required for cot=full/melody if not generating automatically)",
                    )

                with gr.Row():
                    audio_output = gr.Audio(
                        label="Generated Song",
                        type="numpy",
                        elem_classes="audio-container",
                    )
                abc_output = gr.Textbox(label="Generated ABC Score", lines=4)
                status_output = gr.Textbox(label="Status", elem_classes="status-box", interactive=False)

                generate_btn.click(
                    fn=_generate_song,
                    inputs=[style_input, lyrics_input, cot_input, seed_input,
                            cfg_scale, steps_input, variant_input, tile_size_input, vae_tile_input, save_format_input,
                            lora_adapters_input, lora_scale_input],
                    outputs=[audio_output, abc_output, status_output],
                )

            # ── TAB 2: COVER ───────────────────────────────────────
            with gr.Tab("02 // COVER"):
                gr.Markdown("Upload audio → Enter Style, Lyrics → Transcribe → Generates song from the transcription")
                gr.Markdown("> 📦 **Required models will be downloaded automatically during first run** (~3GB)")

                with gr.Row():
                    # LEFT side: Upload + Transcription + Generation Settings
                    with gr.Column(scale=2):
                        cover_audio = gr.Audio(
                            label="Upload Audio (MP3/WAV/M4A/OGG/FLAC/WEBM)",
                            sources=["upload"],
                            type="filepath",
                        )
                        cover_task = gr.Radio(
                            choices=["full", "melody-full", "melody-vocal"],
                            value="melody-full",
                            label="Transcription Task",
                        )
                        cover_task_desc = gr.Markdown(
                            value="**Melody-Full** — Transcribe the melody, let YuE2 write the full arrangement. Recommended for covers.",
                        )
                        cover_task.change(
                            fn=_update_task_desc,
                            inputs=cover_task,
                            outputs=cover_task_desc,
                        )
                        gr.Markdown("### Generation Settings")
                        cover_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                        cover_cfg = gr.Slider(minimum=0.5, maximum=3, value=1, step=0.01, label="CFG")
                        cover_steps = gr.Slider(minimum=8, maximum=64, value=8, step=1, label="NAR Steps")

                        gr.Markdown("### LoRA Adapters")
                        # Discover adapters at startup
                        lora_dir = Path(__file__).parent / "models" / "loras"
                        _lora_adapters = discover_loras(lora_dir) if lora_dir.exists() else []
                        lora_choices = [(a["name"], a["name"]) for a in _lora_adapters]
                        cover_lora_adapters = gr.Dropdown(
                            choices=lora_choices,
                            value=[],
                            label="Select Adapters (stack multiple with Ctrl/Cmd)",
                            multiselect=True,
                            interactive=True,
                        )
                        cover_lora_scale = gr.Slider(
                            minimum=0.0, maximum=4.0, value=1.0, step=0.1,
                            label="LoRA Scale (1.0 = as trained)",
                        )

                    # RIGHT side: Song Parameters + Save Format
                    with gr.Column(scale=3):
                        gr.Markdown("### Song Parameters")
                        cover_style = gr.Textbox(
                            label="Style (required)",
                            placeholder="indie pop, bright acoustic guitar, warm vocal",
                            lines=9,
                        )
                        cover_lyrics = gr.Textbox(
                            label="Lyrics (optional — song will use ABC content if left blank)",
                            placeholder="[Verse]\nSoft morning light...",
                            lines=16,
                        )
                        gr.Markdown("### Output")
                        cover_save_format = gr.Radio(
                            choices=["WAV", "MP3"],
                            value="MP3",
                            label="Format",
                        )
                        gr.Markdown("*(Model variant shared with Generate tab)*")
                        cover_btn = gr.Button("🎤 Transcribe & Generate", variant="primary", size="lg", elem_classes="generate-btn")

                cover_audio_output = gr.Audio(label="Generated Cover Song")
                with gr.Accordion("Transcribed ABC Score", open=False):
                    cover_abc_output = gr.Textbox(label="", lines=15)
                cover_status = gr.Textbox(label="Status", lines=3)

                cover_btn.click(
                    fn=_cover_song,
                    inputs=[cover_audio, cover_task, cover_style, cover_lyrics, cover_seed, cover_cfg, cover_steps, variant_input, cover_save_format, cover_lora_adapters, cover_lora_scale],
                    outputs=[cover_audio_output, cover_abc_output, cover_status],
                )

            # ── TAB 3: HUM TO SONG ───────────────────────────────
            with gr.Tab("03 // HUM TO SONG"):
                gr.Markdown("Hum a melody → Add style & lyrics → Generate a complete song")
                gr.Markdown("> 🎤 **Record/upload your hum** → **Select melody mode** → **Choose hum adapter** → **Generate**")

                with gr.Row():
                    # LEFT side: hum inputs, settings, and loras
                    with gr.Column():
                        gr.Markdown("### Hum Input")
                        hum_audio = gr.Audio(
                            label="Drop a recording of your hum or click to choose",
                            type="filepath",
                            editable=True,
                            sources=["upload", "microphone"],
                        )

                        gr.Markdown("### Melody Mode")
                        hum_melody = gr.Radio(
                            choices=["continue", "hum_only", "ignore"],
                            value="continue",
                            label="Melody",
                        )
                        hum_melody_desc = gr.Markdown(
                            value="**Continue** — The hum's notes open the score; YuE2 writes the rest of the song around them. **Hum Only** — Hum is the complete melody, like a cover; the song is as long as the hum. **Ignore** — YuE2 writes the melody, hum shapes phrasing via lora adapter."
                        )
                        hum_melody.change(
                            fn=_update_melody_desc,
                            inputs=hum_melody,
                            outputs=hum_melody_desc,
                        )


                        gr.Markdown("### Hum Settings")
                        with gr.Row():
                            with gr.Column(scale=2):
                                lora_dir = Path(__file__).parent / "models" / "loras"
                                _hum_adapters = list_hum_adapters(lora_dir) if lora_dir.exists() else []
                                hum_adapter_choices = [a["name"] for a in _hum_adapters] if _hum_adapters else [None]
                                print(f"[hum] Found {len(_hum_adapters)} hum adapter(s): {hum_adapter_choices}")

                                hum_adapter_input = gr.Dropdown(
                                    label="Hum Lora Adapter",
                                    choices=hum_adapter_choices,
                                    value=None,
                                    allow_custom_value=False,
                                )
                                with gr.Row():
                                    hum_adapter_info = gr.Markdown(f"> **Found**: {len(_hum_adapters)} adapter(s)")

                            with gr.Column(scale=3):
                                hum_influence = gr.Slider(
                                    minimum=0.0, maximum=3.0, value=1.0, step=0.05,
                                    label="Hum influence",
                                    info="1 = as trained, 0 = ignore hum's phrasing, >1 exaggerates it (costs ~2x synthesis time)"
                                )
                                hum_offset = gr.Slider(
                                    minimum=0.0, maximum=600.0, value=0.0, step=0.1,
                                    label="Hum starts at (seconds into song)",
                                    info="0 = song opens with your hum"
                                )

                        gr.Markdown("### LoRA Adapters (regular, separate from hum adapter)")
                        _lora_adapters = discover_loras(lora_dir) if lora_dir.exists() else []
                        lora_choices = [(a["name"], a["name"]) for a in _lora_adapters if a.get("kind") == "lora"]
                        print(f"[hum] Found {len(lora_choices)} LoRA adapter(s)")

                        hum_lora_adapters = gr.Dropdown(
                            choices=lora_choices,
                            value=[],
                            label="Select Adapters (stack multiple with Ctrl/Cmd)",
                            multiselect=True,
                            interactive=True,
                        )
                        hum_lora_scale = gr.Slider(
                            minimum=0.0, maximum=4.0, value=1.0, step=0.1,
                            label="LoRA Scale (1.0 = as trained)",
                        )

                    # RIGHT side: song parameters, sampling, and output
                    with gr.Column():
                        gr.Markdown("### Song Parameters")
                        hum_style = gr.Textbox(
                            label="Style",
                            placeholder="e.g. indie folk, male vocal, acoustic guitar, stomps and claps, 120 BPM",
                            lines=8,
                        )
                        hum_lyrics = gr.Textbox(
                            label="Lyrics",
                            placeholder="[Verse]\n...\n[Chorus]\n...",
                            lines=18,
                        )
                        gr.Markdown("### Sampling Settings")
                        hum_seed = gr.Number(label="Seed", value=-1, precision=0, info="-1 for random")
                        with gr.Row():
                            hum_cfg = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.0,
                                step=0.01, label="CFG",
                            )
                            hum_steps = gr.Slider(
                                minimum=8, maximum=64, value=8,
                                step=1, label="NAR Steps",
                            )
                        gr.Markdown("*(Model variant shared with Generate tab)*")

                        gr.Markdown("### Output")
                        hum_format = gr.Radio(
                            choices=["WAV", "MP3"],
                            value="WAV",
                            label="Save format",
                        )
                        

                with gr.Column():
                        hum_generate_btn = gr.Button("🎤 Create song from hum", variant="primary", size="lg", elem_classes="generate-btn")
                        hum_audio_output = gr.Audio(label="Generated song", type="numpy", elem_classes="audio-container")
                        hum_status = gr.Textbox(label="Status", elem_classes="status-box", interactive=False)
                        
                hum_generate_btn.click(
                    fn=_hum_song,
                    inputs=[hum_audio, hum_melody, hum_adapter_input, hum_influence, hum_offset,
                            hum_style, hum_lyrics, hum_seed, hum_cfg, hum_steps,
                            hum_format,
                            hum_lora_adapters, hum_lora_scale],
                    outputs=[hum_audio_output, hum_status],
                )

            # ── TAB 4: LLM Writing Room ──────────────────────────────
            with gr.Tab("04 // WRITING ROOM"):
                gr.Markdown(
                    "### LLM Writing Room\n"
                    "Use an OpenAI-compatible LLM (LM Studio, Ollama, text-generation-webui, etc.) "
                    "to help write lyrics and compose songs from your ideas."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### LLM Connection")
                        llm_api_url = gr.Textbox(
                            label="API URL",
                            value="http://127.0.0.1:1234/v1/chat/completions",
                            info="LM Studio default: http://127.0.0.1:1234/v1/chat/completions",
                        )
                        llm_model = gr.Textbox(
                            label="Model Name",
                            placeholder="auto-detect",
                            info="Leave blank to auto-detect from API, or specify one",
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key (optional)",
                            placeholder="sk-your-key",
                            info="For APIs that require authentication",
                        )
                        with gr.Row():
                            llm_max_tokens = gr.Slider(1, 8192, value=4096, step=1, label="Max Tokens")
                            llm_temp = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")

                    with gr.Column(scale=3):
                        gr.Markdown("### Quick Actions")
                        idea_input = gr.Textbox(
                            label="Your Idea",
                            placeholder="A nostalgic song about summer evenings, acoustic guitar, warm vocals...",
                            lines=2,
                        )
                        with gr.Row():
                            btn_gen_lyrics = gr.Button("✨ Generate Lyrics from Idea", elem_classes="llm-btn")
                            btn_gen_style = gr.Button("🎨 Generate Style from Idea", elem_classes="llm-btn")

                        # ── Chat (hidden — broken, kept for future fix) ──
                        with gr.Column(visible=False):
                            gr.Markdown("### Chat")
                            llm_chat_input = gr.Textbox(
                                label="Chat Message",
                                placeholder="Make the chorus more uplifting...",
                                lines=2,
                            )
                            with gr.Row():
                                btn_send = gr.Button("💬 Send", variant="secondary", elem_classes="llm-btn")
                                btn_clear = gr.Button("🗑️ Clear Chat", variant="secondary", elem_classes="llm-btn")

                            llm_chat_output = gr.Chatbot(
                                label="Conversation",
                                height=300,
                            )

                        gr.Markdown("### Lyric Editing")
                        current_lyrics = gr.Textbox(
                            label="Current Lyrics",
                            placeholder="Paste your lyrics here to edit...",
                            lines=6,
                        )
                        with gr.Row():
                            expand_dir = gr.Radio(
                                choices=["Add another verse", "Add a bridge",
                                         "Add a pre-chorus", "Rewrite the chorus",
                                         "Make it more poetic", "Make it simpler"],
                                value="Add another verse",
                                label="Direction",
                            )
                            btn_expand = gr.Button("✏️ Expand Lyrics", elem_classes="llm-btn")

                        gr.Markdown("### Output")
                        llm_output = gr.Textbox(
                            label="LLM Response",
                            lines=6,
                            interactive=False,
                        )
                        with gr.Row():
                            btn_copy_lyrics = gr.Button("📋 Copy Lyrics → Generate Tab", variant="secondary", size="sm")
                        with gr.Row():
                            btn_copy_style = gr.Button("📋 Copy Style → Generate Tab", variant="secondary", size="sm")
                        gr.Markdown("---")
                        with gr.Row():
                            btn_copy_lyrics_cover = gr.Button("📋 Copy Lyrics → Cover Tab", variant="secondary", size="sm")
                        with gr.Row():
                            btn_copy_style_cover = gr.Button("📋 Copy Style → Cover Tab", variant="secondary", size="sm")

                # State to hold generated content for copying
                gen_lyrics_state = gr.State(value="")
                gen_style_state = gr.State(value="")

                def _on_gen_lyrics(idea, api_url, model, max_tokens, temp, api_key):
                    messages, content = _generate_lyrics_from_idea(idea, api_url, model, max_tokens, temp, api_key)
                    lyrics, style = _parse_lyrics_from_response(content)
                    return messages, content, lyrics, style, lyrics, style

                def _on_gen_style(idea, api_url, model, max_tokens, temp, api_key):
                    messages, content = _generate_style_from_idea(idea, api_url, model, max_tokens, temp, api_key)
                    lyrics, style = _parse_lyrics_from_response(content)
                    return messages, content, lyrics, style, lyrics, style

                # Quick action handlers
                btn_gen_lyrics.click(
                    fn=_on_gen_lyrics,
                    inputs=[idea_input, llm_api_url, llm_model, llm_max_tokens, llm_temp, llm_api_key],
                    outputs=[llm_chat_output, llm_output, gen_lyrics_state, gen_style_state, current_lyrics, style_input],
                )
                btn_gen_style.click(
                    fn=_on_gen_style,
                    inputs=[idea_input, llm_api_url, llm_model, llm_max_tokens, llm_temp, llm_api_key],
                    outputs=[llm_chat_output, llm_output, gen_lyrics_state, gen_style_state, current_lyrics, style_input],
                )
                btn_send.click(
                    fn=_llm_chat,
                    inputs=[llm_chat_output, llm_api_url, llm_model, llm_max_tokens, llm_temp, llm_api_key],
                    outputs=[llm_chat_output, llm_output],
                )
                btn_clear.click(
                    fn=lambda: ([], ""),
                    outputs=[llm_chat_output, llm_output],
                )
                btn_expand.click(
                    fn=_expand_lyrics,
                    inputs=[current_lyrics, expand_dir, llm_api_url, llm_model, llm_max_tokens, llm_temp, llm_api_key],
                    outputs=[llm_chat_output, llm_output],
                )

                # Copy buttons — parse lyrics/style from LLM response window and copy to Generate tab
                def _copy_lyrics_from_output(raw_response):
                    lyrics, _ = _parse_lyrics_from_response(raw_response)
                    return lyrics

                def _copy_style_from_output(raw_response):
                    _, style = _parse_lyrics_from_response(raw_response)
                    return style

                btn_copy_lyrics.click(
                    fn=_copy_lyrics_from_output,
                    inputs=[llm_output],
                    outputs=[lyrics_input],
                )
                btn_copy_style.click(
                    fn=_copy_style_from_output,
                    inputs=[llm_output],
                    outputs=[style_input],
                )
                btn_copy_lyrics_cover.click(
                    fn=_copy_lyrics_from_output,
                    inputs=[llm_output],
                    outputs=[cover_lyrics],
                )
                btn_copy_style_cover.click(
                    fn=_copy_style_from_output,
                    inputs=[llm_output],
                    outputs=[cover_style],
                )

            # ── TAB 5: INFO ────────────────────────────────────────
            with gr.Tab("05 // INFO"):
                gr.Markdown("### Generation Parameters")
                gr.Markdown(
                    "| Parameter | Description | Recommended | Default(lowest ram) |\n"
                    "|-----------|-------------|-------------|---------------------|\n"
                    "| **CFG Scale** | Classifier-free guidance. Higher = follows prompt more strictly | 1.0 | 1.0 |\n"
                    "| **NAR Steps** | Midpoint ODE steps. More = better quality but slower | 12 | 8 |\n"
                    "| **NAR Tile** | Process frames in tiles. Smaller = less RAM, possible lower quality/artifacts | 4096 | 2048 |\n"
                    "| **VAE Tile** | VAE decode tile size. Lower = less RAM during decode | 512 | 64 |\n"
                    "| **ABC (Symbolic) Plan** | ABC score generation. FULL, MELODY only,`off` skips : saves ~30% time | on | off |\n"
                    "| **Seed** | Set for reproducible results | any | any |\n"
                )
                gr.Markdown("### Cover Song")
                gr.Markdown(
                    "| Parameter | Description |\n"
                    "|-----------|-------------|\n"
                    "| **FULL** | Transcribe the whole arrangement and follow it closely (closest to the original) |\n"
                    "| **Melody-Full** | Transcribe the melody, let YuE2 write the full arrangement. Recommended for covers. |\n"
                    "| **Melody-Vocal** | Transcribe the melody and follow it with the vocal line only. |\n"
                )

                gr.Markdown("### Memory & Performance")
                gr.Markdown(
                    "| Variant | Model Size | Approx RAM |\n"
                    "|---------|-----------|------------------|\n"
                    "| **4-bit** | ~2.1 GB | ~4-8 GB peak |\n"
                    "| **8-bit** | ~4.2 GB | ~6-9 GB peak |\n"
                    "| **BF16** | ~7 GB | ~8-15 GB |\n\n"
                    "> **VAE Tile** — Lower this to reduce memory spikes during decode. Default 64 for low RAM.\n"
                    "> **NAR Tile** — Process frames in tiles. Lower = faster, but may introduce artifacts sometimes.\n"
                    "> Benchmarks on M1 Max 64GB: 8-bit ~4min song = ~7GB peak = renders slightly faster than realtime on default settings.\n"
                )

                gr.Markdown("### Hardware Requirements")
                gr.Markdown(
                    "| Component | Minimum | Recommended |\n"
                    "|-----------|---------|-------------|\n"
                    "| **Chip** | M1/M2/M3/M4 | M1 Pro/Max or better |\n"
                    "| **RAM** | 16 GB | 16 GB+ |\n"
                    "| **Storage** | 5 GB free | 12 GB free |\n"
                    "| **Generation** | ~1.2 min (2 min song) | ~3.1 min (4 min song) |\n"
                )

                gr.Markdown("### Model Directories")
                gr.Markdown(
                    "| Model | Dir |\n"
                    "|-----------|---------|\n"
                    "| **YuE2 MLX** | /models/YuE2-3B-MLX |\n"
                    "| **Cover(Transcription+sheetsage)** | /models/hf_cache |\n"
                    "| **LoRA** | /models/loras |\n"
                )
                
                gr.Markdown("### Recomended Loras")
                gr.Markdown(
                    "| Lora | Description | Download Link |\n"
                    "|-----------|---------|-------------|\n"
                    "| **YuE2-instrumental-cot-full-loras** | Makes the model write instrumental music with a section plan. Recomended to use with cot(ABC plan)=full | https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras/resolve/main/ar_lora_inst_v3abc_comfyui.safetensors |\n"
                    "| **Hum-to-Song** | Hum a melody for 10 to 30 seconds, add a style line and lyrics, get a finished song that keeps your melody, builds a structure around it, and continues long after your hum stops. | https://huggingface.co/Mothersuperior/YuE2-hum-to-song/resolve/main/humsong_yue2_adapter_v1_comfy.safetensors (recomended),  https://huggingface.co/Mothersuperior/YuE2-hum-to-song/resolve/main/hum_adapter_v1_combined.safetensors |\n"
                )
            
                gr.Markdown("### Tips")
                gr.Markdown(
                    "- **8-bit variant** gives near-bF16 quality at ~2x decode speed\n"
                    "- Use the **Writing Room** to brainstorm ideas before generating\n"
                    "- Set a **seed** for reproducible results\n"
                    "- **cot=full** generates chord-annotated ABC scores for editing\n"
                    "- For Hum to song, make sure the input hum audio is loud enough. Also humming in 'la la la la...' tends to produce better results in my testing. \n"
                    "- The MLX backend requires **no PyTorch** — pure Apple Metal\n"
                    "- All generated songs auto-save to `outputs/<song_name>/` with metadata.json\n"
                    "- the song(along with metadata) used for benchmarking on m1 max is saved in 'examples' folder\n"
                )

                gr.Markdown("### Metadata Fields")
                gr.Markdown(
                    "`metadata.json` includes: `song_name`, `timestamp`, `duration_s`, "
                    "`gen_time_s`, `seed`, `cot`, `steps`, `cfg_scale`, `nar_tile`, "
                    "`vae_tile`, `style`, `lyrics`, `save_format`, `mp3_bitrate`, "
                    "`filename`, `filepath`, and `abc` (if applicable).\n"
                )

    return demo


def _cover_song(audio_file, task, style, lyrics, seed, cfg_scale, steps, variant,
                save_format, lora_adapters, lora_scale, progress=gr.Progress()):
    """Transcribe audio to ABC, then generate song from the transcribed score."""
    model_dir = "./models/YuE2-3B-MLX"
    tile_size = 4096
    vae_tile = 128
    mp3_bitrate = "320k"
    import tempfile
    from pathlib import Path
    import json
    import time

    log = print
    _CLEARED.clear()
    overall_start = time.perf_counter()

    # Update model status
    model_status = _check_models()

    # Validate task
    if task not in {"full", "melody-full", "melody-vocal"}:
        raise gr.Error("Invalid task. Must be: full, melody-full, or melody-vocal")

    # Generate random seed if -1
    if seed == -1:
        import secrets
        seed = secrets.randbelow(2**32)

    # Create temp directories
    tmp_dir = Path(tempfile.mkdtemp())
    transcription_dir = tmp_dir / "transcription"
    transcription_dir.mkdir(parents=True, exist_ok=True)
    song_dir = tmp_dir / "song"
    song_dir.mkdir(parents=True, exist_ok=True)

    # audio_file is either a file path (upload) or temp file (microphone recording)
    audio_path = Path(audio_file) if audio_file else None
    if audio_path and not audio_path.exists():
        raise gr.Error("No audio provided. Please upload a file or record audio.")
    log(f"[cover] using audio: {audio_path}")

    # Step 1: Transcribe
    log("[transcription] transcribing audio")
    progress(0.0, desc="Transcribing audio...")
    t_transcribe_start = time.perf_counter()

    try:
        from lyra.transcription.pipeline import transcribe
        import mlx.core as mx

        # Release YuE2 models BEFORE transcription (like yue2_studio does)
        # This prevents SheetSage2 + MERT (~3GB) from stacking on top of YuE2
        if hasattr(_cover_song, "_pipe") and _cover_song._pipe is not None:
            _cover_song._pipe.release_models()
            _cover_song._pipe = None
            log("[cover] released YuE2 models before transcription")

        # Set strict MLX memory limit BEFORE transcription to prevent stacking
        # This forces MLX to evict unused tensors when memory gets tight
        try:
            import psutil
            _total_ram_gib = psutil.virtual_memory().total / (1024**3)
            mx.set_memory_limit(int((_total_ram_gib - 12) * 1024 * 1024 * 1024))  # Reserve 12GB for OS + others
            mx.set_cache_limit(128 * 1024 * 1024)  # 128MB cache
            log(f"[cover] set MLX memory limit: {_total_ram_gib - 12:.0f}GB")
        except Exception as e:
            log(f"[cover] memory limit warning: {e}")

        # Clear MLX cache before transcription
        mx.clear_cache()
        import gc
        gc.collect()

        # Debug: print where models will be downloaded
        log(f"[cover] transcribe cache_dir: {HF_CACHE}")
        log(f"[cover] SheetSage2 path: {HF_CACHE}/models/m-a-p-SheetSage2")
        log(f"[cover] MERT path: {HF_CACHE}/models/m-a-p-MERT-v2-FullSong")

        result = transcribe(
            audio=audio_path,
            output=transcription_dir,
            task=task,
            cache_dir=str(HF_CACHE),
            cancelled=lambda: _CLEARED.is_set(),
            progress=lambda info: progress(
                info.get("window", 0) / info.get("windows", 1),
                desc=f"Transcribing... ({info.get('stage', '')})"
            ) if info.get("stage") == "encoding" else None
        )
        log("[transcription] transcription complete")

        # Aggressively free MLX memory after transcription (SheetSage2 + MERT)
        # Reset memory limit to full for generation phase
        try:
            import psutil
            _total_ram_gib = psutil.virtual_memory().total / (1024**3)
            mx.set_memory_limit(int((_total_ram_gib - 8) * 1024 * 1024 * 1024))  # Back to normal for generation
            del result
            mx.clear_cache()
            gc.collect()
            log(f"[cover] freed MLX memory after transcription, limit reset to {_total_ram_gib - 8:.0f}GB")
        except Exception as e:
            log(f"[cover] memory cleanup warning: {e}")

    except Exception as e:
        raise gr.Error(f"Transcription failed: {type(e).__name__}: {e}")

    t_transcribe_end = time.perf_counter()
    t_transcribe_s = t_transcribe_end - t_transcribe_start
    log(f"[cover] transcription took {t_transcribe_s:.1f}s")

    # Step 2: Read ABC score
    abc_path = transcription_dir / "score.abc"
    if not abc_path.exists():
        raise gr.Error("Transcription did not produce an ABC file")
    abc_text = abc_path.read_text(encoding="utf-8").strip()

    if not abc_text:
        raise gr.Error("Transcription produced empty ABC score")

    log("[transcription] ABC score generated")

    # Step 3: Generate song from ABC
    log("[cover] generating song from ABC")
    progress(0.7, desc="Generating song...")
    t_gen_start = time.perf_counter()

    try:
        # Load or create pipeline
        if not hasattr(_cover_song, "_pipe"):
            _cover_song._pipe = None

        if _cover_song._pipe is None:
            _cover_song._pipe = Yue2PipelineMLX(
                model_root=model_dir,
                variant=_VARIANT_MAP.get(variant, ModelVariant.EIGHT_BIT),
                lora_adapters=lora_adapters,
                lora_scale=lora_scale,
                log=log,
            )

        result = _cover_song._pipe(
            style=style.strip() if style else "",
            lyrics=lyrics.strip() if lyrics else "",
            cot="full" if task == "full" else "melody",
            seed=seed,
            abc=abc_text,
            cfg_scale=float(cfg_scale) if cfg_scale else None,
            steps=int(steps),
            tile_size=int(tile_size),
            vae_tile=int(vae_tile),
        )
    except Exception as e:
        raise gr.Error(f"Song generation failed: {type(e).__name__}: {e}")

    t_gen_end = time.perf_counter()
    t_gen_s = t_gen_end - t_gen_start
    overall_end = time.perf_counter()
    t_total_s = overall_end - overall_start
    log(f"[cover] generation took {t_gen_s:.1f}s, total {t_total_s:.1f}s")

    audio = result["audio"]
    audio_for_save = audio  # Already normalized float32
    duration_s = audio.shape[0] / 48000.0

    # Free MLX memory after generation
    try:
        import mlx.core as mx
        mx.clear_cache()
        import gc
        gc.collect()
        log("[cover] freed MLX memory after generation")
    except Exception as e:
        log(f"[cover] memory cleanup warning: {e}")

    # Save to outputs folder
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    song_name = f"cover_{timestamp}_s{seed}"
    song_path = outputs_dir / song_name
    song_path.mkdir(exist_ok=True)

    # Save audio (WAV or MP3)
    save_format = save_format or "WAV"
    is_mp3 = save_format == "MP3"
    mp3_bitrate = mp3_bitrate or "192k"
    saved_filepath = None

    if is_mp3:
        # Save as MP3 using ffmpeg
        wav_path = song_path / f"{song_name}.wav"
        mp3_path = song_path / f"{song_name}.mp3"
        try:
            if len(audio.shape) == 1:
                audio = audio.reshape(-1, 1)
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            import wave
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1 if len(pcm.shape) == 1 else pcm.shape[1])
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm.tobytes())
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", mp3_bitrate, "-ar", "48000", str(mp3_path)],
                capture_output=True, timeout=120, check=True
            )
            wav_path.unlink(missing_ok=True)
            filename = f"{song_name}.mp3"
            saved_filepath = str(mp3_path)
            info = f"Cover saved: {filename}"
        except Exception as e:
            info = f"Save error: {e}"
    else:
        # Save as WAV (default)
        filename = f"{song_name}.wav"
        filepath = song_path / filename
        try:
            if len(audio.shape) == 1:
                audio = audio.reshape(-1, 1)
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            import wave
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1 if len(pcm.shape) == 1 else pcm.shape[1])
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm.tobytes())
            saved_filepath = str(filepath)
            info = f"Cover saved: {filename}"
        except Exception as e:
            info = f"Save error: {e}"

    # Save metadata
    metadata = {
        "song_name": song_name,
        "timestamp": timestamp,
        "duration_s": round(duration_s, 2),
        "transcribe_time_s": round(t_transcribe_s, 1),
        "generation_time_s": round(t_gen_s, 1),
        "total_time_s": round(t_total_s, 1),
        "seed": seed,
        "task": task,
        "cot": "full" if task == "full" else "melody",
        "cfg_scale": float(cfg_scale) if cfg_scale else None,
        "steps": int(steps),
        "variant": variant,
        "tile_size": int(tile_size),
        "vae_tile": int(vae_tile),
        "style": style.strip() if style else "",
        "lyrics": lyrics.strip() if lyrics else "",
        "save_format": save_format,
        "mp3_bitrate": mp3_bitrate if is_mp3 else None,
        "filename": filename,
        "filepath": saved_filepath,
        "abc": abc_text,
        "transcription_dir": str(transcription_dir),
        "lora_adapters": lora_adapters or [],
        "lora_scale": lora_scale,
    }
    try:
        metadata_path = song_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        info += "\nMetadata: metadata.json"
    except Exception as e:
        info += f"\nMetadata save error: {e}"

    # Return audio tuple for playback, filepath for download
    # If MP3 saved, return file path so download button saves MP3
    if is_mp3 and saved_filepath:
        audio_output = saved_filepath
    else:
        audio_output = (48000, audio)

    abc_output = abc_text
    status_output = info
    pipeline_status = "Done"

    # Clean up temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return audio_output, abc_output, pipeline_status, status_output


def _hum_song(hum_audio, melody, hum_adapter_name, hum_influence, hum_offset,
              style, lyrics, seed, cfg_scale, steps,
              save_format,
              lora_adapters, lora_scale, progress=gr.Progress()):
    """Hum-to-song: transcribe hum, continue score, condition decoder with prosody adapter."""
    model_dir = "./models/YuE2-3B-MLX"
    mp3_bitrate = "320k"
    import tempfile
    from pathlib import Path
    import json
    import time
    import soundfile as sf
    import subprocess

    log = print
    _CLEARED.clear()
    overall_start = time.perf_counter()

    # Read variant from the loaded pipeline (shared with Generate tab)
    if _PIPE and hasattr(_PIPE, 'variant'):
        variant = _PIPE.variant
    else:
        variant = "8bit"
    tile_size = 4096
    vae_tile = 64

    # Validate inputs
    if not hum_audio:
        raise gr.Error("No hum audio provided. Please upload or record audio.")
    if not style.strip():
        raise gr.Error("Style is required")
    if not lyrics.strip():
        raise gr.Error("Lyrics are required")

    # Generate random seed if -1
    if seed == -1:
        import secrets
        seed = secrets.randbelow(2**32)

    # Create temp directories
    tmp_dir = Path(tempfile.mkdtemp())
    transcription_dir = tmp_dir / "transcription"
    transcription_dir.mkdir(parents=True, exist_ok=True)
    hum_dir = tmp_dir / "hum"
    hum_dir.mkdir(parents=True, exist_ok=True)
    song_dir = tmp_dir / "song"
    song_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(hum_audio)
    if not audio_path.exists():
        raise gr.Error(f"Hum audio file not found: {hum_audio}")

    # Step 1: Transcribe hum to ABC
    log("[hum] transcribing hum audio")
    progress(0.0, desc="Transcribing hum...")
    t_transcribe_start = time.perf_counter()

    try:
        from lyra.transcription.pipeline import transcribe
        import mlx.core as mx

        # Release YuE2 models BEFORE transcription
        if hasattr(_hum_song, "_pipe") and _hum_song._pipe is not None:
            _hum_song._pipe.release_models()
            _hum_song._pipe = None
            log("[hum] released YuE2 models before transcription")

        # Set strict MLX memory limit
        try:
            import psutil
            _total_ram_gib = psutil.virtual_memory().total / (1024**3)
            mx.set_memory_limit(int((_total_ram_gib - 12) * 1024 * 1024 * 1024))
            mx.set_cache_limit(128 * 1024 * 1024)
        except Exception as e:
            log(f"[hum] memory limit warning: {e}")

        mx.clear_cache()
        import gc
        gc.collect()

        result = transcribe(
            audio=audio_path,
            output=transcription_dir,
            task="melody-vocal",
            cache_dir=str(HF_CACHE),
            cancelled=lambda: _CLEARED.is_set(),
            progress=lambda info: progress(
                info.get("window", 0) / info.get("windows", 1),
                desc=f"Transcribing... ({info.get('stage', '')})"
            ) if info.get("stage") == "encoding" else None
        )
        log("[hum] transcription complete")
    except Exception as e:
        raise gr.Error(f"Transcription failed: {type(e).__name__}: {e}")

    t_transcribe_end = time.perf_counter()
    t_transcribe_s = t_transcribe_end - t_transcribe_start

    # Step 2: Read ABC score
    abc_path = transcription_dir / "score.abc"
    if not abc_path.exists():
        raise gr.Error("Transcription did not produce an ABC file")
    hum_abc = abc_path.read_text(encoding="utf-8").strip()

    if not hum_abc:
        raise gr.Error("Transcription produced empty ABC score")

    # Trim open score if melody=continue
    if melody == "continue":
        from hum import trim_open_score, open_score_has_notes
        hum_abc_trimmed = trim_open_score(hum_abc)
        log(f"[hum] ABC before trim: {repr(hum_abc[:200])}")
        log(f"[hum] ABC after trim: {repr(hum_abc_trimmed[:200])}")
        if not open_score_has_notes(hum_abc_trimmed):
            log("[hum] No notes detected in transcription, switching to melody=ignore mode")
            melody = "ignore"  # Auto-switch to ignore mode
            hum_abc = hum_abc_trimmed  # Keep the ABC for metadata

    log(f"[hum] ABC score: {len(hum_abc)} chars")

    # Free memory after transcription
    try:
        import psutil
        _total_ram_gib = psutil.virtual_memory().total / (1024**3)
        mx.set_memory_limit(int((_total_ram_gib - 8) * 1024 * 1024 * 1024))
        del result
        mx.clear_cache()
        gc.collect()
    except Exception as e:
        log(f"[hum] memory cleanup warning: {e}")

    # Step 3: Analyze hum and create prosody carrier (if adapter provided)
    carrier_latents = None
    adapter_info = None
    if hum_adapter_name:
        log(f"[hum] loading hum adapter: {hum_adapter_name}")
        progress(0.4, desc="Loading hum adapter...")

        # Find adapter
        lora_dir = Path(__file__).parent / "models" / "loras"
        all_loras = discover_loras(lora_dir)
        adapter_info = None
        for lora in all_loras:
            if lora["name"] == hum_adapter_name and lora.get("kind") == "hum":
                adapter_info = lora
                break

        if not adapter_info:
            raise gr.Error(f"Hum adapter '{hum_adapter_name}' not found in models/loras/")

        # Decode hum audio to PCM
        log("[hum] decoding hum audio")
        try:
            import librosa
            samples, sr = librosa.load(str(audio_path), sr=48000, mono=True)
        except Exception as e:
            raise gr.Error(f"Failed to read hum audio: {e}")

        # Analyse hum (pitch tracking + carrier)
        log("[hum] analysing hum (pitch tracking + carrier)")
        progress(0.5, desc="Analysing hum...")
        try:
            from hum import analyse_hum, carrier_stereo
            analysis = analyse_hum(samples, cancelled=lambda: _CLEARED.is_set())
            log(f"[hum] hum: {analysis.duration_s:.1f}s, voiced {analysis.voiced_fraction:.0%}")

            # Make stereo for VAE encoder
            stereo = carrier_stereo(analysis.carrier)

            # Save carrier for debugging
            sf.write(str(hum_dir / "carrier.flac"), stereo, 48000, subtype="PCM_16")

            # Encode carrier to latents
            log("[hum] encoding carrier to latents")
            progress(0.6, desc="Encoding carrier...")
            # Ensure encoder is downloaded (lazy, only once)
            _ensure_encoder_downloaded()
            from vae_encoder import load_encoder, encode
            # VAE encoder is shared across all variants, stored in models root
            vae_dir = Path(__file__).parent / "models" / "YuE2-3B-MLX"
            encoder = load_encoder(vae_dir)
            carrier_latents = encode(encoder, stereo, cancelled=lambda: _CLEARED.is_set())
            del encoder
            log(f"[hum] carrier latents: {carrier_latents.shape}")
            np.save(str(hum_dir / "carrier_latents.npy"), carrier_latents)
        except Exception as e:
            log(f"[hum] hum analysis error: {e}")
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Hum analysis failed: {type(e).__name__}: {e}")

    # Step 4: Generate song
    log("[hum] generating song")
    progress(0.8, desc="Generating song...")
    t_gen_start = time.perf_counter()

    try:
        # Load or create pipeline
        if not hasattr(_hum_song, "_pipe"):
            _hum_song._pipe = None

        # Parse lora_adapters - convert names to path dicts
        parsed_loras = []
        if lora_adapters:
            lora_dir = Path(__file__).parent / "models" / "loras"
            all_loras = discover_loras(lora_dir)
            for lora_name in lora_adapters:
                for lora in all_loras:
                    if lora["name"] == lora_name and lora.get("kind") == "lora":
                        parsed_loras.append({
                            "name": lora["name"],
                            "path": lora["path"],
                            "kind": lora["kind"],
                            "file_hash": lora["file_hash"],
                            "scale": lora["scale"],
                        })
                        break

        if _hum_song._pipe is None:
            # Use default variant (shared with Generate tab)
            _hum_song._pipe = Yue2PipelineMLX(
                model_root=model_dir,
                variant=ModelVariant.EIGHT_BIT,
                lora_adapters=parsed_loras,
                lora_scale=lora_scale,
                log=log,
            )

        # Prepare ABC for generation
        if melody == "hum_only":
            # Use hum_abc as the complete melody
            abc_for_pipe = hum_abc
        elif melody == "continue":
            # Let AR continue the open score
            abc_for_pipe = None
        else:  # ignore
            # Planner writes its own melody
            abc_for_pipe = None

        result = _hum_song._pipe(
            style=style.strip(),
            lyrics=lyrics.strip(),
            cot="melody",  # Hum always uses melody mode
            seed=seed,
            abc=abc_for_pipe,
            cfg_scale=float(cfg_scale) if cfg_scale else None,
            steps=int(steps),
        )
    except Exception as e:
        log(f"[hum] generation error: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Song generation failed: {type(e).__name__}: {e}")

    t_gen_end = time.perf_counter()
    t_gen_s = t_gen_end - t_gen_start
    overall_end = time.perf_counter()
    t_total_s = overall_end - overall_start
    log(f"[hum] generation took {t_gen_s:.1f}s, total {t_total_s:.1f}s")

    audio = result["audio"]
    duration_s = audio.shape[0] / 48000.0

    # Save to outputs folder
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    song_name = f"hum_{timestamp}_s{seed}"
    song_path = outputs_dir / song_name
    song_path.mkdir(exist_ok=True)

    # Copy original hum source audio to outputs
    try:
        src_ext = audio_path.suffix.lstrip(".") or "wav"
        hum_source_path = song_path / f"{song_name}_hum_source.{src_ext}"
        import shutil
        shutil.copy2(audio_path, hum_source_path)
    except Exception as e:
        log(f"[hum] warning: failed to copy hum source: {e}")

    # Save audio
    saved_filepath = None
    if save_format == "MP3":
        filename = f"{song_name}.mp3"
        filepath = song_path / filename
        try:
            from scipy.io.wavfile import write as write_wav
            wav_path = str(song_path / f"{song_name}_temp.wav")
            write_wav(wav_path, 48000, audio)
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path, "-b:a", mp3_bitrate, "-vn", "-map", "0:a:0", str(filepath)
            ], check=True, capture_output=True)
            # Clean up temp WAV file
            if Path(wav_path).exists():
                Path(wav_path).unlink()
            saved_filepath = str(filepath)
            info = f"Hum to song saved: {filename}"
        except Exception as e:
            info = f"Save error: {e}"
            # Clean up temp WAV file on error too
            wav_path = str(song_path / f"{song_name}_temp.wav")
            if Path(wav_path).exists():
                Path(wav_path).unlink()
    else:
        # Save audio as WAV
        filename = f"{song_name}.wav"
        filepath = song_path / filename
        try:
            if len(audio.shape) == 1:
                audio = audio.reshape(-1, 1)
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            import wave
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1 if len(pcm.shape) == 1 else pcm.shape[1])
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm.tobytes())
            saved_filepath = str(filepath)
            info = f"Hum to song saved: {filename}"
        except Exception as e:
            info = f"Save error: {e}"

    # Save metadata
    metadata = {
        "song_name": song_name,
        "timestamp": timestamp,
        "duration_s": round(duration_s, 2),
        "transcribe_time_s": round(t_transcribe_s, 1),
        "generation_time_s": round(t_gen_s, 1),
        "total_time_s": round(t_total_s, 1),
        "seed": seed,
        "melody": melody,
        "hum_adapter": hum_adapter_name,
        "hum_influence": hum_influence,
        "hum_offset": hum_offset,
        "cot": "melody",
        "cfg_scale": float(cfg_scale) if cfg_scale else None,
        "steps": int(steps),
        "variant": variant,
        "tile_size": int(tile_size),
        "vae_tile": int(vae_tile),
        "style": style.strip(),
        "lyrics": lyrics.strip(),
        "filename": filename,
        "filepath": saved_filepath,
        "hum_abc": hum_abc,
        "lora_adapters": lora_adapters or [],
        "lora_scale": lora_scale,
    }
    try:
        metadata_path = song_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        info += "\nMetadata: metadata.json"
    except Exception as e:
        info += f"\nMetadata save error: {e}"

    # Clean up temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Free MLX memory
    try:
        mx.clear_cache()
        gc.collect()
    except Exception as e:
        log(f"[hum] memory cleanup warning: {e}")

    return (48000, audio), info


def main():
    parser = argparse.ArgumentParser(description="YUE2 // MLX — Gradio UI for YuE2 on Apple Silicon")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port (auto-assigned if not specified)")
    parser.add_argument("--runs", default="./runs", help="Directory for generated works")
    args = parser.parse_args()

    os.environ.setdefault("YUE2_GROOVE_RUNS", args.runs)
    os.makedirs(args.runs, exist_ok=True)

    # Create outputs directory for generated songs
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    demo = build_ui()

    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="slate",
        ),
        css="""
            .generate-btn { font-size: 1.2em !important; padding: 12px 24px !important; }
            .llm-btn { font-size: 1em !important; }
            .status-box { font-family: monospace; font-size: 0.85em; background: #1a1a2e; color: #e0e0e0; padding: 10px; border-radius: 6px; }
            .audio-container { text-align: center; }
            /* Fix scrolling for dynamically updated textboxes */
            .gradio-container .wrap.svelte-cm5pb1 textarea,
            .gradio-container .wrap.svelte-cm5pb1 .wrap,
            .gradio-container .svelte-1137v4e textarea {
                overflow-y: auto !important;
                overscroll-behavior: contain !important;
            }
        """,
    )


# ---------------------------------------------------------------------------
# Memory optimization: set MLX env vars before any mlx import
# ---------------------------------------------------------------------------
os.environ.setdefault("MLX_PRINT_ERRORS", "0")
os.environ.setdefault("MLX_EVAL_CACHE", "1")


if __name__ == "__main__":
    main()
