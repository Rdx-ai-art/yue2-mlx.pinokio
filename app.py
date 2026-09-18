#!/usr/bin/env python3
"""YUE2 // GROOVE-MLX — Gradio web UI for YuE2 on Apple Silicon (MLX).

A streamlined, Mac-optimized Gradio interface that wraps the native MLX
inference from https://huggingface.co/ahmadw/YuE2-3B-MLX.

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


def _load_pipeline(model_dir: str, variant: str, progress=None) -> Yue2PipelineMLX:
    """Load (or reuse) the MLX pipeline. Reuses existing one when settings unchanged."""
    global _PIPE, _PIPE_KEY

    mv = _VARIANT_MAP.get(variant, ModelVariant.EIGHT_BIT)
    key = (model_dir, variant)

    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    if _PIPE is not None:
        del _PIPE
        _PIPE = None
        _PIPE_KEY = None

    if progress is not None:
        progress(0.0, desc="Loading MLX model…")
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
    style, lyrics, cot, seed, cfg_scale, steps, variant, model_dir,
    abc_text, tile_size, vae_tile, save_format, mp3_bitrate,
    progress=gr.Progress(),
):
    """Generate a song using the MLX pipeline."""
    pipe = _load_pipeline(model_dir, variant, progress)

    if not style.strip():
        raise gr.Error("Style is required (e.g. 'indie pop, warm vocal')")
    if not lyrics.strip():
        raise gr.Error("Lyrics are required")

    _CLEARED.clear()
    t0 = time.perf_counter()

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


def _detect_models(api_url, api_key):
    """Detect available models from an OpenAI-compatible API.
    Returns the first model name as a string, or an error message."""
    import urllib.request
    import ssl

    if not api_url.strip():
        return "Please enter an API URL first"

    # Strip /chat/completions suffix if present
    base_url = api_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url.rsplit("/", 1)[0]

    models_url = f"{base_url}/v1/models"
    headers = {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            models = result.get("data", [])
            if models:
                first_model = models[0].get("id", "")
                if first_model:
                    return first_model
                return "No model name found (check API response)"
            return "No models available — load a model in your LLM server"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: Check API URL (expected OpenAI-compatible endpoint)"
    except Exception as e:
        return f"Connection failed: {type(e).__name__}"


_CLEARED = __import__("threading").Event()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

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
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Song Parameters")
                        style_input = gr.Textbox(
                            label="Style",
                            placeholder="indie pop, bright acoustic guitar, warm vocal",
                            lines=2,
                        )
                        lyrics_input = gr.Textbox(
                            label="Lyrics",
                            placeholder="[Verse]\nSoft morning light...",
                            lines=5,
                        )
                        with gr.Row():
                            cot_input = gr.Radio(
                                choices=["full", "melody", "off"],
                                value="off",
                                label="ABC Plan",
                            )
                            seed_input = gr.Number(
                                value=831001,
                                label="Seed",
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
                        model_dir_input = gr.Textbox(
                            label="Model Dir",
                            value="./models/YuE2-3B-MLX",
                            lines=1,
                        )
                        variant_input = gr.Radio(
                            choices=["8-bit (4.2GB)", "BF16 (7GB)", "4-bit (3.4GB)"],
                            value="8-bit (4.2GB)",
                            label="Variant",
                        )
                        save_format_input = gr.Radio(
                            choices=["WAV", "MP3"],
                            value="WAV",
                            label="Format",
                        )
                        mp3_bitrate_input = gr.Radio(
                            choices=["128k", "192k", "256k", "320k"],
                            value="192k",
                            label="MP3 kbps",
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
                            cfg_scale, steps_input, variant_input, model_dir_input,
                            abc_score_input, tile_size_input, vae_tile_input, save_format_input,
                            mp3_bitrate_input],
                    outputs=[audio_output, abc_output, status_output],
                )

            # ── TAB 2: LLM Writing Room ──────────────────────────────
            with gr.Tab("02 // WRITING ROOM"):
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
                        with gr.Row():
                            btn_auto_detect = gr.Button("🔍 Auto-detect Model", variant="secondary", size="sm")
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
                btn_auto_detect.click(
                    fn=_detect_models,
                    inputs=[llm_api_url, llm_api_key],
                    outputs=[llm_model],
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

            # ── TAB 3: INFO ────────────────────────────────────────
            with gr.Tab("03 // INFO"):
                gr.Markdown("### Generation Parameters")
                gr.Markdown(
                    "| Parameter | Description | Recommended | Default(lowest ram) |\n"
                    "|-----------|-------------|-------------|---------------------|\n"
                    "| **CFG Scale** | Classifier-free guidance. Higher = follows prompt more strictly | 1.0 | 1.0 |\n"
                    "| **NAR Steps** | Midpoint ODE steps. More = better quality but slower | 12 | 8 |\n"
                    "| **NAR Tile** | Process frames in tiles. Smaller = less RAM, lower quality | 4096 | 2048 |\n"
                    "| **VAE Tile** | VAE decode tile size. Lower = less RAM during decode | 512 | 64 |\n"
                    "| **ABC (Symbolic) Plan** | ABC score generation. FULL, MELODY only,`off` skips : saves ~30% time | on | off |\n"
                    "| **Seed** | Set for reproducible results | any | any |\n"
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

                gr.Markdown("### Tips")
                gr.Markdown(
                    "- **8-bit variant** gives near-bF16 quality at ~2x decode speed\n"
                    "- Use the **Writing Room** to brainstorm ideas before generating\n"
                    "- Set a **seed** for reproducible results\n"
                    "- **cot=full** generates chord-annotated ABC scores for editing\n"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
import os
os.environ.setdefault("MLX_PRINT_ERRORS", "0")
os.environ.setdefault("MLX_EVAL_CACHE", "1")


if __name__ == "__main__":
    main()
