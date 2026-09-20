module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Clone this launcher repo (if app.py doesn't exist)
    {
      when: "{{!exists('app.py')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/Rdx-ai-art/yue2-mlx.pinokio ."
      }
    },

    // 2. Create virtual environment with Python 3.12
    {
      method: "shell.run",
      params: {
        message: [
          "python3.12 -m venv env",
          "./env/bin/pip install --upgrade pip"
        ]
      }
    },

    // 3. Install Python dependencies from requirements.txt
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "./env/bin/pip install -r requirements.txt"
        ]
      }
    },

    // 4. Verification
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "./env/bin/python -c \"import mlx.core as mx; print('MLX version:', mx.__version__)\"",
          "./env/bin/python -c \"import gradio; print('Gradio version:', gradio.__version__)\"",
          "./env/bin/python -c \"import tiktoken; print('tiktoken OK')\"",
          "./env/bin/python -c \"from lyra.transcription.pipeline import transcribe; print('Lyra transcription OK')\"",
          "./env/bin/python -c \"from pathlib import Path; assert Path('yue2_model.py').exists(), 'yue2_model.py missing'; assert Path('yue2_vae.py').exists(), 'yue2_vae.py missing'; print('Inference files OK')\""
        ]
      }
    },

    // 5. Notification
    {
      method: "notify",
      params: {
        html: "<b>Installation complete!</b><br/>Click 'Download Models' in UI to get SheetSage2 + MERT for Cover feature.<br/>Click <b>Start</b> — UI opens in browser."
      }
    }
  ]
}
