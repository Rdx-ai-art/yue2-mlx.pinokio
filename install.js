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

    // 2. Create virtual environment
    {
      method: "shell.run",
      params: {
        message: [
          "python3 -m venv env",
          "python3 -m pip install --upgrade pip"
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
          "python3 -m pip install -r requirements.txt"
        ]
      }
    },

    // 4. Create runs directory
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "mkdir -p runs"
        ]
      }
    },

    // 5. Verification
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "python3 -c \"import mlx.core as mx; print('MLX version:', mx.__version__)\"",
          "python3 -c \"import gradio; print('Gradio version:', gradio.__version__)\"",
          "python3 -c \"import tiktoken; print('tiktoken OK')\"",
          "python3 -c \"from pathlib import Path; assert Path('yue2_model.py').exists(), 'yue2_model.py missing'; assert Path('yue2_vae.py').exists(), 'yue2_vae.py missing'; print('Inference files OK')\""
        ]
      }
    },

    // 6. Notification
    {
      method: "notify",
      params: {
        html: "<b>Installation complete!</b><br/>Click <b>Start</b> — UI opens in browser."
      }
    }
  ]
}
