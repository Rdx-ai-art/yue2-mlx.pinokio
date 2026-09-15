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
          "source env/bin/activate && pip install --upgrade pip"
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
          "pip install -r requirements.txt"
        ]
      }
    },

    // 4. Download inference .py files using curl
    {
      when: "{{!exists('yue2_model.py')}}",
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "curl -fsSL https://huggingface.co/ahmadw/YuE2-3B-MLX/resolve/main/yue2_model.py -o yue2_model.py",
          "curl -fsSL https://huggingface.co/ahmadw/YuE2-3B-MLX/resolve/main/yue2_vae.py -o yue2_vae.py"
        ]
      }
    },

    // 5. Create runs directory
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

    // 6. Verification
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "python -c \"import mlx.core as mx; print('MLX version:', mx.__version__)\"",
          "python -c \"import gradio; print('Gradio version:', gradio.__version__)\"",
          "python -c \"import tiktoken; print('tiktoken OK')\"",
          "python -c \"from pathlib import Path; assert Path('yue2_model.py').exists(), 'yue2_model.py missing'; assert Path('yue2_vae.py').exists(), 'yue2_vae.py missing'; print('Inference files OK')\""
        ]
      }
    },

    // 7. Notification
    {
      method: "notify",
      params: {
        html: "<b>Installation complete!</b><br/>Models downloaded on first use from the UI.<br/>Click <b>Start</b> — UI opens in browser."
      }
    }
  ]
}
