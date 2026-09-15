module.exports = {
  run: [
    // Update venv dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "pip install --upgrade mlx gradio tiktoken numpy soundfile"
        ]
      }
    },

    {
      method: "notify",
      params: {
        html: "Updated — Click Start to relaunch."
      }
    }
  ]
}
