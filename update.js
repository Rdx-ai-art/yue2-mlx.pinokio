module.exports = {
  run: [
    // Update venv dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "pip install -r requirements.txt --upgrade"
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
