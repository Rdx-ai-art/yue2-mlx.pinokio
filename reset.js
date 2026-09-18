module.exports = {
  run: [
    {
      method: "shell.run",
      params: { message: "rm -rf env" }
    },
    {
      method: "shell.run",
      params: { message: "rm -rf runs" }
    },
    {
      method: "shell.run",
      params: { message: "rm -f yue2_model.py yue2_vae.py app.py" }
    },
    {
      method: "notify",
      params: {
        html: "Reset complete. Click 'Install' to reinstall."
      }
    }
  ]
}
