module.exports = {
  run: [
    // 1. Pull latest code from git
    {
      method: "shell.run",
      params: {
        message: [
          "git pull origin main"
        ]
      }
    },

    // 2. Upgrade Python dependencies
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

    // 3. Notification
    {
      method: "notify",
      params: {
        html: "<b>Update complete!</b><br/>Click <b>Start</b> to relaunch with latest changes."
      }
    }
  ]
}
