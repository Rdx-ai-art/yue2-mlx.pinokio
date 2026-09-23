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

    // 2. Check Python version and recreate venv if needed (need 3.12+)
    {
      method: "shell.run",
      params: {
        message: [
          "CURRENT_PYTHON_MAJOR=$(./env/bin/python --version 2>&1 | grep -oP '\\d+' | head -1)",
          "CURRENT_PYTHON_MINOR=$(./env/bin/python --version 2>&1 | grep -oP '\\d+\\.\\K\\d+' | head -1)",
          "echo \"Current Python: $CURRENT_PYTHON_MAJOR.$CURRENT_PYTHON_MINOR\"",
          "if [ \"$CURRENT_PYTHON_MAJOR\" -lt 3 ] || { [ \"$CURRENT_PYTHON_MAJOR\" -eq 3 ] && [ \"$CURRENT_PYTHON_MINOR\" -lt 12 ]; }; then",
          "  echo \"Upgrading Python venv to 3.12...\"",
          "  rm -rf env",
          "  uv venv env --python 3.12",
          "  uv pip install --python ./env/bin/python -r requirements.txt --upgrade",
          "else",
          "  echo \"Python version OK ($CURRENT_PYTHON_MAJOR.$CURRENT_PYTHON_MINOR), skipping venv recreation\"",
          "fi"
        ]
      }
    },

    // 3. Upgrade Python dependencies
    {
      method: "shell.run",
      params: {
        message: [
          "uv pip install --python ./env/bin/python -r requirements.txt --upgrade"
        ]
      }
    },

    // 4. Notification
    {
      method: "notify",
      params: {
        html: "<b>Update complete!</b><br/>Click <b>Start</b> to relaunch with latest changes."
      }
    }
  ]
}
