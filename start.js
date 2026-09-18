module.exports = {
  daemon: true,
  run: [
    // Auto-create venv if it doesn't exist (handles cases where Install was skipped)
    {
      when: "{{!exists('env/bin/python')}}",
      method: "shell.run",
      params: {
        message: [
          "python3 -m venv env",
          "./env/bin/pip install --upgrade pip",
          "./env/bin/pip install -r requirements.txt"
        ]
      }
    },
    // Start the app
    {
      method: "shell.run",
      params: {
        path: ".",
        env: {
          YUE2_GROOVE_VIEW: "song"
        },
        message: "./env/bin/python app.py --host 127.0.0.1 --port {{port}}",
        on: [
          {
            event: "/(http:\\/\\/(?:127\\.0\\.0\\.1|localhost):\\d+)/",
            done: true
          }
        ]
      }
    },
    {
      when: "{{Boolean(input && input.event && input.event[1])}}",
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
