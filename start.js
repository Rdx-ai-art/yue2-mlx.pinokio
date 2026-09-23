module.exports = {
  daemon: true,
  run: [
    // Auto-create venv if it doesn't exist (handles cases where Install was skipped)
    {
      when: "{{!exists('env/bin/python')}}",
      method: "shell.run",
      params: {
        message: [
          "uv venv env --python 3.12",
          "uv pip install --python ./env/bin/python -r requirements.txt"
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
