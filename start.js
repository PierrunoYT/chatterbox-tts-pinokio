module.exports = {
  daemon: true,
  run: [
    // Edit this step to customize your app's launch command
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",
        env: {
          GRADIO_SERVER_PORT: "{{port}}"
        },                   // Edit this to customize environment variables (see documentation)
        message: [
          "python app.py",    // Edit with your custom commands
        ],
        on: [{
          // Generic URL capture (see gepeto skill / PINOKIO.md); group 1 → input.event[1] for local.set
          "event": "/(http:\\/\\/\\S+)/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
  ]
}

