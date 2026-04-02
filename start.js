module.exports = {
  daemon: true,
  run: [
    // Edit this step to customize your app's launch command
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        env: { },                   // Edit this to customize environment variables (see documentation)
        message: [
          "python app.py",    // Edit with your custom commands
        ],
        on: [{
          "event": "/(http:\\/\\/[0-9.:]+)/",
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

