module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      message: "uv pip install -r requirements.txt --no-build-isolation"
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
      }
    }
  }]
}
