module.exports = {
  run: [
    // Step 1: Install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install numpy wheel",
          "uv pip install -r requirements.txt --no-build-isolation"
        ],
      }
    },
    // Step 2: Reinstall torch with GPU support in case requirements overrode it
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
        }
      }
    }
  ]
}
