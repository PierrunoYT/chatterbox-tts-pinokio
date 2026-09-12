module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{exists('installed.flag')}}",
      method: "fs.rm",
      params: { path: "installed.flag" }
    },
    // Step 1: Install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        path: "app",
        message: [
          "python -c \"import platform, sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), 'Use Python 3.10-3.13; reset to recreate the environment'; assert not (sys.platform == 'darwin' and platform.machine() == 'x86_64'), 'Chatterbox requires Apple Silicon on macOS'\"",
          "uv pip install \"numpy<2; python_version < '3.13'\" \"numpy>=2; python_version >= '3.13'\" setuptools wheel \"uv_build~=0.12.7\"",
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
          path: "app",
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip check"
      }
    },
    // Only report installed after dependency validation succeeds.
    {
      method: "fs.write",
      params: {
        path: "installed.flag",
        text: "ok"
      }
    }
  ]
}
