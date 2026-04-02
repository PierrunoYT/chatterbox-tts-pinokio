module.exports = {
  run: [
    // Step 1: Create venv and install numpy first
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install numpy",
          "uv pip uninstall chatterbox-tts gradio resemble-perth"
        ],
      }
    },
    // Step 2: Install torch with GPU support FIRST (before other deps)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
        }
      }
    },
    // Step 3: Install remaining dependencies (torch already installed with GPU support)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt --no-build-isolation"
        ],
      }
    },
    // Step 4: Reinstall torch with GPU support in case requirements overrode it
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
        }
      }
    },
    // Step 5: Download Chatterbox models from Hugging Face
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "huggingface-cli download resemble-ai/chatterbox",
          "huggingface-cli download resemble-ai/chatterbox-multilingual",
          "huggingface-cli download ResembleAI/chatterbox-turbo"
        ]
      }
    }
  ]
}
