module.exports = {
  run: [
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        message: [
          "uv pip install numpy",
          "uv pip uninstall chatterbox-tts gradio",
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Install torch after other dependencies
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
    // Download Chatterbox models from Hugging Face
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "huggingface-cli download resemble-ai/chatterbox --local-dir models/chatterbox",
          "huggingface-cli download resemble-ai/chatterbox-multilingual --local-dir models/chatterbox-multilingual",
          "huggingface-cli download resemble-ai/chatterbox-turbo --local-dir models/chatterbox-turbo"
        ]
      }
    }
  ]
}
