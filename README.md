# 🎙️ Chatterbox TTS App

AI-Powered Text-to-Speech with Voice Cloning using Chatterbox TTS and Gradio interface.

## ⚡ Model Zoo

Chatterbox is a family of three state-of-the-art, open-source text-to-speech models by Resemble AI:

| Model | Size | Languages | Key Features | Best For |
|-------|------|-----------|--------------|----------|
| **Chatterbox-Turbo** | 350M | English | Paralinguistic Tags ([laugh]), Lower Compute and VRAM | Zero-shot voice agents, Production |
| **Chatterbox-Multilingual** | 500M | 23+ | Zero-shot cloning, Multiple Languages | Global applications, Localization |
| **Chatterbox** | 500M | English | CFG & Exaggeration tuning | General zero-shot TTS with creative controls |

## ✨ Features

- 🎭 **Voice Cloning**: Clone any voice with just 10 seconds of audio
- ⚡ **Turbo Mode**: Ultra-fast generation with lower VRAM requirements
- 🎭 **Paralinguistic Tags**: Add [laugh], [cough], [chuckle] for realism
- 🌍 **23+ Languages**: Multilingual support (Arabic, Chinese, French, Spanish, etc.)
- 🎨 **Emotion Control**: Adjust expressiveness and pacing
- 🆓 **Free & Open Source**: MIT license, completely free to use
- 🔒 **Privacy**: Runs completely locally on your machine
- 🌐 **Cross-Platform**: Works on Windows, Mac, and Linux
- 🖥️ **Web Interface**: Easy-to-use Gradio interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU

### Installation

1. Clone this repository:
```bash
git clone https://github.com/PierrunoYT/chatterbox-tts-pinokio.git
cd chatterbox-tts-pinokio
```

2. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```
> **Note:** This installs a default `torch` build only. For GPU acceleration (NVIDIA/AMD), install the matching `torch` build for your platform afterward — see the `when` blocks in `torch.js` at the repo root for the exact commands per platform. The Pinokio install flow (`install.js`) does this automatically.

3. (Optional) For Chatterbox-Turbo model access, login to Hugging Face:
```bash
huggingface-cli login
```
Or set `HF_TOKEN` in the `env` block of `start.js` (there's a commented-out example line to uncomment).

4. Run the application:
```bash
cd app
python app.py
```

5. Open your browser and go to `http://127.0.0.1:7860`

## 🎯 Usage

### Basic Text-to-Speech
1. Select your preferred model (Turbo, Multilingual, or Original)
2. Enter your text in the input field
3. Adjust emotion and CFG settings as desired
4. Click "Generate Speech"
5. Download your generated audio

### Turbo Model with Paralinguistic Tags
1. Select "Chatterbox-Turbo" model
2. Use tags in your text for added realism:
   - `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`
   - Example: "Hi there! [chuckle] Let me tell you something funny."
3. Generate ultra-fast, realistic speech

### Voice Cloning
1. Upload a reference audio file (10+ seconds recommended)
2. Enter your text
3. Adjust settings
4. Generate speech with the cloned voice

### Multilingual Support
1. Select "Chatterbox-Multilingual" model
2. Enter text in any supported language (auto-detected)
3. Optionally specify language code for better accuracy

## 🌍 Supported Languages (Multilingual Model)

Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

## 🎨 Settings

- **Model Selection**: Choose between Turbo (fastest), Multilingual (23+ languages), or Original (best quality)
- **Emotion Exaggeration**: Controls how expressive the speech is (0.0 = calm, 1.0 = very expressive)
- **CFG Scale**: Controls speech pacing (0.0 = slower/deliberate, 1.0 = faster/natural)
- **Paralinguistic Tags** (Turbo only): `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]` for added realism

## 📁 Project Structure

```
chatterbox-tts-pinokio/
├── app/                 # Application code (Pinokio convention)
│   ├── app.py           # Main Gradio application
│   ├── requirements.txt # Python dependencies
│   ├── pyproject.toml   # UV / build hints
│   └── outputs/         # Generated audio (created at runtime)
├── install.js, start.js, …  # Pinokio launcher scripts (repo root)
└── icon.png             # Application icon (optional)
```

## 🔧 Technical Details

- **Model**: Chatterbox TTS by Resemble AI
- **Interface**: Gradio web interface
- **Audio Format**: WAV files
- **Device Support**: CUDA GPU / CPU automatic detection

## 📝 Tips for Best Results

### Voice Cloning
- Use at least 10 seconds of clear reference audio
- Ensure single speaker with no background noise
- WAV format preferred, 24kHz+ sample rate
- Professional microphone recommended

### Text Input
- Use natural punctuation for better prosody
- Longer texts generally produce better results
- Avoid special characters or formatting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Credits

- **Chatterbox TTS**: [Resemble AI](https://www.resemble.ai/)
- **Interface**: [Gradio](https://gradio.app/)
- **Integration**: Pinokio Community

## 🐛 Issues

If you encounter any issues, please report them on the [GitHub Issues](https://github.com/PierrunoYT/chatterbox-tts-pinokio/issues) page.

## 🔌 API Reference

The Gradio app exposes a programmatic API once running. Replace `7860` with the actual port shown in the Pinokio terminal.

### Python (gradio_client)

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")

result = client.predict(
    model_choice="⚡ Turbo (Fastest, English)",
    text="Hello, this is a test.",
    reference_audio=None,
    exaggeration=0.5,
    cfg_value=0.5,
    temperature=0.8,
    min_p=0.05,
    top_p=0.95,
    repetition_penalty=1.2,
    top_k=1000,
    norm_loudness=True,
    language_code="auto",
    output_filename="output.wav",
    api_name="/generate_speech",
)
# result is (audio_filepath, status_message)
print(result)
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://127.0.0.1:7860/call/generate_speech", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    data: [
      "⚡ Turbo (Fastest, English)", // model_choice
      "Hello, this is a test.",      // text
      null,                          // reference_audio
      0.5,                           // exaggeration
      0.5,                           // cfg_value
      0.8,                           // temperature
      0.05,                          // min_p
      0.95,                          // top_p
      1.2,                           // repetition_penalty
      1000,                          // top_k
      true,                          // norm_loudness
      "auto",                        // language_code
      "output.wav"                   // output_filename
    ]
  })
});
const { event_id } = await response.json();

// Poll for result
const resultResponse = await fetch(`http://127.0.0.1:7860/call/generate_speech/${event_id}`);
const text = await resultResponse.text();
console.log(text);
```

### curl

```bash
# Submit the request
EVENT_ID=$(curl -s -X POST http://127.0.0.1:7860/call/generate_speech \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "⚡ Turbo (Fastest, English)",
      "Hello, this is a test.",
      null, 0.5, 0.5, 0.8, 0.05, 0.95, 1.2, 1000, true, "auto", "output.wav"
    ]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['event_id'])")

# Retrieve the result
curl -s http://127.0.0.1:7860/call/generate_speech/$EVENT_ID
```
