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
git clone https://github.com/PierrunoYT/chatterbox-tts-app.git
cd chatterbox-tts-app
```

2. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```

3. (Optional) For Chatterbox-Turbo model access, login to Hugging Face:
```bash
huggingface-cli login
```
Or set the HF_TOKEN environment variable in `start.js`.

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

## � Supported Languages (Multilingual Model)

Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

## 🎨 Settings

- **Model Selection**: Choose between Turbo (fastest), Multilingual (23+ languages), or Original (best quality)
- **Emotion Exaggeration**: Controls how expressive the speech is (0.0 = calm, 1.0 = very expressive)
- **CFG Scale**: Controls speech pacing (0.0 = slower/deliberate, 1.0 = faster/natural)
- **Paralinguistic Tags** (Turbo only): `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]` for added realism

## 📁 Project Structure

```
chatterbox-tts-app/
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

If you encounter any issues, please report them on the [GitHub Issues](https://github.com/PierrunoYT/chatterbox-tts-app/issues) page.
