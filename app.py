import gradio as gr
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import devicetorch
import torch
import os
import tempfile
import time
from pathlib import Path

# Initialize output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Get the appropriate device
device = devicetorch.get(torch)
print(f"Using device: {device}")
if str(device) == "cpu":
    print("⚠️ Running on CPU — generation will be slow. For GPU support, reinstall with CUDA-enabled PyTorch.")
elif "cuda" in str(device):
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load all models at startup (no token needed - using public repos)
print("Loading Chatterbox TTS models...")
models = {}

try:
    print("Loading Chatterbox-Turbo...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    models['turbo'] = ChatterboxTurboTTS.from_pretrained(device)
    print("✅ Turbo model loaded!")
except Exception as e:
    print(f"⚠️ Turbo model failed: {e}")
    models['turbo'] = None

try:
    print("Loading Chatterbox-Multilingual...")
    models['multilingual'] = ChatterboxMultilingualTTS.from_pretrained(device)
    print("✅ Multilingual model loaded!")
except Exception as e:
    print(f"⚠️ Multilingual model failed: {e}")
    models['multilingual'] = None

try:
    print("Loading Chatterbox (Original)...")
    models['original'] = ChatterboxTTS.from_pretrained(device)
    print("✅ Original model loaded!")
except Exception as e:
    print(f"⚠️ Original model failed: {e}")
    models['original'] = None

if not any(models.values()):
    print("❌ No models loaded successfully!")
else:
    print("✅ Models ready!")

def generate_speech(model_choice, text, reference_audio, exaggeration, cfg_value, temperature, min_p, top_p, repetition_penalty, top_k, norm_loudness, language_code, output_filename):
    """Generate speech using Chatterbox TTS"""
    if not any(models.values()):
        return None, "❌ No models loaded. Please check the installation."
    
    # Map model choice to model key
    model_map = {
        "Chatterbox-Turbo (Fastest, English only)": 'turbo',
        "Chatterbox-Multilingual (23+ Languages)": 'multilingual',
        "Chatterbox-Original (Best Quality)": 'original'
    }
    
    model_key = model_map.get(model_choice)
    model = models.get(model_key)
    
    if model is None:
        return None, f"❌ {model_choice} not loaded. Try another model."
    
    if not text or not text.strip():
        return None, "❌ Please enter some text to convert to speech."
    
    try:
        print(f"Generating speech with {model_choice} for: {text[:50]}...")
        
        # Prepare parameters
        params = {}
        
        # Add reference audio if provided
        if reference_audio is not None:
            params["audio_prompt_path"] = reference_audio
            print(f"Using voice cloning with reference audio: {reference_audio}")
        
        if model_key == 'turbo':
            # Turbo-specific parameters (no exaggeration/cfg_weight)
            params["temperature"] = temperature
            params["min_p"] = min_p
            params["top_p"] = top_p
            params["top_k"] = int(top_k)
            params["repetition_penalty"] = repetition_penalty
            params["norm_loudness"] = norm_loudness
        elif model_key == 'multilingual':
            params["exaggeration"] = exaggeration
            params["cfg_weight"] = max(cfg_value, 0.2)  # Multilingual minimum is 0.2
            params["temperature"] = temperature
            if language_code and language_code != "auto":
                params["language_id"] = language_code
                print(f"Using language: {language_code}")
        else:
            # Original model
            params["exaggeration"] = exaggeration
            params["cfg_weight"] = cfg_value
            params["temperature"] = temperature
            params["min_p"] = min_p
            params["top_p"] = top_p
            params["repetition_penalty"] = repetition_penalty
        
        # Generate speech (multilingual has 300 char limit)
        gen_text = text[:300] if model_key == 'multilingual' else text
        wav = model.generate(gen_text, **params)
        
        # Create output filename
        if not output_filename:
            timestamp = int(time.time())
            output_filename = f"chatterbox_{model_key}_{timestamp}.wav"
        
        if not output_filename.endswith('.wav'):
            output_filename += '.wav'
        
        # Ensure wav is 2D (channels, samples) as required by torchaudio.save
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Save to outputs directory, avoid overwriting existing files
        output_path = output_dir / output_filename
        if output_path.exists():
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_dir / f"{stem}_{int(time.time())}{suffix}"
            output_filename = output_path.name

        ta.save(str(output_path), wav, model.sr)

        print(f"✅ Speech generated successfully: {output_path}")
        return str(output_path), f"✅ Speech generated successfully with {model_choice}!\nSaved as: {output_filename}"
        
    except Exception as e:
        error_msg = f"❌ Error generating speech: {str(e)}"
        print(error_msg)
        return None, error_msg

def get_audio_info(audio_file):
    """Get information about uploaded audio file"""
    if audio_file is None:
        return "No audio file uploaded"
    
    try:
        # Load audio to get info
        waveform, sample_rate = ta.load(audio_file)
        duration = waveform.shape[1] / sample_rate
        channels = waveform.shape[0]
        
        return f"📊 Audio Info:\n• Duration: {duration:.2f} seconds\n• Sample Rate: {sample_rate} Hz\n• Channels: {channels}\n• Recommended: 10+ seconds, 24kHz+, mono"
    except Exception as e:
        return f"❌ Error reading audio: {str(e)}"

# Create the Gradio interface
with gr.Blocks(
    title="Chatterbox TTS",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
    }
    """
) as app:
    
    with gr.Row():
        gr.Markdown(
            """
            # 🎙️ Chatterbox TTS
            ### AI-Powered Text-to-Speech with Voice Cloning
            
            Generate natural-sounding speech from text with optional voice cloning capabilities.
            """,
            elem_classes=["main-header"]
        )
    
    with gr.Tabs():
        # Main TTS Tab
        with gr.TabItem("🎤 Text-to-Speech"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚡ Model Selection")
                    model_selector = gr.Dropdown(
                        choices=[
                            "Chatterbox-Turbo (Fastest, English only)",
                            "Chatterbox-Multilingual (23+ Languages)",
                            "Chatterbox-Original (Best Quality)"
                        ],
                        value="Chatterbox-Turbo (Fastest, English only)",
                        label="Choose Model",
                        info="Turbo = fastest, Multilingual = 23+ languages, Original = best quality"
                    )
                    
                    gr.Markdown("### 📝 Input Text")
                    text_input = gr.Textbox(
                        label="Text to Convert",
                        placeholder="Enter text... For Turbo: try adding [laugh], [chuckle], [cough], [sigh] for realism!",
                        lines=5,
                        max_lines=10
                    )
                    
                    gr.Markdown(
                        """
                        🎭 **Turbo Model Tags**: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`
                        
                        Example: *"Hi there! [chuckle] Let me tell you something funny."*
                        """
                    )
                    gr.Markdown("### 🌍 Language (Multilingual Model Only)")
                    language_selector = gr.Dropdown(
                        choices=[
                            ("Auto-detect", "auto"),
                            ("Arabic (العربية)", "ar"),
                            ("Chinese (中文)", "zh"),
                            ("Danish (Dansk)", "da"),
                            ("Dutch (Nederlands)", "nl"),
                            ("English", "en"),
                            ("Finnish (Suomi)", "fi"),
                            ("French (Français)", "fr"),
                            ("German (Deutsch)", "de"),
                            ("Greek (Ελληνικά)", "el"),
                            ("Hebrew (עברית)", "he"),
                            ("Hindi (हिन्दी)", "hi"),
                            ("Italian (Italiano)", "it"),
                            ("Japanese (日本語)", "ja"),
                            ("Korean (한국어)", "ko"),
                            ("Malay (Bahasa Melayu)", "ms"),
                            ("Norwegian (Norsk)", "no"),
                            ("Polish (Polski)", "pl"),
                            ("Portuguese (Português)", "pt"),
                            ("Russian (Русский)", "ru"),
                            ("Spanish (Español)", "es"),
                            ("Swedish (Svenska)", "sv"),
                            ("Swahili (Kiswahili)", "sw"),
                            ("Turkish (Türkçe)", "tr"),
                        ],
                        value="auto",
                        label="Language Code",
                        info="Select language for multilingual model (auto-detect if not specified)"
                    )
                    
                    gr.Markdown("### 🎨 Voice Settings")
                    gr.Markdown("💡 *Exaggeration & CFG: Original/Multilingual only. Top-K & Loudness Norm: Turbo only.*")
                    with gr.Row():
                        exaggeration = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.5,
                            step=0.05,
                            label="Emotion Exaggeration",
                            info="Higher values make speech more expressive (Original/Multilingual)"
                        )
                        cfg_value = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="CFG Scale",
                            info="Lower values = slower, more deliberate speech (Original/Multilingual, min 0.2 for Multilingual)"
                        )
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.05,
                            maximum=5.0,
                            value=0.8,
                            step=0.05,
                            label="Temperature",
                            info="Controls randomness of generation"
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.2,
                            step=0.05,
                            label="Repetition Penalty",
                            info="Penalizes repeated tokens"
                        )
                    with gr.Row():
                        min_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.05,
                            step=0.01,
                            label="Min-P",
                            info="Minimum probability threshold"
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            label="Top-P",
                            info="Nucleus sampling threshold"
                        )
                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=1000,
                            step=10,
                            label="Top-K (Turbo only)",
                            info="Top-K sampling limit"
                        )
                        norm_loudness = gr.Checkbox(
                            value=True,
                            label="Normalize Loudness (Turbo only)",
                            info="Normalize output to -27 LUFS"
                        )
                    
                    output_filename = gr.Textbox(
                        label="Output Filename (optional)",
                        placeholder="my_speech.wav",
                        info="Leave empty for auto-generated name"
                    )
                    
                    generate_btn = gr.Button(
                        "🎵 Generate Speech",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Voice Cloning (Optional)")
                    reference_audio = gr.Audio(
                        label="Reference Audio for Voice Cloning (Upload 10+ seconds of clear speech)",
                        type="filepath"
                    )
                    
                    audio_info = gr.Textbox(
                        label="Audio Information",
                        interactive=False,
                        max_lines=5
                    )
                    
                    gr.Markdown("### 🔊 Generated Output")
                    output_audio = gr.Audio(
                        label="Generated Speech",
                        type="filepath"
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=3
                    )
        
        # Examples Tab
        with gr.TabItem("📚 Examples & Tips"):
            gr.Markdown("""
            ## ⚡ Chatterbox-Turbo Paralinguistic Tags
            
            Add natural vocal expressions to your speech:
            
            | Tag | Effect | Example Usage |
            |-----|--------|---------------|
            | `[laugh]` | Full laughter | "That's hilarious! [laugh]" |
            | `[chuckle]` | Light laugh | "Well, [chuckle] that's interesting." |
            | `[cough]` | Cough sound | "[cough] Excuse me, as I was saying..." |
            | `[sigh]` | Sighing | "[sigh] It's been a long day." |
            
            ### Turbo Example Texts:
            - "Hi there, Sarah here from MochaFone [chuckle], do you have a minute?"
            - "Let me tell you something funny [laugh] you won't believe this!"
            - "[sigh] I've been working on this all day, but it's finally done."
            
            ## 🎯 Voice Cloning Tips
            
            ### For Best Results:
            - **Duration**: Use at least 10 seconds of reference audio
            - **Quality**: Clear speech, no background noise
            - **Format**: WAV format preferred, 24kHz+ sample rate
            - **Content**: Single speaker, natural speaking style
            - **Microphone**: Professional microphone recommended
            
            ### General Example Texts:
            - "Hello, this is a test of the Chatterbox text-to-speech system."
            - "The quick brown fox jumps over the lazy dog."
            - "Welcome to our AI-powered voice synthesis demonstration."
            
            ### Multilingual Examples:
            - **French**: "Bonjour, comment ça va? Ceci est un test."
            - **Spanish**: "Hola, ¿cómo estás? Esta es una prueba."
            - **Chinese**: "你好，今天天气真不错。"
            - **Japanese**: "こんにちは、お元気ですか。"
            
            ### Emotion Control (Original & Multilingual):
            - **Low exaggeration (0.0-0.3)**: Calm, professional tone
            - **Medium exaggeration (0.4-0.6)**: Natural, conversational
            - **High exaggeration (0.7-1.0)**: Expressive, dramatic
            
            ### CFG Scale (Original & Multilingual):
            - **Low CFG (0.0-0.3)**: Slower, more deliberate speech
            - **Medium CFG (0.4-0.6)**: Balanced pacing
            - **High CFG (0.7-1.0)**: Faster, more natural rhythm
            
            ### Model Selection Guide:
            - **Turbo**: Best for real-time voice agents, fastest generation, lowest VRAM
            - **Multilingual**: Use when you need non-English languages
            - **Original**: Best overall quality with fine-tuned emotion control
            """)
        
        # About Tab
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(f"""
            ## About Chatterbox TTS
            
            **Chatterbox** is a family of three state-of-the-art, open-source text-to-speech models by Resemble AI.
            
            ### Model Family:
            
            | Model | Parameters | Features |
            |-------|------------|----------|
            | Chatterbox-Turbo | 350M | Paralinguistic tags, ultra-fast, low VRAM |
            | Chatterbox-Multilingual | 500M | 23+ languages, zero-shot cloning |
            | Chatterbox (Original) | 500M | CFG & exaggeration tuning, best quality |
            
            ### Features:
            - 🎭 **Voice Cloning**: Clone any voice with just 10 seconds of audio
            - ⚡ **Turbo Mode**: Ultra-fast generation with paralinguistic tags
            - 🌍 **23+ Languages**: Global language support
            - 🎨 **Emotion Control**: Adjust expressiveness and pacing
            - 🆓 **Free & Open Source**: MIT license, completely free to use
            - 🔒 **Privacy**: Runs completely locally on your machine
            - 🌐 **Cross-Platform**: Works on Windows, Mac, and Linux
            
            ### Technical Details:
            - **Device**: {device}
            - **Models**: Chatterbox TTS family (Resemble AI)
            - **Output Format**: WAV audio files
            - **Sample Rate**: Variable (model default)
            
            ### Built-in Watermarking:
            All audio includes Resemble AI's Perth watermarker - imperceptible neural watermarks
            for responsible AI use.
            
            ### Credits:
            - **Chatterbox TTS**: Resemble AI
            - **Integration**: Pinokio Community
            - **Interface**: Gradio
            
            For more information, visit:
            - [Resemble AI](https://www.resemble.ai/)
            - [Chatterbox-Turbo on Hugging Face](https://huggingface.co/ResembleAI/chatterbox-turbo)
            """)
    
    # Event handlers
    reference_audio.change(
        fn=get_audio_info,
        inputs=[reference_audio],
        outputs=[audio_info]
    )
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[model_selector, text_input, reference_audio, exaggeration, cfg_value, temperature, min_p, top_p, repetition_penalty, top_k, norm_loudness, language_selector, output_filename],
        outputs=[output_audio, status_output]
    )

# Launch the application
if __name__ == "__main__":
    print(f"\n🚀 Starting Chatterbox TTS on device: {device}")
    print("🌐 The application will be available in your web browser")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    ) 