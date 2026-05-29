import gradio as gr
import torchaudio as ta
import devicetorch
import torch
import time
import os
from pathlib import Path

# Output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Device detection
device = devicetorch.get(torch)
print(f"Using device: {device}")
if str(device) == "cpu":
    print("⚠️ Running on CPU — generation will be slow.")
elif "cuda" in str(device):
    print(
        f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)"
    )

# Lazy model cache
_models = {}


def _get_model(key):
    """Load a model on first use and cache it."""
    if key in _models:
        return _models[key]

    if key == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        print("⏳ Downloading & loading Chatterbox-Turbo …")
        _models[key] = ChatterboxTurboTTS.from_pretrained(device)
    elif key == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        print("⏳ Downloading & loading Chatterbox-Multilingual …")
        _models[key] = ChatterboxMultilingualTTS.from_pretrained(device)
    elif key == "original":
        from chatterbox.tts import ChatterboxTTS

        print("⏳ Downloading & loading Chatterbox-Original …")
        _models[key] = ChatterboxTTS.from_pretrained(device)
    else:
        raise ValueError(f"Unknown model key: {key}")

    print(f"✅ {key.capitalize()} model loaded!")
    return _models[key]


MODEL_CHOICES = {
    "⚡ Turbo (Fastest, English)": "turbo",
    "🌍 Multilingual (23+ Languages)": "multilingual",
    "🎯 Original (Best Quality)": "original",
}


def generate_speech(
    model_choice,
    text,
    reference_audio,
    exaggeration,
    cfg_value,
    temperature,
    min_p,
    top_p,
    repetition_penalty,
    top_k,
    norm_loudness,
    language_code,
    output_filename,
):
    if not text or not text.strip():
        yield None, "❌ Please enter some text."
        return

    model_key = MODEL_CHOICES.get(model_choice)
    if not model_key:
        yield None, "❌ Invalid model selection."
        return

    # Show loading status if model isn't cached yet
    if model_key not in _models:
        yield (
            None,
            f"⏳ Downloading & loading {model_choice} model… (first use only, may take a minute)",
        )

    try:
        model = _get_model(model_key)
    except Exception as e:
        yield None, f"❌ Failed to load model: {e}"
        return

    yield None, "🎙️ Generating speech…"

    try:
        params = {}
        if reference_audio is not None:
            params["audio_prompt_path"] = reference_audio

        if model_key == "turbo":
            params.update(
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
            )
        elif model_key == "multilingual":
            params.update(
                exaggeration=exaggeration,
                cfg_weight=max(cfg_value, 0.2),
                temperature=temperature,
            )
            if language_code and language_code != "auto":
                params["language_id"] = language_code
        else:
            params.update(
                exaggeration=exaggeration,
                cfg_weight=cfg_value,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        gen_text = text[:300] if model_key == "multilingual" else text
        wav = model.generate(gen_text, **params)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if not output_filename:
            output_filename = f"chatterbox_{model_key}_{int(time.time())}.wav"
        if not output_filename.endswith(".wav"):
            output_filename += ".wav"

        output_path = output_dir / output_filename
        if output_path.exists():
            output_path = output_dir / f"{output_path.stem}_{int(time.time())}.wav"

        ta.save(str(output_path), wav, model.sr)
        yield str(output_path), f"✅ Saved as **{output_path.name}**"

    except Exception as e:
        yield None, f"❌ Generation error: {e}"


def get_audio_info(audio_file):
    if audio_file is None:
        return ""
    try:
        waveform, sr = ta.load(audio_file)
        dur = waveform.shape[1] / sr
        return f"Duration: {dur:.1f}s · {sr} Hz · {'Mono' if waveform.shape[0] == 1 else 'Stereo'}"
    except Exception as e:
        return f"Error: {e}"


# ── Custom CSS ──────────────────────────────────────────────────────────────────
css = """
/* layout */
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* header */
.app-header { text-align: center; padding: 1.2rem 0 0.4rem; }
.app-header h1 { font-size: 2rem; margin: 0; }
.app-header p  { margin: 0.2rem 0 0; opacity: 0.7; font-size: 0.95rem; }

/* generate button */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em;
}
.generate-btn:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}

/* status box */
.status-box textarea {
    font-weight: 500;
    border-left: 3px solid #667eea !important;
}

/* section labels */
.section-title {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.55;
    margin: 0.6rem 0 0.2rem !important;
}
"""

LANGUAGES = [
    ("Auto-detect", "auto"),
    ("Arabic", "ar"),
    ("Chinese", "zh"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("English", "en"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("German", "de"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Malay", "ms"),
    ("Norwegian", "no"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Spanish", "es"),
    ("Swedish", "sv"),
    ("Swahili", "sw"),
    ("Turkish", "tr"),
]

# ── UI ──────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Chatterbox TTS", theme=gr.themes.Soft(), css=css) as app:
    # Header
    gr.HTML("""
    <div class="app-header">
        <h1>🎙️ Chatterbox TTS</h1>
        <p>AI text-to-speech with voice cloning · Models load on first use</p>
    </div>
    """)

    with gr.Tabs():
        # ── Generate Tab ────────────────────────────────────────────────────
        with gr.TabItem("Generate", id="gen"):
            # ── Row 1: Text inputs (left) + Output (right) ──────────────────
            with gr.Row(equal_height=False):
                # Left — model, text, generate
                with gr.Column(scale=5):
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_CHOICES.keys()),
                        value=list(MODEL_CHOICES.keys())[0],
                        label="Model",
                    )
                    language_selector = gr.Dropdown(
                        choices=LANGUAGES,
                        value="auto",
                        label="Language (Multilingual only)",
                        visible=False,
                    )
                    text_input = gr.Textbox(
                        label="Text",
                        placeholder="Type something… Turbo supports [laugh], [chuckle], [cough], [sigh]",
                        lines=6,
                    )
                    output_filename = gr.Textbox(
                        label="Output filename (optional)", placeholder="my_speech.wav"
                    )
                    generate_btn = gr.Button(
                        "🎵 Generate",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"],
                    )

                # Right — output audio + status + tips
                with gr.Column(scale=5):
                    output_audio = gr.Audio(label="Output", type="filepath")
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=2,
                        elem_classes=["status-box"],
                    )
                    gr.Markdown("""
                    **Quick tips**
                    - First generation is slower (model download + load)
                    - Turbo tags: `[laugh]` `[chuckle]` `[cough]` `[sigh]`
                    - Voice clone works best with 10 s+ clean speech
                    """)

            # ── Row 2: Voice clone (left) + Parameters (right) ──────────────
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.Markdown("### Voice clone (optional)")
                    reference_audio = gr.Audio(
                        label="Reference audio (10 s+)", type="filepath"
                    )
                    audio_info = gr.Textbox(
                        label="Info", interactive=False, max_lines=1
                    )

                with gr.Column(scale=5):
                    gr.Markdown("### Voice parameters")
                    with gr.Row():
                        exaggeration = gr.Slider(
                            0,
                            2,
                            0.5,
                            step=0.05,
                            label="Exaggeration",
                            info="Expressiveness (Original / Multilingual)",
                        )
                        cfg_value = gr.Slider(
                            0,
                            1,
                            0.5,
                            step=0.05,
                            label="CFG",
                            info="Pacing control (Original / Multilingual)",
                        )
                    with gr.Row():
                        temperature = gr.Slider(
                            0.05, 5, 0.8, step=0.05, label="Temperature"
                        )
                        repetition_penalty = gr.Slider(
                            1, 2, 1.2, step=0.05, label="Rep. Penalty"
                        )
                    with gr.Row():
                        min_p = gr.Slider(0, 1, 0.05, step=0.01, label="Min-P")
                        top_p = gr.Slider(0, 1, 0.95, step=0.05, label="Top-P")
                    with gr.Row():
                        top_k = gr.Slider(0, 1000, 1000, step=10, label="Top-K (Turbo)")
                        norm_loudness = gr.Checkbox(
                            True, label="Normalize loudness (Turbo)"
                        )

        # ── Guide Tab ───────────────────────────────────────────────────────
        with gr.TabItem("Guide"):
            gr.Markdown("""
            ## Models

            | Model | Params | Best for |
            |-------|--------|----------|
            | **Turbo** | 350 M | Real-time, English, paralinguistic tags |
            | **Multilingual** | 500 M | 23+ languages, zero-shot cloning |
            | **Original** | 500 M | Highest quality, fine emotion control |

            ## Paralinguistic tags (Turbo)
            `[laugh]` · `[chuckle]` · `[cough]` · `[sigh]`

            *Example:* "Hi there! [chuckle] Let me tell you something funny."

            ## Parameter guide

            | Parameter | Low | High |
            |-----------|-----|------|
            | Exaggeration | Calm, professional | Expressive, dramatic |
            | CFG | Slow, deliberate | Fast, natural |
            | Temperature | Predictable | Creative |

            ## Voice cloning tips
            - **10+ seconds** of clear speech
            - Single speaker, no background noise
            - WAV preferred, 24 kHz+
            """)

        # ── About Tab ───────────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown(f"""
            **Chatterbox** — open-source TTS family by [Resemble AI](https://www.resemble.ai/)

            - 🎭 Voice cloning from ~10 s of audio
            - 🌍 23+ language support
            - 🔒 Runs fully local · MIT license
            - 🔊 Built-in Perth neural watermarking

            Device: `{device}`

            [GitHub](https://github.com/resemble-ai/chatterbox) ·
            [Hugging Face](https://huggingface.co/ResembleAI/chatterbox-turbo)
            """)

    # ── Events ──────────────────────────────────────────────────────────────
    def toggle_language(choice):
        return gr.update(visible="Multilingual" in choice)

    model_selector.change(toggle_language, model_selector, language_selector)
    reference_audio.change(get_audio_info, reference_audio, audio_info)

    generate_btn.click(
        generate_speech,
        inputs=[
            model_selector,
            text_input,
            reference_audio,
            exaggeration,
            cfg_value,
            temperature,
            min_p,
            top_p,
            repetition_penalty,
            top_k,
            norm_loudness,
            language_selector,
            output_filename,
        ],
        outputs=[output_audio, status_output],
    )

if __name__ == "__main__":
    print(f"\n🚀 Chatterbox TTS · {device}")
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), show_error=True)
