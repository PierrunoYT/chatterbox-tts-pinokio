import gradio as gr
import torchaudio as ta
import devicetorch
import torch
import gc
import re
from copy import deepcopy
from uuid import uuid4
from pathlib import Path

# Output directory
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)


def cuda_arch_supported(capability, arch_list):
    """Whether torch ships kernels usable on a GPU of the given (major, minor) capability.

    Kernels (sm_XY) run on the same major capability with an equal or newer minor
    version; PTX (compute_XY) can be JIT-compiled for any newer GPU.
    """
    major, minor = capability
    for arch in arch_list:
        match = re.fullmatch(r"(sm|compute)_(\d+)[a-z]?", arch)
        if not match:
            continue
        kind, cc = match[1], int(match[2])
        if kind == "sm" and cc // 10 == major and cc % 10 <= minor:
            return True
        if kind == "compute" and cc <= major * 10 + minor:
            return True
    return False


# Device detection
device = devicetorch.get(torch)
if "cuda" in str(device):
    # torch 2.6 (pinned by Chatterbox) ships no kernels for newer GPUs such as
    # RTX 50-series; fall back to CPU instead of failing on first generation.
    major, minor = torch.cuda.get_device_capability(0)
    if not cuda_arch_supported((major, minor), torch.cuda.get_arch_list()):
        print(
            f"WARNING: {torch.cuda.get_device_name(0)} (sm_{major}{minor}) is not supported "
            f"by torch {torch.__version__}; falling back to CPU."
        )
        device = "cpu"
print(f"Using device: {device}")
if str(device) == "cpu":
    print("Running on CPU - generation will be slow.")
elif "cuda" in str(device):
    print(
        f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)"
    )

# Lazy model cache
_models = {}
_default_conditionals = {}


def _get_model(key):
    """Load a model on first use and cache it, keeping only one model in memory."""
    if key in _models:
        return _models[key]

    # Free the previously loaded model so switching models doesn't exhaust VRAM.
    if _models:
        _models.clear()
        _default_conditionals.clear()
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    if key == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        print("Downloading & loading Chatterbox-Turbo...")
        _models[key] = ChatterboxTurboTTS.from_pretrained(device)
    elif key == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        print("Downloading & loading Chatterbox-Multilingual...")
        _models[key] = ChatterboxMultilingualTTS.from_pretrained(device)
    elif key == "original":
        from chatterbox.tts import ChatterboxTTS

        print("Downloading & loading Chatterbox-Original...")
        _models[key] = ChatterboxTTS.from_pretrained(device)
    else:
        raise ValueError(f"Unknown model key: {key}")

    _default_conditionals[key] = deepcopy(_models[key].conds)
    print(f"{key.capitalize()} model loaded!")
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

    if model_key == "multilingual":
        if language_code not in {code for _, code in LANGUAGES}:
            yield None, "❌ Please select a supported language."
            return
        if len(text) > 300:
            yield None, "❌ Multilingual text is limited to 300 characters. Please split it into shorter passages."
            return

    # Show loading status if model isn't cached yet
    if model_key not in _models:
        yield (
            None,
            f"⏳ Loading {model_choice} model… (downloads on first use, may take a minute)",
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
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
            )
        elif model_key == "multilingual":
            params.update(
                exaggeration=exaggeration,
                cfg_weight=cfg_value,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                language_id=language_code,
            )
        else:
            params.update(
                exaggeration=exaggeration,
                cfg_weight=cfg_value,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # Chatterbox mutates voice conditioning when a reference is supplied.
        # Every request starts from the built-in voice, including after failures.
        model.conds = deepcopy(_default_conditionals[model_key])
        try:
            wav = model.generate(text, **params)
        finally:
            model.conds = None

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        output_path = save_audio(wav, model.sr, output_filename, model_key)
        yield str(output_path), f"✅ Saved as **{output_path.name}**"

    except Exception as e:
        yield None, f"❌ Generation error: {e}"


def save_audio(wav, sample_rate, filename, model_key):
    """Reserve a portable filename atomically so existing audio is never overwritten."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).strip(" .")
    if name.lower().endswith(".wav"):
        name = name[:-4].rstrip(" .")
    name = name[:100] or f"chatterbox_{model_key}"
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", name, re.I):
        name = "_" + name
    output_path = output_dir / f"{name}.wav"
    while True:
        try:
            audio_file = output_path.open("xb")
            break
        except FileExistsError:
            output_path = output_dir / f"{name}_{uuid4().hex}.wav"
    try:
        with audio_file:
            ta.save(audio_file, wav.detach().cpu(), sample_rate, format="wav")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def get_audio_info(audio_file):
    if audio_file is None:
        return ""
    try:
        waveform, sr = ta.load(audio_file)
        dur = waveform.shape[1] / sr
        channels = {1: "Mono", 2: "Stereo"}.get(waveform.shape[0], f"{waveform.shape[0]} channels")
        return f"Duration: {dur:.1f}s · {sr} Hz · {channels}"
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
with gr.Blocks(title="Chatterbox TTS") as app:
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
                        value="en",
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
                    - Voice clone works best with 10 s+ clean speech (Turbo requires over 5 s)
                    """)

            # ── Row 2: Voice clone (left) + Parameters (right) ──────────────
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.Markdown("### Voice clone (optional)")
                    reference_audio = gr.Audio(
                        label="Reference audio (10 s+; Turbo requires over 5 s)", type="filepath"
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
                        min_p = gr.Slider(0, 1, 0.05, step=0.01, label="Min-P (Original / Multilingual)")
                        top_p = gr.Slider(0, 1, 0.95, step=0.05, label="Top-P")
                    with gr.Row():
                        top_k = gr.Slider(0, 1000, 1000, step=10, label="Top-K (Turbo)")
                        norm_loudness = gr.Checkbox(
                            True, label="Normalize reference loudness (Turbo)"
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
        api_name="generate_speech",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    print(f"\nChatterbox TTS - {device}")
    # Binds to 127.0.0.1 by default; set GRADIO_SERVER_NAME (see start.js) to change it.
    app.launch(theme=gr.themes.Soft(), css=css, show_error=True)
