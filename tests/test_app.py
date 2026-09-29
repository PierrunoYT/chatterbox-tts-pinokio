"""UI and request regression tests; inference is stubbed (no model downloads)."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio  # Exercise the real pinned UI library before stubbing ML modules.

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tts_app", ROOT / "app/app.py")
tts = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    "torch": MagicMock(), "torchaudio": MagicMock(),
    "devicetorch": MagicMock(get=lambda _: "cpu"),
}):
    spec.loader.exec_module(tts)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = patch.object(tts, "output_dir", Path(self.temp.name))
        self.output.start()
        self.addCleanup(self.output.stop)
        self.model = MagicMock(sr=24000, conds={"voice": "default"})
        self.model.generate.return_value.dim.return_value = 2
        tts._models.clear()
        tts._default_conditionals.clear()
        for key in ("turbo", "multilingual", "original"):
            tts._models[key] = self.model
            tts._default_conditionals[key] = {"voice": "default"}
        tts.ta.save.reset_mock(side_effect=True)

    def generate(self, key="turbo", **overrides):
        args = dict(model_choice=next(k for k, v in tts.MODEL_CHOICES.items() if v == key),
                    text="Hello", reference_audio=None, exaggeration=.5, cfg_value=.1,
                    temperature=.8, min_p=.05, top_p=.95, repetition_penalty=1.2,
                    top_k=1000, norm_loudness=True, language_code="en", output_filename="test.wav")
        args.update(overrides)
        return list(tts.generate_speech(**args))[-1]

    def test_ui_constructs_and_api_is_serialized(self):
        fn = next(fn for fn in tts.app.fns.values() if fn.api_name == "generate_speech")
        self.assertEqual(fn.concurrency_limit, 1)
        self.assertEqual(tts.language_selector.value, "en")

    def test_multilingual_passes_language_and_settings(self):
        path, status = self.generate("multilingual", language_code="fr")
        self.assertIsNotNone(path, status)
        kwargs = self.model.generate.call_args.kwargs
        self.assertEqual(kwargs["language_id"], "fr")
        self.assertEqual(kwargs["cfg_weight"], .1)
        self.assertEqual(kwargs["min_p"], .05)
        self.assertEqual(kwargs["top_p"], .95)

    def test_turbo_omits_unsupported_min_p(self):
        path, status = self.generate("turbo")
        self.assertIsNotNone(path, status)
        self.assertNotIn("min_p", self.model.generate.call_args.kwargs)

    def test_invalid_requests_do_not_generate(self):
        for overrides in ({"text": " "}, {"text": "x" * 301}, {"language_code": "auto"}):
            path, _ = self.generate("multilingual", **overrides)
            self.assertIsNone(path)
        self.model.generate.assert_not_called()

    def test_cloned_voice_does_not_leak_to_next_request(self):
        seen = []
        def generate(text, **kwargs):
            seen.append(self.model.conds["voice"])
            self.model.conds["voice"] = "clone"
            if kwargs.get("audio_prompt_path"):
                raise RuntimeError("inference failed after conditioning")
            return MagicMock(dim=lambda: 2)
        self.model.generate.side_effect = generate
        self.generate(reference_audio="voice.wav")
        self.generate()
        self.assertEqual(seen, ["default", "default"])
        self.assertIsNone(self.model.conds)

    def test_existing_files_are_preserved(self):
        paths = [self.generate()[0] for _ in range(3)]
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(Path(path).exists() for path in paths))

    def test_portable_filenames(self):
        for name in ("../../test.wav", r"..\..\test.wav", "CON.wav", "a\x00.wav", "...", "X.WAV"):
            path, status = self.generate(output_filename=name)
            self.assertIsNotNone(path, status)
            self.assertEqual(Path(path).parent, tts.output_dir)
            self.assertEqual(Path(path).suffix, ".wav")

    def test_switching_models_unloads_previous(self):
        tts._models.pop("original")
        loaded = MagicMock(conds={"voice": "default"})
        module = MagicMock(ChatterboxTTS=MagicMock(from_pretrained=lambda _: loaded))
        with patch.dict(sys.modules, {"chatterbox": MagicMock(), "chatterbox.tts": module}):
            self.assertIs(tts._get_model("original"), loaded)
        self.assertEqual(list(tts._models), ["original"])
        self.assertEqual(list(tts._default_conditionals), ["original"])

    def test_cuda_arch_support(self):
        torch26 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
        self.assertTrue(tts.cuda_arch_supported((8, 9), torch26))  # RTX 40 via sm_86
        self.assertTrue(tts.cuda_arch_supported((8, 6), torch26))
        self.assertFalse(tts.cuda_arch_supported((12, 0), torch26))  # RTX 50
        self.assertTrue(tts.cuda_arch_supported((12, 0), torch26 + ["compute_90"]))
        self.assertTrue(tts.cuda_arch_supported((9, 0), ["sm_90a"]))

    def test_failed_save_removes_partial_file(self):
        tts.ta.save.side_effect = RuntimeError("disk full")
        path, status = self.generate()
        self.assertIsNone(path)
        self.assertIn("disk full", status)
        self.assertEqual(list(tts.output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
