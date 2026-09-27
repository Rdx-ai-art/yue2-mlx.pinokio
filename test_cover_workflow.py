import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import gradio as gr
import numpy as np
import shutil
import app


class TestCoverWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_ui_contains_cover_step_components(self):
        """Verify that build_ui initializes properly and includes all new Cover step elements."""
        demo = app.build_ui()
        self.assertIsInstance(demo, gr.Blocks)
        
        # Check that relevant component variable names or labels are registered
        labels = [getattr(comp, "label", None) for comp in demo.blocks.values() if hasattr(comp, "label")]
        self.assertTrue(any("ABC Score" in str(label) for label in labels))
        self.assertTrue(any("Upload Audio" in str(label) for label in labels))

    def test_transcribe_audio_validation(self):
        """Verify input validations for _transcribe_audio_to_abc."""
        # Test invalid task
        with self.assertRaises(gr.Error) as ctx:
            app._transcribe_audio_to_abc("non_existent.wav", "invalid_task")
        self.assertIn("Invalid task", str(ctx.exception))

        # Test missing audio file
        with self.assertRaises(gr.Error) as ctx:
            app._transcribe_audio_to_abc("non_existent_file.wav", "melody-full")
        self.assertIn("No audio provided", str(ctx.exception))

    def test_transcribe_audio_success(self):
        """Verify that _transcribe_audio_to_abc returns (abc_text, status) upon successful transcription."""
        fake_audio = self.tmp_dir / "sample.wav"
        fake_audio.write_bytes(b"RIFFdummydata")

        # Mock lyra transcribe function
        def mock_transcribe(audio, output, task, cache_dir, cancelled, progress):
            output_path = Path(output)
            (output_path / "score.abc").write_text("X:1\nT:Test Song\nK:C\nC D E F|", encoding="utf-8")
            return {"status": "ok"}

        with patch.dict("sys.modules", {"lyra.transcription.pipeline": MagicMock(transcribe=mock_transcribe)}):
            abc_text, status = app._transcribe_audio_to_abc(str(fake_audio), "melody-full")
            self.assertEqual(abc_text, "X:1\nT:Test Song\nK:C\nC D E F|")
            self.assertIn("Transcription complete", status)

    def test_generate_cover_validation(self):
        """Verify input validations for _generate_cover_from_abc."""
        # Test empty ABC
        with self.assertRaises(gr.Error) as ctx:
            app._generate_cover_from_abc(
                abc_text="",
                task="melody-full",
                style="rock",
                lyrics="lyrics",
                seed=42,
                cfg_scale=1.0,
                steps=8,
                variant="8bit",
                save_format="WAV",
                lora_adapters=[],
                lora_scale=1.0,
            )
        self.assertIn("No ABC score provided", str(ctx.exception))

        # Test empty Style
        with self.assertRaises(gr.Error) as ctx:
            app._generate_cover_from_abc(
                abc_text="X:1\nK:C\nC D E F|",
                task="melody-full",
                style="",
                lyrics="lyrics",
                seed=42,
                cfg_scale=1.0,
                steps=8,
                variant="8bit",
                save_format="WAV",
                lora_adapters=[],
                lora_scale=1.0,
            )
        self.assertIn("Style is required", str(ctx.exception))

    def test_generate_cover_success(self):
        """Verify _generate_cover_from_abc successfully runs pipeline and saves output."""
        mock_pipe = MagicMock()
        mock_audio = np.zeros((48000,), dtype=np.float32)
        mock_pipe.return_value = {"audio": mock_audio}

        with patch("app.Yue2PipelineMLX", return_value=mock_pipe):
            if hasattr(app._generate_cover_from_abc, "_pipe"):
                app._generate_cover_from_abc._pipe = None

            audio_output, info = app._generate_cover_from_abc(
                abc_text="X:1\nK:C\nC D E F|",
                task="melody-full",
                style="acoustic pop",
                lyrics="hello world",
                seed=1234,
                cfg_scale=1.0,
                steps=8,
                variant="8bit",
                save_format="WAV",
                lora_adapters=[],
                lora_scale=1.0,
            )

            self.assertEqual(audio_output[0], 48000)
            self.assertIn("Cover saved", info)

    def test_cover_song_all_in_one(self):
        """Verify _cover_song coordinates transcription and generation into 3 outputs."""
        fake_audio = self.tmp_dir / "sample.wav"
        fake_audio.write_bytes(b"RIFFdummydata")

        with patch("app._transcribe_audio_to_abc", return_value=("X:1\nK:C\nC D E F|", "Transcription done")):
            with patch("app._generate_cover_from_abc", return_value=((48000, np.zeros((48000,), dtype=np.float32)), "Cover saved")):
                audio_output, abc_out, status = app._cover_song(
                    audio_file=str(fake_audio),
                    task="melody-full",
                    style="pop",
                    lyrics="la la",
                    seed=1,
                    cfg_scale=1.0,
                    steps=8,
                    variant="8bit",
                    save_format="WAV",
                    lora_adapters=[],
                    lora_scale=1.0,
                )

                self.assertEqual(audio_output[0], 48000)
                self.assertEqual(abc_out, "X:1\nK:C\nC D E F|")
                self.assertIn("Transcription done", status)
                self.assertIn("Cover saved", status)


if __name__ == "__main__":
    unittest.main()
