from pathlib import Path


def test_avatar_pipeline_smoke_with_fake_engines(tmp_path, monkeypatch):
    import subprocess

    import avatarpipeline.engines as engines
    from avatarpipeline.core.media import generate_silence, normalize_to_16k_mono, video_info
    from avatarpipeline.pipelines.avatar import run_pipeline

    class FakeTts:
        def generate(self, text: str, voice: str, out_path: str) -> str:
            generate_silence(0.8, out_path)
            return str(Path(out_path).resolve())

        def convert_to_16k(self, wav_path: str, out_path: str) -> str:
            normalize_to_16k_mono(wav_path, out_path)
            return str(Path(out_path).resolve())

        def list_voices(self) -> list[str]:
            return ["fake"]

    class FakeLipsync:
        def prepare_avatar(self, avatar_png: str) -> str:
            return avatar_png

        def run(self, avatar_png: str, audio_wav: str, **kwargs) -> str:
            output = tmp_path / "fake_lipsync.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1:r=25",
                    "-i", audio_wav,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-shortest",
                    "-pix_fmt", "yuv420p",
                    str(output),
                ],
                capture_output=True,
                check=True,
            )
            return str(output)

    monkeypatch.setattr(engines, "get_tts_engine", lambda name: FakeTts())
    monkeypatch.setattr(engines, "get_lipsync_engine", lambda name: FakeLipsync())

    output = tmp_path / "avatar_pipeline.mp4"
    result = run_pipeline(
        script="Smoke test.",
        output_path=str(output),
        include_enhance=False,
        include_captions=False,
        lipsync_engine="musetalk",
    )

    assert Path(result).exists()
    assert video_info(result)["width"] == 1080
