from pathlib import Path


def test_media_audio_helpers_generate_probe_and_concat(tmp_path):
    from avatarpipeline.core.media import audio_duration, concat_audio, generate_silence

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "concat.wav"

    generate_silence(0.2, first)
    generate_silence(0.3, second)
    concat_audio([first, second], output)

    assert first.exists()
    assert second.exists()
    assert output.exists()
    assert 0.45 <= audio_duration(output) <= 0.65


def test_media_video_info_reports_dimensions(tmp_path):
    import subprocess

    from avatarpipeline.core.media import video_info

    video = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=160x90:d=0.2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video),
        ],
        capture_output=True,
        check=True,
    )

    info = video_info(video)

    assert info["width"] == 160
    assert info["height"] == 90
    assert info["fps"] > 0
