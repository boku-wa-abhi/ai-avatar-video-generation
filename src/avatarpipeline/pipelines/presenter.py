"""
avatarpipeline.pipelines.presenter — Persistent slide narrator with presenter lip-sync.

This pipeline is designed for long-running presentation jobs where the user
needs reusable assets:
  1. Validate PDF + JSON narration sync.
  2. Generate or reuse per-slide narration audio.
  3. Generate or reuse per-slide lip-sync presenter clips.
  4. Save a master audio file for the selected slides.
  5. Render the current PDF pages.
  6. Composite the presenter in the bottom-left corner for each selected slide.
  7. Either return per-slide outputs or concatenate them into a single export.

All persistent assets are written under ``data/presentations/<project-tag>/`` so
they can be reused later even when the PDF changes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Generator

from loguru import logger

from avatarpipeline import PRESENTATIONS_DIR
from avatarpipeline.core.media import (
    audio_duration as _audio_duration,
    concat_audio as _concat_audio,
    generate_silence as _gen_silence,
    normalize_to_16k_mono as _normalize_audio,
    video_info as _video_info,
)
from avatarpipeline.pipelines._slide_pdf import render_slides
from avatarpipeline.pipelines._validate import validate_sync

DEFAULT_SELECTION = "all"
DEFAULT_OUTPUT_MODE = "Output everything"
OUTPUT_MODE_ALL = "Output everything"
OUTPUT_MODE_ONE_BY_ONE = "Output one by one"
PRESENTER_LAYOUT_VERSION = "consulting_left_presenter_v4"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha1_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slugify(text: str) -> str:
    chars: list[str] = []
    for ch in (text or "").strip().lower():
        chars.append(ch if ch.isalnum() else "-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "presentation"


def _concat_videos(paths: list[str | Path], output_path: str | Path) -> None:
    list_path = Path(output_path).with_suffix(".concat.txt")
    with open(list_path, "w") as f:
        for p in paths:
            f.write(f"file '{Path(p).resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video concat failed:\n{result.stderr[-700:]}")


def parse_slide_selection(selection: str | None, slide_count: int) -> list[int]:
    text = (selection or DEFAULT_SELECTION).strip().lower()
    if not text or text == "all":
        return list(range(1, slide_count + 1))

    selected: list[int] = []
    seen: set[int] = set()
    for token in [t.strip() for t in text.split(",") if t.strip()]:
        if "-" in token:
            start_s, end_s = [part.strip() for part in token.split("-", 1)]
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError(f"Invalid slide range {token!r}. Use ascending ranges like '1-3'.")
            items = range(start, end + 1)
        else:
            items = [int(token)]

        for item in items:
            if item < 1 or item > slide_count:
                raise ValueError(f"Slide {item} is out of range. Valid slide numbers are 1..{slide_count}.")
            if item not in seen:
                seen.add(item)
                selected.append(item)

    if not selected:
        raise ValueError("No slides were selected.")
    return selected


def _selection_slug(selected_slides: list[int], slide_count: int) -> str:
    if selected_slides == list(range(1, slide_count + 1)):
        return "all"
    return "slides_" + "_".join(str(n) for n in selected_slides)


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"slides": {}, "exports": []}
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {"slides": {}, "exports": []}


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))


def _copy_source_file(source_path: str | Path, dest_dir: Path) -> Path:
    source_path = Path(source_path).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_path.name
    if source_path != dest:
        shutil.copy2(source_path, dest)
    return dest


def _write_json_source(json_data: dict | list, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    return dest_path


def _existing_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    return path.resolve() if path.exists() else None


def _ensure_rendered_slides(
    pdf_path: str | Path,
    render_dir: Path,
    slide_count: int,
    pdf_hash: str,
) -> tuple[list[Path], bool]:
    existing = sorted(render_dir.glob("slide_*.png"))
    hash_marker = render_dir / ".pdf_hash"
    existing_hash = hash_marker.read_text().strip() if hash_marker.exists() else ""
    if existing_hash == pdf_hash and len(existing) == slide_count:
        return existing, True
    if render_dir.exists():
        shutil.rmtree(render_dir, ignore_errors=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    rendered = render_slides(pdf_path, render_dir)
    renamed: list[Path] = []
    for idx, path in enumerate(rendered, start=1):
        dest = render_dir / f"slide_{idx:03d}.png"
        if path != dest:
            path.replace(dest)
        renamed.append(dest)
    hash_marker.write_text(pdf_hash)
    return renamed, False


def _copy_avatar_to_project(avatar_path: str | Path, avatar_dir: Path) -> Path:
    avatar_path = Path(avatar_path).expanduser().resolve()
    avatar_dir.mkdir(parents=True, exist_ok=True)
    dest = avatar_dir / avatar_path.name
    if avatar_path != dest:
        shutil.copy2(avatar_path, dest)
    return dest


def _compose_presenter_overlay(
    slide_image: str | Path,
    presenter_video: str | Path,
    output_path: str | Path,
    logo_image: str | Path | None = None,
    bottom_margin: int = 36,
    left_margin: int = 40,
    width_ratio: float = 0.28,
) -> str:
    slide_image = Path(slide_image).resolve()
    presenter_video = Path(presenter_video).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dims = _video_info(presenter_video)
    canvas_w, canvas_h = 1920, 1080
    header_h = 108
    content_x, content_y = 292, 134
    content_w, content_h = 1588, 818
    slide_w = content_w
    slide_h = content_h

    presenter_w = 320
    presenter_x = max(48, left_margin + 24)
    presenter_y_expr = "952-overlay_h"

    logo_path = _existing_path(logo_image)
    logo_h = 50
    logo_x = 96
    logo_y = (header_h - logo_h) // 2
    fps = max(1.0, float(dims.get("fps", 25.0)))

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=white:s={canvas_w}x{canvas_h}:r={fps}",
        "-loop", "1", "-i", str(slide_image),
        "-i", str(presenter_video),
    ]
    filter_parts = [
        (
            "[0:v]format=rgba,"
            f"drawbox=x=0:y={header_h}:w={canvas_w}:h=2:color=0xDADFD6@1.0:t=fill[base]"
        ),
        f"[1:v]scale=w={slide_w}:h={slide_h}:force_original_aspect_ratio=decrease,format=rgba[slide]",
        f"[2:v]scale={presenter_w}:-2,colorkey=0xFFFFFF:0.18:0.06,format=rgba[presenter]",
    ]
    bg_label = "base"
    if logo_path is not None:
        cmd += ["-i", str(logo_path)]
        filter_parts.append(f"[3:v]scale=-2:{logo_h},format=rgba[logo]")
        filter_parts.append(f"[base][logo]overlay={logo_x}:{logo_y}:format=auto[with_logo]")
        bg_label = "with_logo"
    filter_parts.append(
        f"[{bg_label}][slide]overlay={content_x}+(({content_w}-overlay_w)/2):{content_y}+(({content_h}-overlay_h)/2):format=auto[with_slide]"
    )
    filter_parts.append(
        f"[with_slide][presenter]overlay={presenter_x}:{presenter_y_expr}:format=auto[vout]"
    )
    cmd += [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", "2:a?",
        "-t", str(max(0.1, dims.get("duration", 0.0))),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Presenter composite failed:\n{result.stderr[-800:]}")
    return str(output_path)


def compose_slide_presenter_video(
    pdf_path: str | Path,
    json_data: dict | list,
    avatar_path: str | Path,
    json_source_path: str | Path | None = None,
    logo_path: str | Path | None = None,
    project_tag: str | None = None,
    slide_selection: str | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    voice: str = "af_heart",
    pause_seconds: float = 1.5,
    tts_engine: str = "kokoro",
    mlx_voice_choice: str | None = None,
    mlx_preset_voice: str | None = None,
    mlx_model_id: str | None = None,
    mlx_language: str | None = None,
    lipsync_engine: str = "musetalk",
    enhance_face: bool = False,
    mt_batch_size: int = 8,
    mt_bbox_shift: int = 0,
    st_expression_scale: float = 1.0,
    st_pose_style: int = 0,
    st_still: bool = True,
    st_preprocess: str = "full",
) -> Generator[tuple[str, dict | None], None, None]:
    """Build persistent per-slide presenter assets and export selected outputs."""
    pdf_path = Path(pdf_path).resolve()
    avatar_path = Path(avatar_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar file not found: {avatar_path}")

    result = validate_sync(pdf_path, json_data)
    if not result.ok:
        raise ValueError("Validation failed:\n" + "\n".join(result.errors))

    normalized_json = result.json_data
    slide_count = result.slide_count
    selected_slides = parse_slide_selection(slide_selection, slide_count)
    selection_slug = _selection_slug(selected_slides, slide_count)
    project_slug = _slugify(project_tag or pdf_path.stem)
    project_dir = PRESENTATIONS_DIR / project_slug
    manifest_path = project_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_hash = _sha1_file(pdf_path)[:12]
    avatar_hash = _sha1_file(avatar_path)[:12]
    asset_dirs = {
        "source": project_dir / "source",
        "avatar": project_dir / "avatar",
        "audio": project_dir / "audio",
        "lipsync": project_dir / "lipsync",
        "masters": project_dir / "masters",
        "slides": project_dir / "slides",
        "composite": project_dir / f"composite_{pdf_hash}",
        "exports": project_dir / "exports",
    }
    for path in asset_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    project_pdf = _copy_source_file(pdf_path, asset_dirs["source"])
    project_avatar = _copy_avatar_to_project(avatar_path, asset_dirs["avatar"])
    if json_source_path and Path(json_source_path).exists():
        project_json = _copy_source_file(json_source_path, asset_dirs["source"])
    else:
        project_json = _write_json_source(normalized_json, asset_dirs["source"] / "narration.json")
    if logo_path and Path(logo_path).exists():
        project_logo = _copy_source_file(logo_path, asset_dirs["source"])
    else:
        project_logo = _existing_path(manifest.get("logo_path"))
    logo_hash = _sha1_file(project_logo)[:12] if project_logo else None

    manifest.update(
        {
            "project_tag": project_slug,
            "updated_at": datetime.now().isoformat(),
            "pdf_path": str(project_pdf),
            "pdf_hash": pdf_hash,
            "json_path": str(project_json),
            "logo_path": str(project_logo) if project_logo else None,
            "logo_hash": logo_hash,
            "layout_version": PRESENTER_LAYOUT_VERSION,
            "avatar_path": str(project_avatar),
            "avatar_hash": avatar_hash,
            "slide_count": slide_count,
            "warnings": result.warnings,
            "slides": manifest.get("slides", {}),
            "exports": manifest.get("exports", []),
        }
    )

    yield f"Validation passed — {slide_count} pages, selected slides: {selected_slides}", None

    entries_by_num = {
        int(entry["slide_number"]): entry
        for entry in normalized_json.get("slides", [])
    }

    default_pause = float(normalized_json.get("default_pause_seconds", pause_seconds))
    default_display = float(normalized_json.get("default_display_seconds", 0.0) or 0.0)

    # Lazy init heavy backends.
    voice_gen = None
    mlx_studio = None
    lipsync_runner = None
    prepared_avatar = None
    face_enhancer = None

    per_slide_audio: dict[int, Path] = {}
    per_slide_video: dict[int, Path] = {}
    per_slide_duration: dict[int, float] = {}
    report_lines = [
        f"Project: {project_slug}",
        f"PDF: {project_pdf.name}",
        f"JSON: {project_json.name}",
        f"Logo: {project_logo.name if project_logo else 'none'}",
        f"Avatar: {project_avatar.name}",
        f"Layout: {PRESENTER_LAYOUT_VERSION}",
        f"Selected slides: {selected_slides}",
    ]

    for idx, slide_num in enumerate(selected_slides, start=1):
        entry = entries_by_num[slide_num]
        narration = (entry.get("narration") or "").strip()
        min_display = float(entry.get("display_seconds", default_display) or 0.0)
        slide_pause = float(entry.get("pause_seconds", default_pause) or 0.0)

        audio_key = {
            "slide": slide_num,
            "narration": narration,
            "tts_engine": tts_engine,
            "voice": voice,
            "mlx_voice_choice": mlx_voice_choice,
            "mlx_preset_voice": mlx_preset_voice,
            "mlx_model_id": mlx_model_id,
            "mlx_language": mlx_language,
        }
        audio_hash = _sha1_text(json.dumps(audio_key, sort_keys=True))[:12]
        audio_path = asset_dirs["audio"] / f"slide_{slide_num:03d}_{audio_hash}.wav"

        if not audio_path.exists():
            yield f"Generating narration audio {idx}/{len(selected_slides)} — slide {slide_num}", None
            if narration:
                if (tts_engine or "kokoro").strip().lower() == "mlx":
                    if mlx_studio is None:
                        from avatarpipeline.engines.tts.mlx import MlxVoiceStudio
                        mlx_studio = MlxVoiceStudio()

                    raw_audio = audio_path.with_name(f"{audio_path.stem}_raw.wav")
                    if mlx_preset_voice:
                        generated = mlx_studio.synthesize_with_preset(
                            text=narration,
                            preset_voice=mlx_preset_voice,
                            model_id=mlx_model_id,
                            lang_code=mlx_language or "ja",
                            output_path=str(raw_audio),
                        )
                    else:
                        if not mlx_voice_choice:
                            raise ValueError("Select an MLX saved voice or preset voice.")
                        generated = mlx_studio.synthesize_with_voice(
                            text=narration,
                            voice_choice=mlx_voice_choice,
                            model_id=mlx_model_id,
                            lang_code=mlx_language or "ja",
                            output_path=str(raw_audio),
                        )
                    _normalize_audio(generated, audio_path)
                    Path(generated).unlink(missing_ok=True)
                else:
                    if voice_gen is None:
                        from avatarpipeline.engines.tts.kokoro import VoiceGenerator
                        voice_gen = VoiceGenerator()
                    voice_gen.generate(narration, voice=voice, out_path=str(audio_path))
            else:
                _gen_silence(0.1, audio_path)
        else:
            yield f"Reusing narration audio {idx}/{len(selected_slides)} — slide {slide_num}", None

        audio_duration = max(0.1, _audio_duration(audio_path))
        display_duration = max(audio_duration, min_display)
        per_slide_duration[slide_num] = display_duration + slide_pause
        per_slide_audio[slide_num] = audio_path

        lipsync_key = {
            "audio_hash": audio_hash,
            "avatar_hash": avatar_hash,
            "lipsync_engine": lipsync_engine,
            "enhance_face": enhance_face,
            "mt_batch_size": mt_batch_size,
            "mt_bbox_shift": mt_bbox_shift,
            "st_expression_scale": st_expression_scale,
            "st_pose_style": st_pose_style,
            "st_still": st_still,
            "st_preprocess": st_preprocess,
        }
        lipsync_hash = _sha1_text(json.dumps(lipsync_key, sort_keys=True))[:12]
        lipsync_path = asset_dirs["lipsync"] / f"slide_{slide_num:03d}_{lipsync_hash}.mp4"

        if not lipsync_path.exists():
            yield f"Generating lip-sync presenter {idx}/{len(selected_slides)} — slide {slide_num}", None
            if lipsync_engine in ("sadtalker", "sadtalker_hd"):
                if lipsync_runner is None:
                    from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference
                    lipsync_runner = SadTalkerInference(preset=lipsync_engine)
                clip_path = lipsync_runner.run(
                    str(project_avatar),
                    str(audio_path),
                    output_path=str(lipsync_path),
                    expression_scale=st_expression_scale,
                    pose_style=st_pose_style,
                    still=st_still,
                    preprocess=st_preprocess,
                )
            else:
                if lipsync_runner is None:
                    from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
                    lipsync_runner = MuseTalkInference()
                    prepared_avatar = project_dir / f"{project_avatar.stem}_musetalk_prepared.png"
                    if not prepared_avatar.exists():
                        prepared_path = lipsync_runner.prepare_avatar(str(project_avatar))
                        shutil.copy2(prepared_path, prepared_avatar)
                    lipsync_runner.prepare_avatar(str(project_avatar))
                clip_dir = asset_dirs["lipsync"] / f"musetalk_slide_{slide_num:03d}_{lipsync_hash}"
                clip_dir.mkdir(parents=True, exist_ok=True)
                clip_path = lipsync_runner.run(
                    str(prepared_avatar or project_avatar),
                    str(audio_path),
                    output_dir=str(clip_dir),
                    batch_size=mt_batch_size,
                    bbox_shift=mt_bbox_shift,
                )
                shutil.copy2(clip_path, lipsync_path)

            if enhance_face:
                yield f"Enhancing presenter clip {idx}/{len(selected_slides)} — slide {slide_num}", None
                if face_enhancer is None:
                    from avatarpipeline.postprocess.enhancer import FaceEnhancer
                    face_enhancer = FaceEnhancer()
                enhanced_path = lipsync_path.with_name(f"{lipsync_path.stem}_enhanced.mp4")
                face_enhancer.enhance(str(lipsync_path), str(enhanced_path))
                shutil.move(str(enhanced_path), str(lipsync_path))
        else:
            yield f"Reusing lip-sync presenter {idx}/{len(selected_slides)} — slide {slide_num}", None

        per_slide_video[slide_num] = lipsync_path
        manifest["slides"][str(slide_num)] = {
            "slide_number": slide_num,
            "audio_hash": audio_hash,
            "audio_path": str(audio_path),
            "audio_duration": audio_duration,
            "display_duration": per_slide_duration[slide_num],
            "lipsync_hash": lipsync_hash,
            "lipsync_path": str(lipsync_path),
            "tts_engine": tts_engine,
            "voice": voice,
            "mlx_voice_choice": mlx_voice_choice,
            "mlx_preset_voice": mlx_preset_voice,
            "mlx_model_id": mlx_model_id,
            "mlx_language": mlx_language,
            "lipsync_engine": lipsync_engine,
            "enhance_face": enhance_face,
            "avatar_path": str(project_avatar),
            "updated_at": datetime.now().isoformat(),
        }

    yield "Building master audio…", None
    master_segments: list[Path] = []
    for slide_num in selected_slides:
        entry = entries_by_num[slide_num]
        audio_path = per_slide_audio[slide_num]
        audio_duration = max(0.1, _audio_duration(audio_path))
        min_display = float(entry.get("display_seconds", default_display) or 0.0)
        slide_pause = float(entry.get("pause_seconds", default_pause) or 0.0)
        display_duration = max(audio_duration, min_display)
        hold_duration = max(0.0, display_duration - audio_duration)
        master_segments.append(audio_path)
        if hold_duration > 0:
            hold_path = asset_dirs["masters"] / f"hold_{slide_num:03d}.wav"
            _gen_silence(hold_duration, hold_path)
            master_segments.append(hold_path)
        if slide_pause > 0:
            pause_path = asset_dirs["masters"] / f"pause_{slide_num:03d}.wav"
            _gen_silence(slide_pause, pause_path)
            master_segments.append(pause_path)

    master_audio_path = asset_dirs["masters"] / f"master_{selection_slug}.wav"
    _concat_audio(master_segments, master_audio_path)

    yield "Rendering slides…", None
    slide_images, reused_slides = _ensure_rendered_slides(project_pdf, asset_dirs["slides"], slide_count, pdf_hash)
    slide_image_map = {idx: path for idx, path in enumerate(slide_images, start=1)}

    composite_paths: list[Path] = []
    for idx, slide_num in enumerate(selected_slides, start=1):
        yield f"Composing slide {idx}/{len(selected_slides)} — slide {slide_num}", None
        composite_hash = _sha1_text(
            json.dumps(
                {
                    "slide_num": slide_num,
                    "layout_version": PRESENTER_LAYOUT_VERSION,
                    "pdf_hash": pdf_hash,
                    "logo_hash": logo_hash,
                    "presenter_clip": str(per_slide_video[slide_num]),
                    "duration": per_slide_duration[slide_num],
                },
                sort_keys=True,
            )
        )[:12]
        composite_path = asset_dirs["composite"] / f"slide_{slide_num:03d}_{composite_hash}.mp4"
        if not composite_path.exists():
            _compose_presenter_overlay(
                slide_image=slide_image_map[slide_num],
                presenter_video=per_slide_video[slide_num],
                output_path=composite_path,
                logo_image=project_logo,
            )
        composite_paths.append(composite_path)

    export_files: list[Path] = [master_audio_path]
    export_files.extend(per_slide_audio[n] for n in selected_slides)
    export_files.extend(per_slide_video[n] for n in selected_slides)
    export_files.extend(composite_paths)

    preview_path = composite_paths[-1]
    final_export = None
    if output_mode == OUTPUT_MODE_ALL:
        yield "Exporting combined presentation…", None
        final_export = asset_dirs["exports"] / f"presenter_{selection_slug}_{run_id}.mp4"
        _concat_videos(composite_paths, final_export)
        preview_path = final_export
        export_files = [final_export] + export_files

    manifest["exports"].append(
        {
            "created_at": datetime.now().isoformat(),
            "selection": selected_slides,
            "selection_slug": selection_slug,
            "output_mode": output_mode,
            "master_audio_path": str(master_audio_path),
            "composite_paths": [str(p) for p in composite_paths],
            "final_export": str(final_export) if final_export else None,
            "pdf_hash": pdf_hash,
        }
    )
    _save_manifest(manifest_path, manifest)

    report_lines.extend(
        [
            f"Master audio: {master_audio_path.name}",
            f"Slides: {'reused existing numbered renders' if reused_slides else 'rendered and saved as slide_001.png, slide_002.png, ...'}",
            f"Presenter clips: {len(per_slide_video)}",
            f"Composite slide videos: {len(composite_paths)}",
            f"Output mode: {output_mode}",
            f"Project folder: {project_dir}",
        ]
    )
    if final_export:
        report_lines.append(f"Combined export: {final_export.name}")

    yield "Done!", {
        "preview_video": str(preview_path),
        "generated_files": [str(p) for p in export_files],
        "report": "\n".join(report_lines),
        "project_dir": str(project_dir),
    }
