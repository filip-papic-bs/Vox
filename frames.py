"""Frame extraction from videos via ffmpeg.

Replaces the pre-frames architecture where Gemini received the raw video file
(and any non-video-native backend couldn't participate in stats extraction).
Now every backend consumes the same list of JPEG-encoded frames, which lets
any vision-capable LLM stand in for stats extraction or persona evaluation.

Knobs:
  frame_step  — 1 = every frame, 2 = every other, 3 = every third, etc.
                Default 1. Higher values trade vision fidelity for token cost,
                which matters at 16 personas × 5 runs = 80 calls per video.
  max_frames  — Safety cap. Set to None for unbounded; otherwise the selected
                frames are evenly down-sampled to fit.
  max_dim     — Downscale longest edge to this many pixels (default 768,
                matching Gemini's native vision tile). Keeps payload sane on
                1080p+ source video.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_MAX_DIM = 768
DEFAULT_JPEG_QUALITY = 4  # ffmpeg's -q:v scale (2 = best, 31 = worst); 4 ≈ JPEG quality 85


class FFmpegNotFound(RuntimeError):
    pass


class FrameExtractionError(RuntimeError):
    pass


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFound(
            "ffmpeg not found on PATH. Install it (e.g. `sudo dnf install ffmpeg` "
            "or `sudo apt install ffmpeg`) and try again."
        )
    return path


def extract_frames(
    video_path: str,
    frame_step: int = 1,
    max_frames: int | None = None,
    max_dim: int = DEFAULT_MAX_DIM,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[bytes]:
    """Return a list of JPEG-encoded frames extracted from `video_path`.

    `frame_step=1` keeps every frame; `frame_step=3` keeps every third frame
    (i.e. skips two between keepers). `max_frames` is a final hard cap applied
    *after* frame_step — useful as a safety net so a 10-minute clip with
    frame_step=1 doesn't accidentally produce 18000 frames.
    """
    if not video_path or not os.path.exists(video_path):
        raise FrameExtractionError(f"video not found: {video_path!r}")
    if frame_step < 1:
        raise FrameExtractionError(f"frame_step must be >= 1, got {frame_step}")

    ffmpeg = _require_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        out_pattern = str(Path(tmp) / "f_%06d.jpg")

        # Filter chain:
        #   select='not(mod(n,N))'  — keep every Nth frame (n is 0-based frame index)
        #   scale=...               — downscale longest edge to max_dim, preserving aspect
        # -vsync 0 stops ffmpeg from duplicating frames to match output fps.
        select_expr = "1" if frame_step == 1 else f"not(mod(n\\,{frame_step}))"
        # min(iw,max_dim) trick: if iw < max_dim, scale=-2:-2 → no upscale.
        # Use force_original_aspect_ratio=decrease to clamp longest edge.
        scale_expr = (
            f"scale='if(gt(iw,ih),min(iw,{max_dim}),-2)':'if(gt(ih,iw),min(ih,{max_dim}),-2)'"
            ":force_original_aspect_ratio=decrease"
        )
        vf = f"select='{select_expr}',{scale_expr}"

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", vf,
            "-vsync", "0",
            "-q:v", str(jpeg_quality),
            out_pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            raise FrameExtractionError(f"ffmpeg failed: {stderr.strip()[:400]}") from e

        frame_paths = sorted(Path(tmp).glob("f_*.jpg"))
        if not frame_paths:
            raise FrameExtractionError(
                "ffmpeg produced no frames — check that the video is decodable and frame_step isn't larger than the total frame count."
            )

        if max_frames is not None and len(frame_paths) > max_frames:
            # Evenly down-sample to the cap so we keep coverage of the whole clip.
            step = len(frame_paths) / max_frames
            keep_indices = {int(i * step) for i in range(max_frames)}
            frame_paths = [p for i, p in enumerate(frame_paths) if i in keep_indices]

        return [p.read_bytes() for p in frame_paths]
