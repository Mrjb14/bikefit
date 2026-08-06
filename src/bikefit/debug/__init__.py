"""Vidéo de debug annotée (squelette + métriques superposées), toute source confondue."""

from bikefit.debug.video import (
    DebugVideoWriter,
    render_debug_video_from_landmarks,
    render_debug_video_live,
)

__all__ = [
    "DebugVideoWriter",
    "render_debug_video_from_landmarks",
    "render_debug_video_live",
]
