"""Sound-related utility functions."""

from __future__ import annotations

__all__ = [
    "get_full_sound_file_path",
]

import os
from typing import TYPE_CHECKING

from ..utils.file_ops import seek_full_path_from_defaults

if TYPE_CHECKING:
    from pathlib import Path

    from manim.typing import StrPath


def audio_timeline() -> bool:
    """MANIM_AUDIO_TIMELINE=1: skip every animation but advance time as if
    rendering and keep the sounds, then write only the scene's .wav.
    """
    return os.environ.get("MANIM_AUDIO_TIMELINE") == "1"


# Still in use by add_sound() function in scene_file_writer.py
def get_full_sound_file_path(sound_file_name: StrPath, assets_dir: Path) -> Path:
    """Locate a sound path directly or relative to ``assets_dir``."""
    return seek_full_path_from_defaults(
        sound_file_name,
        default_dir=assets_dir,
        extensions=[".wav", ".mp3"],
    )
