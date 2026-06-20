from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "Images"
SPRITES_DIR = ASSETS_DIR / "Sprites"
TRACK_ASSET_DIR = ASSETS_DIR / "TracksMapGen"
GENERATED_TRACK_FRONT_PATH = PROJECT_ROOT / "randomGeneratedTrackFront.png"
GENERATED_TRACK_BACK_PATH = PROJECT_ROOT / "randomGeneratedTrackBack.png"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate