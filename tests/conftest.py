"""
Dogfood swarm fixtures — real images, real model, real inference.

No mocks. SigLIP2 loads once per session and every test hits the GPU.
This IS the dogfood: if the model can't distinguish a sword from a
cooking pot, we want to know.
"""

import os
from pathlib import Path

import pytest

from ai_eyes_mcp.engine import SigLIPEngine

# ---------------------------------------------------------------------------
# Image paths — real assets from the workspace
# ---------------------------------------------------------------------------

SPRITE_DIR = Path("F:/AI/sprite-foundry/pipeline/gold/crops")
PHOTO_DIR = Path("F:/AI/backpropagate/.venv/Lib/site-packages/gradio/media_assets/images")
ENEMY_DIR = Path("F:/AI/the-fractured-road/assets/sprites/ch1-enemies/assets")


def _sprite(name: str) -> str:
    if not SPRITE_DIR.exists():
        pytest.skip(f"Sprite directory not found: {SPRITE_DIR}")
    p = SPRITE_DIR / name
    if not p.exists():
        pytest.skip(f"Missing test sprite: {p}")
    return str(p)


def _photo(name: str) -> str:
    if not PHOTO_DIR.exists():
        pytest.skip(f"Photo directory not found: {PHOTO_DIR}")
    p = PHOTO_DIR / name
    if not p.exists():
        pytest.skip(f"Missing test photo: {p}")
    return str(p)


def _enemy(subdir: str, name: str) -> str:
    if not ENEMY_DIR.exists():
        pytest.skip(f"Enemy directory not found: {ENEMY_DIR}")
    p = ENEMY_DIR / subdir / "albedo" / name
    if not p.exists():
        pytest.skip(f"Missing test enemy: {p}")
    return str(p)


# --- Sprites with known content ---

@pytest.fixture(scope="session")
def knight_sword_front():
    """Knight holding a sword and shield — front view."""
    return _sprite("h3d_knight_sword_shield_batch_20260409_070001_candidate_0_front.png")


@pytest.fixture(scope="session")
def knight_sword_left():
    """Knight holding a sword and shield — left view (same character)."""
    return _sprite("h3d_knight_sword_shield_batch_20260409_070001_candidate_0_left.png")


@pytest.fixture(scope="session")
def knight_battleaxe_front():
    """Knight holding a battleaxe — front view."""
    return _sprite("h3d_knight_battleaxe_turn_20260409_061535_root_front.png")


@pytest.fixture(scope="session")
def goblin_cook_front():
    """Goblin cook — no weapon, cooking theme."""
    return _sprite("goblin_cook_front.png")


@pytest.fixture(scope="session")
def goblin_merchant_front():
    """Goblin merchant — no weapon, commerce theme."""
    return _sprite("goblin_merchant_front.png")


@pytest.fixture(scope="session")
def hero_bard_front():
    """Bard hero — front view, musical/performance theme."""
    return _sprite("hero_bard_front.png")


# --- Natural photos ---

@pytest.fixture(scope="session")
def photo_cheetah():
    return _photo("cheetah.jpg")


@pytest.fixture(scope="session")
def photo_lion():
    return _photo("lion.jpg")


@pytest.fixture(scope="session")
def photo_bus():
    return _photo("bus.png")


@pytest.fixture(scope="session")
def photo_tower():
    return _photo("tower.jpg")


# --- Enemy sprites ---

@pytest.fixture(scope="session")
def avar_armed_front():
    """Armed avar enemy — has visible weapon."""
    return _enemy("avar_armed", "front.png")


@pytest.fixture(scope="session")
def avar_desperate_front():
    """Desperate avar — unarmed variant."""
    return _enemy("avar_desperate", "front.png")


# --- Engine (loaded once per session) ---

@pytest.fixture(scope="session")
def engine():
    """Session-scoped SigLIP2 engine — loads model once, reused for all tests."""
    e = SigLIPEngine(
        cache_dir=os.environ.get("AI_EYES_MODEL_DIR", "F:/AI-Models/huggingface"),
    )
    # Force load now so we see any load failures immediately
    try:
        e._ensure_loaded()
    except (OSError, RuntimeError, ImportError) as exc:
        pytest.skip(f"SigLIP2 model not available: {exc}")
    return e
