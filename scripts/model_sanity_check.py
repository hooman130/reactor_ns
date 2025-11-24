"""Minimal INSwapper/hyperswap sanity check without Stable Diffusion WebUI.

This script stubs the webui-specific modules so the existing ReActor patches
can be exercised directly against ONNX models. It does **not** perform face
recognition; instead it feeds synthetic embeddings to validate the model input
ordering and embedding map sizing. Provide --model, --source, and --target
paths to run against real images, or omit them to use synthetic placeholders.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Tuple

# Ensure repository modules are importable when running as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Stub the webui-only modules before importing the ReActor patches. This keeps
# console_log_patch importable in environments that do not include the Stable
# Diffusion WebUI runtime while avoiding changes to the production modules.
# ---------------------------------------------------------------------------
shared = types.SimpleNamespace(
    cmd_opts=types.SimpleNamespace(reactor_loglevel="INFO"),
    opts=types.SimpleNamespace(
        save_to_dirs=False,
        directories_filename_pattern=None,
        save_images_add_number=False,
        samples_filename_pattern=None,
    ),
)


class _FilenameGenerator:
    def __init__(self, p=None, seed=None, prompt=None, image=None):
        pass

    def apply(self, pattern):
        return "stub"


def _next_sequence_number(path, basename):  # pragma: no cover - helper for compatibility
    return 0


authored_images = types.SimpleNamespace(
    FilenameGenerator=_FilenameGenerator, get_next_sequence_number=_next_sequence_number
)
authored_callbacks = types.SimpleNamespace(ImageSaveParams=lambda *a, **k: None)

modules_mod = types.SimpleNamespace(
    shared=shared, images=authored_images, script_callbacks=authored_callbacks
)
sys.modules.setdefault("modules", modules_mod)
sys.modules.setdefault("modules.shared", shared)
sys.modules.setdefault("modules.images", authored_images)
sys.modules.setdefault("modules.script_callbacks", authored_callbacks)

sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault(
    "safetensors.torch",
    types.SimpleNamespace(save_file=lambda *a, **k: None, safe_open=lambda *a, **k: None),
)
sys.modules.setdefault("safetensors", types.SimpleNamespace(torch=sys.modules["safetensors.torch"]))

# ---------------------------------------------------------------------------
# Import patched INSwapper now that stubs are present.
# ---------------------------------------------------------------------------
from scripts.console_log_patch import apply_logging_patch
from insightface.model_zoo.inswapper import INSwapper


class DummyFace:
    """Minimal stand-in for insightface.app.common.Face."""

    def __init__(self, kps: np.ndarray, embedding: np.ndarray):
        self.kps = kps
        self.normed_embedding = embedding


def _load_image(path: Path | None, fallback_size: Tuple[int, int]) -> np.ndarray:
    if path is None:
        h, w = fallback_size
        return np.full((h, w, 3), 127, dtype=np.uint8)

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return img


def _center_kps(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    return np.array(
        [
            [cx - 10, cy - 10],
            [cx + 10, cy - 10],
            [cx, cy],
            [cx - 8, cy + 10],
            [cx + 8, cy + 10],
        ],
        dtype=np.float32,
    )


def _validate_swapper(swapper: INSwapper) -> None:
    """Run lightweight consistency checks on the initialized swapper."""

    latent_dim = swapper.emap.shape[0]
    if swapper.emap.shape[0] != swapper.emap.shape[1]:
        raise RuntimeError(
            f"Embedding map must be square; got {swapper.emap.shape}"  # pragma: no cover - sanity guard
        )

    if latent_dim < 64:
        raise RuntimeError(
            f"Latent dim looks too small to be a face embedding ({latent_dim}); check ONNX metadata."
        )

    # Ensure the input ordering picked by the patched initializer keeps the image first.
    if len(swapper.input_shape) >= 4 and swapper.input_shape[1] not in (3, 4):
        raise RuntimeError(
            f"Unexpected input shape for image tensor: {swapper.input_shape}; model inputs may be mis-ordered."
        )


def run_swap(model_path: Path, source_path: Path | None, target_path: Path | None) -> Path:
    apply_logging_patch(2)

    swapper = INSwapper(model_file=str(model_path))
    _validate_swapper(swapper)

    h, w = swapper.input_size

    target_img = _load_image(target_path, (max(512, h * 2), max(512, w * 2)))
    _load_image(source_path, (h, w))

    kps = _center_kps(target_img)
    embed_dim = swapper.emap.shape[0]
    embedding = np.ones(embed_dim, dtype=np.float32)
    src_face = DummyFace(kps=kps, embedding=embedding)
    tgt_face = DummyFace(kps=kps, embedding=embedding)

    swapped = swapper.get(target_img, tgt_face, src_face, paste_back=True)

    # Basic output sanity checks to ensure the swapper returned a plausible image.
    if swapped.shape[:2] != target_img.shape[:2]:
        raise RuntimeError(
            f"Swapped output shape {swapped.shape[:2]} does not match target image {target_img.shape[:2]}."
        )
    if swapped.dtype != target_img.dtype:
        raise RuntimeError(
            f"Swapped output dtype {swapped.dtype} does not match target image {target_img.dtype}."
        )

    out_path = Path("/tmp") / f"swap_output_{model_path.stem}.png"
    cv2.imwrite(str(out_path), swapped)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quickly validate ONNX swapper models without WebUI.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the ONNX swapper model.")
    parser.add_argument("--source", type=Path, help="Optional source face image (defaults to synthetic patch).")
    parser.add_argument("--target", type=Path, help="Optional target image (defaults to synthetic patch).")
    args = parser.parse_args()

    output = run_swap(args.model, args.source, args.target)
    print(f"Wrote swapped preview to {output}")


if __name__ == "__main__":
    main()
