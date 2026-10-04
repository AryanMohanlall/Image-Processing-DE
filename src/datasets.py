"""Image loading for the BSD500 and CHAOS datasets"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"

BSD500 = "bsd500"
CHAOS = "chaos"
DATASETS: tuple[str, ...] = (BSD500, CHAOS)

# "BDS500" is the spelling the files ship with.
BSD500_DIR = "BDS500"
CHAOS_DIR = "CHAOS"

RGBA_CHANNELS = 4
OPAQUE = 255

LUMA_BT601 = (0.299, 0.587, 0.114)

@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    dataset: str
    path: Path

    def load(self) -> np.ndarray:
        return load_grayscale(self.path)


def to_grayscale(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim == 2:
        return pixels.astype(np.uint8, copy=False)

    red, green, blue = (channel.astype(np.float64) for channel in _opaque_rgb(pixels))
    red_weight, green_weight, blue_weight = LUMA_BT601
    luma = red_weight * red + green_weight * green + blue_weight * blue

    return np.floor(luma + 0.5).astype(np.uint8)


def _opaque_rgb(pixels: np.ndarray) -> tuple[np.ndarray, ...]:
    has_alpha = pixels.shape[2] == RGBA_CHANNELS

    if has_alpha and not np.all(pixels[..., 3] == OPAQUE):
        raise ValueError(
            "image has partial transparency; the background colour to "
            "composite against is undefined for this study"
        )
    return tuple(pixels[..., channel] for channel in range(3))


def load_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as handle:
        grayscale = to_grayscale(np.asarray(handle))
    if grayscale.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got {grayscale.shape}")
    return grayscale


def load_dataset(name: str, data_root: Path | None = None) -> list[ImageRecord]:
    root = data_root or DATA_ROOT
    paths = _image_paths(name, root)
    if not paths:
        raise FileNotFoundError(f"no images found for dataset {name!r} under {root}")
    return [ImageRecord(path.stem, name, path) for path in paths]


def _image_paths(name: str, root: Path) -> list[Path]:
    if name == BSD500:
        return _bsd500_paths(root / BSD500_DIR)
    if name == CHAOS:
        return sorted((root / CHAOS_DIR).glob("IMG-*.png"))
    raise KeyError(f"unknown dataset {name!r}; available: {DATASETS}")


def _bsd500_paths(directory: Path) -> list[Path]:
    images = (p for p in directory.glob("img*.png") if not p.stem.endswith("_gt"))
    return sorted(images, key=lambda p: int(p.stem.removeprefix("img")))



def load_all(data_root: Path | None = None) -> list[ImageRecord]:
    return [record for name in DATASETS for record in load_dataset(name, data_root)]


def manifest(records: list[ImageRecord] | None = None) -> pd.DataFrame:
    records = records if records is not None else load_all()
    return pd.DataFrame([_manifest_row(record) for record in records])


def _manifest_row(record: ImageRecord) -> dict[str, object]:
    image = record.load()
    return {
        "image_id": record.image_id,
        "dataset": record.dataset,
        "path": _relative_to_root(record.path),
        "height": image.shape[0],
        "width": image.shape[1],
        "n_pixels": image.size,
        "min_level": int(image.min()),
        "max_level": int(image.max()),
    }


def _relative_to_root(path: Path) -> str:
    return str(path.relative_to(ROOT))
