"""Shared output handling: never report success after an image write fails."""
from pathlib import Path

import cv2


def write_image(path: Path, image) -> None:
    try:
        success = cv2.imwrite(str(path), image)
    except cv2.error as exc:
        raise OSError(f"Cannot write image: {path}") from exc
    if not success:
        raise OSError(f"Cannot write image: {path}")
