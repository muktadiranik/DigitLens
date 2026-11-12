"""Detect digit-shaped components and classify their normalized crops."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]
Features = NDArray[np.float32]
UNKNOWN_CLASS = 10
MIN_CONFIDENCE = 0.72
MIN_MARGIN = 0.20
MAX_DIGITS = 16


class Detection(TypedDict):
    """A glyph and its bounding box in normalized image coordinates."""

    x: float
    y: float
    width: float
    height: float
    digit: str


class Result(TypedDict):
    """The detected row, with a placeholder for each uncertain glyph."""

    sequence: str
    digits: list[Detection]


@dataclass(frozen=True)
class Glyph:
    """One candidate component in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    contrast: float
    image: GrayImage

    @property
    def center_y(self) -> float:
        """Return the vertical midpoint of the bounding box."""
        return self.y + self.height / 2


def normalize_glyph(image: GrayImage) -> GrayImage:
    """Center bright ink on black in a 28×28 tile, preserving aspect ratio."""
    ink = cv2.findNonZero(np.where(image >= 32, 255, 0).astype(np.uint8))
    tile = np.zeros((28, 28), dtype=np.uint8)
    if ink is None:
        return tile

    x, y, width, height = cv2.boundingRect(ink)
    crop = image[y : y + height, x : x + width]
    scale = 22 / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(
        crop, (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    tile[top : top + new_height, left : left + new_width] = resized
    return tile


class GlyphFeatures:
    """Extract HOG and a downsampled intensity grid from each glyph."""

    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)

    def extract(self, image: GrayImage) -> Features:
        """Create the same feature vector for training and inference."""
        tile = normalize_glyph(image)
        gradients = self.hog.compute(tile).reshape(-1)
        pixels = cv2.resize(tile, (14, 14), interpolation=cv2.INTER_AREA)
        return np.concatenate((gradients, pixels.ravel().astype(np.float32) / 255))


class DigitModel:
    """Run a small feed-forward classifier using NumPy."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as weights:
            self.layers = [(weights[f"w{i}"], weights[f"b{i}"]) for i in range(3)]
        self.features = GlyphFeatures()

    def probabilities(self, images: list[GrayImage]) -> Features:
        """Return a row of 11 class scores per glyph, including unknown."""
        activations = np.stack([self.features.extract(image) for image in images])
        for weights, bias in self.layers[:-1]:
            activations = np.maximum(activations @ weights + bias, 0)
        weights, bias = self.layers[-1]
        logits = activations @ weights + bias
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)


def read_digit(scores: Features) -> str:
    """Return a digit only when the model score and winning margin pass."""
    runner_up, best = np.argsort(scores)[-2:]
    if (
        best == UNKNOWN_CLASS
        or scores[best] < MIN_CONFIDENCE
        or scores[best] - scores[runner_up] < MIN_MARGIN
    ):
        return "?"
    return str(best)


def find_glyphs(contrast: GrayImage, gray: GrayImage, polarity: int) -> list[Glyph]:
    """Binarize a local contrast image and keep plausible digit components."""
    height, width = contrast.shape
    threshold, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, mask = cv2.threshold(contrast, max(17, threshold * 0.82), 255, cv2.THRESH_BINARY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    glyphs = []
    for label, (x, y, box_width, box_height, area) in enumerate(stats.tolist()[1:], 1):
        if (
            x < 2 or y < 2 or x + box_width > width - 2 or y + box_height > height - 2
            or box_height < max(13, height * 0.08)
            or box_width < 3
            or not 0.065 <= box_width / box_height <= 1.30
            or area < max(24, box_width * box_height * 0.10)
        ):
            continue
        region = np.s_[y : y + box_height, x : x + box_width]
        ink = labels[region] == label
        intensity = contrast[region][ink]
        strength = float(np.percentile(intensity, 90))
        # A morphological residual can also highlight the background in a loop
        # or between strokes. Check that its ink differs from the nearby paper.
        padding = max(8, round(box_height * 0.35))
        nearby = gray[
            max(0, y - padding) : y + box_height + padding,
            max(0, x - padding) : x + box_width + padding,
        ]
        local_contrast = polarity * (float(np.median(gray[region][ink])) - float(np.median(nearby)))
        if strength < 23 or local_contrast < max(12, 0.22 * strength):
            continue
        crop = np.zeros_like(contrast[region])
        crop[ink] = np.clip(intensity * (255 / strength), 32, 255).astype(np.uint8)
        glyphs.append(Glyph(x, y, box_width, box_height, min(strength, local_contrast), crop))

    return sorted(glyphs, key=lambda glyph: glyph.height, reverse=True)[:48]


def best_row(glyphs: list[Glyph]) -> tuple[float, list[Glyph]]:
    """Pick a row of similarly sized glyphs with a common vertical alignment."""
    best_score = 0.0
    row: list[Glyph] = []
    for anchor in glyphs:
        aligned = [
            glyph for glyph in glyphs
            if 0.55 * anchor.height <= glyph.height <= 1.48 * anchor.height
            and abs(glyph.center_y - anchor.center_y)
            <= 0.36 * anchor.height + 0.18 * abs(glyph.x - anchor.x)
        ]
        score = sum(glyph.height * glyph.contrast for glyph in aligned)
        if score > best_score:
            best_score, row = score, aligned
    return best_score, sorted(row, key=lambda glyph: glyph.x)[:MAX_DIGITS]


def recognize(gray: GrayImage, model: DigitModel) -> Result:
    """Process one image and return an ordered row of glyphs and boxes."""
    height, width = gray.shape
    if min(height, width) < 24:
        return {"sequence": "", "digits": []}

    # Morphological closing estimates light paper through dark strokes.
    # Opening provides the opposite polarity for light strokes on dark paper.
    size = min(91, max(17, round(min(height, width) * 0.23) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    light = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel)
    dark_row = best_row(find_glyphs(dark, smooth, -1))
    light_row = best_row(find_glyphs(light, smooth, 1))
    _, glyphs = max(dark_row, light_row, key=lambda row: row[0])
    if not glyphs:
        return {"sequence": "", "digits": []}

    digits: list[Detection] = []
    for glyph, scores in zip(glyphs, model.probabilities([glyph.image for glyph in glyphs]), strict=True):
        digits.append({
            "x": glyph.x / width,
            "y": glyph.y / height,
            "width": glyph.width / width,
            "height": glyph.height / height,
            "digit": read_digit(scores),
        })
    return {"sequence": "".join(glyph["digit"] for glyph in digits), "digits": digits}
