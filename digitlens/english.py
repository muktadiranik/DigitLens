"""Read horizontal English text with PaddleOCR's mobile ONNX models."""

import math
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from digitlens.recognition import GrayImage
from digitlens.text_models import DETECTOR, DICTIONARY, RECOGNIZER

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

Quad = NDArray[np.float32]
DETECTOR_SIDE = 832
RECOGNIZER_HEIGHT = 48
MAX_REGIONS = 32
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TextRegion(TypedDict):
    """A recognized fragment and its normalized bounds and row index."""

    x: float
    y: float
    width: float
    height: float
    text: str
    row: int
    uncertain: bool


class TextResult(TypedDict):
    """The transcription with reading order and overlay positions."""

    text: str
    regions: list[TextRegion]


def cpu_session(path: Path) -> "InferenceSession":
    """Open an ONNX session using a small, CPU-only thread pool."""
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def crop_line(gray: GrayImage, quad: Quad) -> GrayImage:
    """Rectify a slightly tilted region into a horizontal strip."""
    by_x = quad[np.argsort(quad[:, 0])]
    top_left, bottom_left = by_x[:2][np.argsort(by_x[:2, 1])]
    top_right, bottom_right = by_x[2:][np.argsort(by_x[2:, 1])]
    width = max(2, round(np.linalg.norm(top_left - top_right)))
    height = max(2, round(np.linalg.norm(top_left - bottom_left)))
    source = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(gray, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)


def merge_lines(boxes: list[Quad]) -> list[Quad]:
    """Group detector fragments along a common deskewed baseline."""
    if not boxes:
        return []
    angles = []
    for box in boxes:
        edges = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(edges, axis=1)
        if lengths.max() > 2 * lengths.min():
            horizontal = edges[lengths.argmax()]
            angle = math.atan2(horizontal[1], horizontal[0])
            angle = (angle + math.pi / 2) % math.pi - math.pi / 2
            if abs(angle) < math.pi / 6:
                angles.append(angle)
    skew = median(angles) if angles else 0
    cosine, sine = math.cos(skew), math.sin(skew)
    basis = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    aligned = [box @ basis for box in boxes]
    tolerance = median(float(np.ptp(box[:, 1])) for box in aligned) * 0.6
    rows: list[list[Quad]] = []
    centers: list[float] = []
    for box in sorted(aligned, key=lambda item: float(item[:, 1].mean())):
        center = float(box[:, 1].mean())
        if not rows or center - centers[-1] > tolerance:
            rows.append([])
            centers.append(center)
        rows[-1].append(box)

    lines = []
    for row in rows:
        points = np.concatenate(row)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        rect = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
        lines.append(rect @ basis.T)
    return lines


class EnglishOCR:
    """Run line detection and CTC recognition in local CPU sessions."""

    def __init__(self) -> None:
        for path in (DETECTOR, RECOGNIZER, DICTIONARY):
            if not path.is_file():
                raise FileNotFoundError("Run python -m digitlens.text_models to enable English text.")

        import yaml

        config = yaml.safe_load(DICTIONARY.read_text(encoding="utf-8"))
        self.alphabet = ["", *config["PostProcess"]["character_dict"], " "]
        self.detector = cpu_session(DETECTOR)
        self.recognizer = cpu_session(RECOGNIZER)
        if self.recognizer.get_outputs()[0].shape[-1] != len(self.alphabet):
            raise ValueError("English OCR dictionary does not match the recognizer.")

    def detect(self, gray: GrayImage) -> list[Quad]:
        """Decode the DB detector's probability map into text quadrilaterals."""
        height, width = gray.shape
        scale = min(1, DETECTOR_SIDE / max(width, height))
        size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
        pixels = cv2.resize(gray, size).astype(np.float32) / 255
        tensor = np.ascontiguousarray(((pixels[:, :, None] - MEAN) / STD).transpose(2, 0, 1)[None])
        probabilities = self.detector.run(None, {"x": tensor})[0][0, 0]
        mask = (probabilities > 0.30).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        multiplier = np.array([width / mask.shape[1], height / mask.shape[0]], dtype=np.float32)

        boxes = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_REGIONS * 3]:
            center, (box_width, box_height), angle = cv2.minAreaRect(contour)
            if min(box_width, box_height) < 4:
                continue
            x, y, rect_width, rect_height = cv2.boundingRect(contour)
            support = np.zeros((rect_height, rect_width), dtype=np.uint8)
            cv2.drawContours(support, [contour], 0, 255, cv2.FILLED, offset=(-x, -y))
            if cv2.mean(probabilities[y : y + rect_height, x : x + rect_width], support)[0] < 0.50:
                continue
            # DB's contours cover the core of the line. Expand before rectifying.
            padding = box_width * box_height / (box_width + box_height) * 1.4
            quad = cv2.boxPoints((center, (box_width + padding, box_height + padding), angle))
            boxes.append(quad * multiplier)
            if len(boxes) == MAX_REGIONS:
                break
        return boxes

    def read_line(self, crop: GrayImage) -> tuple[str, bool]:
        """Decode CTC text and flag regions containing low-confidence glyphs."""
        height, width = crop.shape
        resized_width = min(1536, max(1, math.ceil(RECOGNIZER_HEIGHT * width / height)))
        pixels = cv2.resize(crop, (resized_width, RECOGNIZER_HEIGHT)).astype(np.float32) / 127.5 - 1
        tensor = np.zeros((1, 3, RECOGNIZER_HEIGHT, max(320, resized_width)), dtype=np.float32)
        tensor[0, :, :, :resized_width] = pixels
        scores = self.recognizer.run(None, {"x": tensor})[0][0]
        labels = scores.argmax(axis=-1)
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        ends = np.r_[starts[1:], len(labels)]
        characters = []
        confidence = []
        for start, end in zip(starts, ends, strict=True):
            label = labels[start]
            if label:
                characters.append(self.alphabet[label])
                if self.alphabet[label] != " ":
                    confidence.append(float(scores[start:end, label].max()))
        uncertain = bool(confidence) and (min(confidence) < 0.50 or float(np.mean(confidence)) < 0.83)
        return "".join(characters).strip(), uncertain

    def recognize(
        self,
        gray: GrayImage,
        read_line: Callable[[GrayImage], tuple[str, bool]] | None = None,
        max_lines: int = MAX_REGIONS,
    ) -> TextResult:
        """Read printed or handwritten English, preserving horizontal reading order."""
        height, width = gray.shape
        if min(width, height) < 16:
            return {"text": "", "regions": []}
        boxes = self.detect(gray)
        if boxes and width >= 7 * height:
            # A tightly cropped line can leave the detector with partial fragments.
            boxes = [np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)]
        recognize_line = read_line or self.read_line
        regions: list[TextRegion] = []
        for quad in merge_lines(boxes)[:max_lines]:
            crop = crop_line(gray, quad)
            if crop.shape[0] > 2 * crop.shape[1]:
                continue
            text, uncertain = recognize_line(crop)
            if not text:
                continue
            left, top = np.maximum(quad.min(axis=0), 0)
            right, bottom = np.minimum(quad.max(axis=0), [width, height])
            regions.append({
                "text": text, "row": len(regions), "uncertain": uncertain,
                "x": float(left / width), "y": float(top / height),
                "width": float((right - left) / width), "height": float((bottom - top) / height),
            })
        return {
            "text": "\n".join(item["text"] for item in regions),
            "regions": regions,
        }
