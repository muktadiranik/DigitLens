"""Evaluate the full pipeline and generate browser smoke-test fixtures."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from digitlens.recognition import DigitModel, GrayImage, Result, recognize
from digitlens.server import MODEL_PATH
from digitlens.training_data import TEST_FONTS, augment, load_fonts, mnist_path, render_glyph


def scene(
    glyphs: list[GrayImage], *, angle: float = 0, blur: float = 0, inverted: bool = False,
) -> GrayImage:
    """Compose separated glyphs on synthetic paper with an illumination gradient."""
    height, width = 216, max(320, len(glyphs) * 86 + 50)
    ink = np.zeros((height, width), dtype=np.uint8)
    left = (width - len(glyphs) * 86) // 2
    for index, glyph in enumerate(glyphs):
        x = left + index * 86 + 3
        ink[68:148, x : x + 80] = cv2.resize(glyph, (80, 80))
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    ink = cv2.warpAffine(ink, transform, (width, height))
    light = 241 - np.linspace(0, 50, width) - np.linspace(0, 14, height)[:, None]
    image = (light * (1 - 0.94 * ink / 255)).astype(np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return 255 - image if inverted else image


def tally(name: str, model: DigitModel, samples: Iterable[tuple[GrayImage, str]]) -> None:
    """Report correct, uncertain and remaining errors independently."""
    correct = uncertain = total = 0
    for image, expected in samples:
        found = recognize(image, model)["sequence"]
        correct += found == expected
        uncertain += found == "?"
        total += 1
    print(f"{name}: {correct}/{total} correct ({correct / total:.2%}), "
          f"{uncertain} uncertain, {total - correct - uncertain} other errors", flush=True)


def remote_recognize(url: str, image: GrayImage) -> Result:
    """Submit real PNG bytes to the same endpoint as the browser."""
    _, payload = cv2.imencode(".png", image)
    request = Request(
        f"{url.rstrip('/')}/api/recognize",
        data=payload.tobytes(),
        headers={"Content-Type": "image/png"},
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def main() -> None:
    """Run held-out handwritten and synthetic print checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Also check an already running server, e.g. http://localhost:8000.")
    parser.add_argument("--fixtures", type=Path, help="Write the named smoke-test images to this directory.")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    model = DigitModel(MODEL_PATH)
    rng = np.random.default_rng(612)
    fonts = load_fonts(TEST_FONTS)
    with np.load(mnist_path(), allow_pickle=False) as data:
        images, labels = data["x_test"], data["y_test"]

    hand_index = int(np.flatnonzero(labels == 8)[0])
    zero_zero_seven = [render_glyph(character, fonts[0]) for character in "007"]
    cases = [
        ("handwritten-8", scene([images[hand_index]], angle=-7, blur=0.8), "8"),
        ("synthetic-printed-007", scene(zero_zero_seven, angle=8, blur=1.0), "007"),
        ("synthetic-inverted-007", scene(zero_zero_seven, angle=-6, inverted=True), "007"),
        ("synthetic-blank", scene([]), ""),
        ("synthetic-uncertain", scene([render_glyph("X", fonts[0])]), "?"),
    ]
    failures = []
    if args.fixtures:
        args.fixtures.mkdir(parents=True, exist_ok=True)
    for name, image, expected in cases:
        if args.fixtures:
            cv2.imwrite(str(args.fixtures / f"{name}.png"), image)
        found = recognize(image, model)
        print(f"{name}: {found['sequence']!r} (expected {expected!r}), {len(found['digits'])} boxes", flush=True)
        if found["sequence"] != expected or len(found["digits"]) != len(expected):
            failures.append(name)
        if args.url and remote_recognize(args.url, image) != found:
            failures.append(f"{name} via HTTP")

    if args.url:
        invalid = Request(
            f"{args.url.rstrip('/')}/api/recognize", data=b"invalid", headers={"Content-Type": "image/png"},
        )
        try:
            urlopen(invalid, timeout=5).close()
        except HTTPError as error:
            if error.code != 400:
                failures.append("invalid upload HTTP status")
        else:
            failures.append("invalid upload accepted")

    # Generate frames lazily to avoid retaining thousands of full-sized images.
    handwriting = (
        (scene([image], angle=float(rng.uniform(-9, 9)), blur=0.75), str(label))
        for image, label in zip(images, labels, strict=True)
    )
    clean_print = (
        (scene([render_glyph(digit, font)], inverted=inverted), digit)
        for font in fonts for digit in "0123456789" for inverted in (False, True)
    )
    degraded_print = (
        (scene([augment(render_glyph(digit, font), rng)], angle=float(rng.uniform(-5, 5)), blur=1.1), digit)
        for font in fonts for digit in "0123456789" for _ in range(12)
    )
    tally("Handwriting (MNIST test split in simulated camera frames)", model, handwriting)
    tally("Print (synthetic, clean glyphs in both polarities)", model, clean_print)
    tally("Print (synthetic, rotated, blurred and stroke-varied)", model, degraded_print)
    if failures:
        raise SystemExit(f"Failed checks: {', '.join(failures)}")
    print("Pipeline smoke checks passed" + (" over HTTP as well." if args.url else "."), flush=True)


if __name__ == "__main__":
    main()
