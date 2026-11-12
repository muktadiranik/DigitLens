"""Smoke-test the running English text endpoint with print and handwriting."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from digitlens.english import TextResult


def printed_sample(tilted: bool = False) -> bytes:
    """Render two synthetic lines with mixed case, punctuation and digits."""
    width, height = 820, 240
    paper = np.repeat(np.linspace(245, 207, width, dtype=np.uint8)[None, :], height, axis=0)
    cv2.putText(paper, "The quick brown fox.", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 22, 3, cv2.LINE_AA)
    cv2.putText(paper, "Room 007 is open.", (45, 166), cv2.FONT_HERSHEY_COMPLEX, 1.4, 28, 2, cv2.LINE_AA)
    if tilted:
        transform = cv2.getRotationMatrix2D((width / 2, height / 2), -5, 1)
        paper = cv2.warpAffine(paper, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)
        paper = cv2.GaussianBlur(paper, (5, 5), 0.8)
    return cv2.imencode(".png", paper)[1].tobytes()


def scan(url: str, payload: bytes, content_type: str, handwritten: bool = False) -> TextResult:
    """Submit actual image bytes through the same API used by the browser."""
    route = "recognize-handwriting" if handwritten else "recognize-text"
    request = Request(
        f"{url.rstrip('/')}/api/{route}",
        data=payload,
        headers={"Content-Type": content_type},
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def main() -> None:
    """Check multiline print, blank clearing and any supplied handwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:9000")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--handwriting", nargs=2, action="append", default=[], metavar=("IMAGE", "EXPECTED"))
    args = parser.parse_args()
    expected_print = "The quick brown fox.\nRoom 007 is open."
    blank = cv2.imencode(".png", np.full((240, 820), 232, dtype=np.uint8))[1].tobytes()
    samples = [
        ("synthetic-english-print.png", printed_sample(), expected_print, False),
        ("synthetic-english-tilted.png", printed_sample(tilted=True), expected_print, False),
        ("synthetic-english-blank.png", blank, "", False),
    ]
    if args.fixtures:
        args.fixtures.mkdir(parents=True, exist_ok=True)
        for name, payload, _, _ in samples:
            (args.fixtures / name).write_bytes(payload)
    samples.append(("synthetic-handwriting-blank.png", blank, "", True))
    samples.extend((path, Path(path).read_bytes(), expected, True) for path, expected in args.handwriting)

    failures = []
    for name, payload, expected, handwritten in samples:
        suffix = Path(name).suffix.lower()
        content_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}[suffix[1:]]
        result = scan(args.url, payload, content_type, handwritten)
        print(f"{name}: {result['text']!r}, {len(result['regions'])} regions", flush=True)
        if result["text"] != expected or bool(result["regions"]) != bool(expected):
            failures.append(name)
    if failures:
        raise SystemExit(f"Transcription differed for {', '.join(failures)}")


if __name__ == "__main__":
    main()
