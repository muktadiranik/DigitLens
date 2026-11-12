"""Serve the local UI and its memory-only recognition endpoint."""

import argparse
from io import BytesIO
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image, ImageOps
from werkzeug.exceptions import RequestEntityTooLarge

from digitlens.recognition import DigitModel, GrayImage, recognize

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.npz"
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 32_000_000
MAX_SIDE = 1152
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")


def decode_image(payload: bytes) -> GrayImage:
    """Validate image format and dimensions, orient it and convert to grayscale."""
    try:
        with Image.open(BytesIO(payload), formats=IMAGE_FORMATS) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Choose an image smaller than 32 megapixels.")
            if getattr(source, "is_animated", False):
                raise ValueError("Choose a still JPEG, PNG, or WebP image.")
            image = ImageOps.exif_transpose(source)
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                image = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba)
            return np.asarray(image.convert("L"))
    except Image.DecompressionBombError as error:
        raise ValueError(
            "Choose an image smaller than 32 megapixels.") from error
    except (OSError, SyntaxError) as error:
        raise ValueError(
            "Could not read the image. Choose a valid JPEG, PNG, or WebP.") from error


def create_app(model: DigitModel) -> Flask:
    """Serve browser assets and run at most one inference at a time."""
    app = Flask(__name__, static_folder=ROOT /
                "web", static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    inference_lock = Lock()

    @app.get("/")
    def index() -> Response:
        return app.send_static_file("index.html")

    @app.errorhandler(RequestEntityTooLarge)
    def image_too_large(_error: RequestEntityTooLarge) -> tuple[Response, int]:
        return jsonify(error="Choose an image smaller than 12 MB."), 413

    @app.post("/api/recognize")
    def recognize_image() -> tuple[Response, int]:
        if request.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
            return jsonify(error="Choose a JPEG, PNG, or WebP image."), 415
        payload = request.get_data(cache=False)
        if not inference_lock.acquire(blocking=False):
            return jsonify(error="The recognition engine is busy. Try again."), 429
        try:
            image = decode_image(payload)
            return jsonify(recognize(image, model)), 200
        except ValueError as error:
            return jsonify(error=str(error)), 400
        finally:
            inference_lock.release()

    @app.after_request
    def prevent_api_caching(response: Response) -> Response:
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def main() -> None:
    """Run the app on loopback for access from a local browser."""
    parser = argparse.ArgumentParser(description="Run DigitLens on localhost.")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        parser.exit(
            1, "Model weights are missing. Run python -m digitlens.train first.\n")
    cv2.setNumThreads(1)
    create_app(DigitModel(MODEL_PATH)).run(
        host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
