"""Pinned public data sources and reproducible synthetic print generation."""

from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from digitlens.downloads import fetch
from digitlens.recognition import GrayImage

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "training"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
FONT_REVISION = "c30b221beb0c1da59d563feb91ec8656a1fa972a"
FONT_ROOT = f"https://raw.githubusercontent.com/google/fonts/{FONT_REVISION}"
TRAIN_FONTS = (
    ("ofl/arimo/Arimo[wght].ttf", "e43898b143ec826ac8cb4034816458a7047fbe0836558de2a1f8c6223ae3e0ca"),
    ("ofl/bebasneue/BebasNeue-Regular.ttf", "08e4623805102d819f58601e46e345648846075e363b2ceb23313c2d1c83ec73"),
    ("ofl/firasans/FiraSans-Regular.ttf", "c29556a2719bf613ef3d5e070e40d903a8965d9c081beca1375dc1e6e0f93c23"),
    ("ofl/ibmplexmono/IBMPlexMono-Regular.ttf", "6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46"),
    ("ofl/instrumentserif/InstrumentSerif-Regular.ttf", "498efd461f6ddfcb7a111bf9a565709d2085d48201d501ead960d93e84ffbb88"),
    ("ofl/lato/Lato-Regular.ttf", "d636e4683231f931eda222d588e944d082bfd3bdba02f928bee461c0f185b251"),
    ("ofl/lato/Lato-BoldItalic.ttf", "62c1b7f0d2e74b45960154c3520efc337b553db0961bfdc950d5618334596cc8"),
    ("ofl/notosans/NotoSans[wdth,wght].ttf", "bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d"),
    ("ofl/notoserif/NotoSerif[wdth,wght].ttf", "4d8e6761424656867019081a1a01336f3cb086982682698714054fc33f782713"),
    ("ofl/oswald/Oswald[wght].ttf", "5b38c246e255a12f5712d640d56bcced0472466fc68983d2d0410ec0457c2817"),
    ("ofl/roboto/Roboto[wdth,wght].ttf", "d7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134"),
    ("ofl/robotomono/RobotoMono[wght].ttf", "66a80e79d17e4c7cabd162e2916578a4cc08fd19eef6e2a643305eae9c567b2b"),
    ("ofl/rubik/Rubik[wght].ttf", "1b3a7437ba2af80e465e773ed60c5036d1ba6ace492d89046dbcf18fb31e4e88"),
)
TEST_FONTS = (
    ("ofl/crimsontext/CrimsonText-Regular.ttf", "48e6c5d5ad1d01599d374ecb817e15890d1feb3b8a3a88e527d44c90389e1f06"),
    ("ofl/firacode/FiraCode[wght].ttf", "9335b082b3c7850d98a64b584f3417f65355f3471278bb5eeb8c6c0e8657aeeb"),
    ("ofl/spacegrotesk/SpaceGrotesk[wght].ttf", "acad6de1fc93436f5c0f1f4137751ef04f1aea3063e7036535970ffcfbd79f72"),
)


def mnist_path() -> Path:
    """Get the canonical 60,000/10,000 MNIST split from the Keras mirror."""
    return fetch(MNIST_URL, CACHE / "mnist.npz", MNIST_SHA256)


def load_fonts(specs: tuple[tuple[str, str], ...]) -> list[ImageFont.FreeTypeFont]:
    """Load font faces and add heavier instances of variable fonts."""
    fonts = []
    for location, checksum in specs:
        path = fetch(f"{FONT_ROOT}/{quote(location)}", CACHE / "fonts" / Path(location).name, checksum)
        font = ImageFont.truetype(path, 52)
        fonts.append(font)
        try:
            axes = font.get_variation_axes()
        except OSError:
            continue
        heavier = ImageFont.truetype(path, 52)
        heavier.set_variation_by_axes([
            min(axis["maximum"], 700) if axis["name"] == b"Weight" else axis["default"]
            for axis in axes
        ])
        fonts.append(heavier)
    return fonts


def render_glyph(character: str, font: ImageFont.FreeTypeFont) -> GrayImage:
    """Render a centered antialiased font glyph on a dark tile."""
    x0, y0, x1, y1 = font.getbbox(character)
    image = Image.new("L", (72, 72))
    offset = ((72 - x1 + x0) // 2 - x0, (72 - y1 + y0) // 2 - y0)
    ImageDraw.Draw(image).text(offset, character, font=font, fill=255)
    return np.asarray(image)


def augment(image: GrayImage, rng: np.random.Generator) -> GrayImage:
    """Add small rotations, shear, aspect changes, blur and stroke variation."""
    image = cv2.resize(image, (72, 72), interpolation=cv2.INTER_LINEAR)
    transform = cv2.getRotationMatrix2D((36, 36), rng.uniform(-15, 15), rng.uniform(0.79, 1.01))
    stretch = rng.uniform(0.83, 1.17)
    shear = rng.uniform(-0.15, 0.15)
    transform[0, :2] *= stretch
    transform[0, 1] += shear
    transform[0, 2] += 36 * (1 - stretch - shear)
    varied = cv2.warpAffine(image, transform, (72, 72))
    thickness = int(rng.integers(0, 5))
    if thickness == 0:
        varied = cv2.erode(varied, np.ones((3, 3), dtype=np.uint8))
    elif thickness == 1:
        varied = cv2.dilate(varied, np.ones((3, 3), dtype=np.uint8))
    if rng.random() < 0.5:
        varied = cv2.GaussianBlur(varied, (5, 5), rng.uniform(0.6, 1.6))
    if rng.random() < 0.55:
        varied = np.where(varied >= rng.integers(36, 114), varied, 0).astype(np.uint8)
    return varied
