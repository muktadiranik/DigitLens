"""Train and export a small MLP without a training-framework runtime dependency."""

from pathlib import Path

import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from digitlens.recognition import UNKNOWN_CLASS, GlyphFeatures
from digitlens.training_data import TRAIN_FONTS, augment, load_fonts, mnist_path, render_glyph

MODEL_PATH = Path(__file__).resolve().parent / "model.npz"
SEED = 2026


def main() -> None:
    """Train on MNIST's train split and on augmented synthetic glyphs."""
    rng = np.random.default_rng(SEED)
    cv2.setNumThreads(1)
    extractor = GlyphFeatures()
    fonts = load_fonts(TRAIN_FONTS)
    with np.load(mnist_path(), allow_pickle=False) as data:
        images, labels = data["x_train"], data["y_train"]
    vectors = []
    targets = []
    print("Preparing MNIST handwriting", flush=True)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        vectors.append(extractor.extract(augment(image, rng)))
        targets.append(label)
        if index % 2 == 0:
            vectors.append(extractor.extract(image))
            targets.append(label)

    print("Rendering synthetic print and non-digit shapes", flush=True)
    for font in fonts:
        for character in "0123456789AEFKMNTUVWXYHCDPR&@+#":
            tile = render_glyph(character, font)
            is_digit = character.isdigit()
            label = int(character) if is_digit else UNKNOWN_CLASS
            for _ in range(100 if is_digit else 17):
                vectors.append(extractor.extract(augment(tile, rng)))
                targets.append(label)

    print(f"Training on {len(targets):,} examples", flush=True)
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64), batch_size=256, max_iter=32,
        random_state=SEED, learning_rate_init=0.002, tol=0.00015, verbose=True,
    )
    with threadpool_limits(limits=2):
        classifier.fit(np.asarray(vectors, dtype=np.float32), np.asarray(targets))

    weights = {}
    for index, (weight, bias) in enumerate(zip(classifier.coefs_, classifier.intercepts_, strict=True)):
        weights[f"w{index}"] = weight.astype(np.float32)
        weights[f"b{index}"] = bias.astype(np.float32)
    temporary = MODEL_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **weights)
    temporary.replace(MODEL_PATH)
    print(f"Saved {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
