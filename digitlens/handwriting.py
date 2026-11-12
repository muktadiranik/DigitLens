"""Transcribe English handwriting with a quantized TrOCR model."""

import re

import cv2
import numpy as np

from digitlens.english import cpu_session
from digitlens.recognition import GrayImage
from digitlens.text_models import HANDWRITING_DECODER, HANDWRITING_ENCODER, HANDWRITING_TOKENIZER

# Image size and decoder tokens from the pinned TrOCR model configuration.
IMAGE_SIZE = 384
START_TOKEN = END_TOKEN = 2
MAX_TOKENS = 96
MAX_HANDWRITTEN_LINES = 6


class HandwritingOCR:
    """Run TrOCR's small vision encoder and cached text decoder on the CPU."""

    def __init__(self) -> None:
        for path in (HANDWRITING_ENCODER, HANDWRITING_DECODER, HANDWRITING_TOKENIZER):
            if not path.is_file():
                raise FileNotFoundError("Run python -m digitlens.text_models to enable English handwriting.")

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(HANDWRITING_TOKENIZER))
        self.encoder = cpu_session(HANDWRITING_ENCODER)
        self.decoder = cpu_session(HANDWRITING_DECODER)
        self.empty_cache = {
            item.name: np.empty((1, item.shape[1], 0, item.shape[3]), dtype=np.float32)
            for item in self.decoder.get_inputs()
            if item.name.startswith("past_key_values.")
        }
        self.cache_outputs = {
            item.name: item.name.replace("present.", "past_key_values.", 1)
            for item in self.decoder.get_outputs()
            if item.name.startswith("present.")
        }

    def read_line(self, crop: GrayImage) -> tuple[str, bool]:
        """Return a line of decoded text and an approximate uncertainty flag."""
        pixels = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        pixels = pixels.astype(np.float32) / 127.5 - 1
        tensor = np.repeat(pixels[None, None], 3, axis=1)
        hidden = self.encoder.run(["last_hidden_state"], {"pixel_values": tensor})[0]

        inputs = {"encoder_hidden_states": hidden, **self.empty_cache}
        tokens = []
        confidence = []
        token = START_TOKEN
        for step in range(MAX_TOKENS):
            inputs["input_ids"] = np.array([[token]], dtype=np.int64)
            inputs["use_cache_branch"] = np.array([step > 0])
            outputs = self.decoder.run(["logits", *self.cache_outputs], inputs)
            logits = outputs[0][0, -1]
            token = int(np.argmax(logits))
            if token == END_TOKEN:
                break
            tokens.append(token)
            confidence.append(float(1 / np.exp(logits - logits[token]).sum()))
            for key, cache in zip(self.cache_outputs.values(), outputs[1:], strict=True):
                # After the first step the cross-attention cache stays constant.
                if cache.size:
                    inputs[key] = cache

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        # The model's handwriting annotations put spaces around punctuation.
        text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        uncertain = len(tokens) == MAX_TOKENS or (bool(confidence) and min(confidence) < 0.45)
        return text, uncertain
