"""Install the pinned PaddleOCR models for offline English recognition."""

from pathlib import Path

from digitlens.downloads import fetch

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "ocr"
DETECTOR = CACHE / "det.onnx"
RECOGNIZER = CACHE / "rec.onnx"
DICTIONARY = CACHE / "rec.yml"
HANDWRITING_ENCODER = CACHE / "encoder_model_quantized.onnx"
HANDWRITING_DECODER = CACHE / "decoder_model_merged_quantized.onnx"
HANDWRITING_TOKENIZER = CACHE / "tokenizer.json"
DETECTOR_SOURCE = (
    "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx"
    "/resolve/e6f4fa85f00e168c862bc462aebca69eef9b3d3d"
)
RECOGNIZER_SOURCE = (
    "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx"
    "/resolve/3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b"
)
HANDWRITING_SOURCE = (
    "https://huggingface.co/Xenova/trocr-small-handwritten"
    "/resolve/9ef961fec33b56615026fbe77b9da68d6237cddb"
)


def main() -> None:
    """Download and validate the OCR models and their alphabets."""
    fetch(f"{DETECTOR_SOURCE}/inference.onnx", DETECTOR,
          "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d")
    fetch(f"{RECOGNIZER_SOURCE}/inference.onnx", RECOGNIZER,
          "b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f")
    fetch(f"{RECOGNIZER_SOURCE}/inference.yml", DICTIONARY,
          "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067")
    fetch(f"{HANDWRITING_SOURCE}/onnx/{HANDWRITING_ENCODER.name}", HANDWRITING_ENCODER,
          "2f29edbd925f8a49c9c7d1349895f960cf09d2efdc76fe23f957048d476e0d03")
    fetch(f"{HANDWRITING_SOURCE}/onnx/{HANDWRITING_DECODER.name}", HANDWRITING_DECODER,
          "51076aa396ab5939c4668db9de901ad51765094b4c05c4c8c1f8ae2012ba1e08")
    fetch(f"{HANDWRITING_SOURCE}/{HANDWRITING_TOKENIZER.name}", HANDWRITING_TOKENIZER,
          "68bcb5468c854362a615f3d2ff6a5e4091a85f4c8198993ed9a30afe0b143737")
    print("English OCR models are ready.")


if __name__ == "__main__":
    main()
