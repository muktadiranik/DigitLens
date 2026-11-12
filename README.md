# DigitLens

DigitLens reads digits, printed English, and handwritten English from a Windows browser camera or an uploaded image. Choose the mode before scanning. **Digits** preserves leading zeros and marks uncertain digits with `?`. Both English modes preserve line order and outline uncertain lines in amber.

## Run in Ubuntu on WSL 2

Open your Ubuntu terminal. If you need uv, install it with the official standalone installer. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From Ubuntu, run the following. The virtual environment lives in the Linux home directory and uses Python 3.12, even if Ubuntu's default Python is newer.

```bash
cd /mnt/d/Python/DigitLens
mkdir -p ~/.venvs
~/.local/bin/uv venv --python 3.12 ~/.venvs/digitlens
~/.local/bin/uv pip install --python ~/.venvs/digitlens/bin/python -r requirements.txt
source ~/.venvs/digitlens/bin/activate
python -m digitlens.text_models
python -m digitlens.server
```

Open [http://localhost:9000](http://localhost:9000) in Chrome or Edge on Windows. Leave the Ubuntu terminal running while you use the app. Use `Ctrl+C` to stop the server. WSL 2 forwards its loopback listener to Windows. [Microsoft WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking#accessing-linux-networking-apps-from-windows-localhost)

For an existing installation, stop the server, rerun the dependency installation and `python -m digitlens.text_models`, then restart. For later launches, use `cd /mnt/d/Python/DigitLens` and run `~/.venvs/digitlens/bin/python -m digitlens.server`. If port 9000 is in use, add `--port 9001` and open [http://localhost:9001](http://localhost:9001).

## Scan

1. Select **Digits**, **Printed English**, or **Handwritten English**. Click **Start Camera** and allow access for localhost. Center one row of digits or a short block of horizontal English text in the scan box. Use **Stop Camera** to release the webcam.
2. Use **Upload Image** for a JPEG, PNG, or WebP still. The app scans the whole uploaded image. Keep it under 12 MB and 32 megapixels. Changing the mode scans the current image again.
3. If the browser denies access, allow the site to use the camera and check Windows **Settings → Privacy & security → Camera**. If the camera is busy, close other apps using it and try again.

The browser sends only the area inside the scan box, at up to four frames per second for digits and about two for English. It waits for each response before sending another. A changed label in the same position needs two agreeing frames. An empty frame clears the result on the next response. Camera frames and uploaded images are processed in memory on localhost.

## Model and training data

The included `digitlens/model.npz` contains about 278 KiB of weights and runs on the CPU. OpenCV estimates local contrast, extracts connected components, and normalizes the selected row's glyphs. A two-hidden-layer scikit-learn MLP trained on HOG and grayscale features is exported to NumPy for inference. A separate non-digit class and score thresholds provide uncertainty handling.

- **Handwriting.** Training uses the 60,000-image MNIST train split, with augmentation. Verification uses the separate 10,000-image test split. The download comes from the Keras mirror and is checked by SHA-256. MNIST is by Yann LeCun and Corinna Cortes, derived from NIST data. [Keras dataset source and license](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py)
- **Print.** The trainer renders synthetic digits from 13 face files across 12 Google Fonts families, including serif, sans-serif, condensed, italic, and monospaced styles. It varies stroke weight, blur, rotation, and shear. The exact font revision, paths, and hashes are recorded in [training_data.py](digitlens/training_data.py). The evaluation uses Crimson Text, Fira Code, and Space Grotesk, which are held out from model training. [Google Fonts font files and licenses](https://github.com/google/fonts/tree/c30b221beb0c1da59d563feb91ec8656a1fa972a)

To rebuild the included model, stop the running server, then run these commands in the project directory. The first training run downloads the checked data into `.cache/training`.

```bash
source ~/.venvs/digitlens/bin/activate
~/.local/bin/uv pip install --python ~/.venvs/digitlens/bin/python -r requirements-train.txt
python -m digitlens.train
```

Training uses a fixed seed and CPU settings. Exact weight bytes can vary across platforms and numerical library builds. Launch the server again to load the rebuilt weights.

### English OCR models

`python -m digitlens.text_models` downloads about 81 MB of models and dictionaries into `.cache/ocr`. The installer pins each upstream revision and verifies SHA-256. Later scans use the cached models and CPU-only ONNX Runtime sessions.

- Both English modes use PaddleOCR's PP-OCRv5 mobile detector. OpenCV groups detected regions into lines and corrects modest tilt. [PaddleOCR detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx)
- Printed English uses PaddleOCR's English PP-OCRv5 mobile recognizer. [PaddleOCR English recognizer](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec)
- Handwritten English uses Microsoft's TrOCR Small model, fine-tuned on IAM handwriting, with Xenova's quantized ONNX export. [Microsoft handwriting model](https://huggingface.co/microsoft/trocr-small-handwritten) · [Quantized ONNX export](https://huggingface.co/Xenova/trocr-small-handwritten)

## Verification

With the server running, open a second Ubuntu terminal and run the following from the project directory after installing the training dependencies. The command evaluates the full detector and classifier, exercises the HTTP endpoint with real image bytes, and writes smoke-test images for browser uploads.

```bash
cd /mnt/d/Python/DigitLens
source ~/.venvs/digitlens/bin/activate
python -m digitlens.verify --url http://localhost:9000 --fixtures .cache/checks
python -m digitlens.verify_text --url http://localhost:9000 --fixtures .cache/checks
```

For a handwriting check, add `--handwriting path/to/line.png "Expected transcription"` to the `verify_text` command. Set `--url http://localhost:9001` for either check if you launched on port 9001.

Results from Ubuntu on WSL 2 using the included weights

| Input | Correct | Uncertain | Other errors |
| --- | ---: | ---: | ---: |
| MNIST test handwriting composited into simulated camera frames | 9,633 / 10,000 (96.33%) | 60 | 307 |
| Synthetic print, clean glyphs on light and dark backgrounds | 100 / 100 (100%) | 0 | 0 |
| Synthetic print, combined rotation, blur, shear, and stroke changes | 522 / 600 (87.00%) | 8 | 70 |

The smoke checks read a handwritten MNIST `8`, a synthetic printed `007` in both polarities, and an uncertain synthetic glyph. The synthetic blank produced zero boxes. Windows Chrome uploads exercised the live endpoint and displayed the corresponding result and boxes.

An A4TECH USB webcam delivered 640×480 video to Windows Chrome. A frame reached the WSL inference endpoint, and **Stop Camera** ended the track. A separate synthetic browser stream verified scan-box cropping, stable `007`, and immediate clearing when the scan box became empty. The paper check below covers recognition with physical samples.

### English results

| Input | Result |
| --- | --- |
| Synthetic printed English | Both two-line samples transcribed exactly, including one with 5° tilt, mild blur, and uneven lighting |
| Real IAM handwriting | 4 of 10 sample lines transcribed exactly after normalizing annotation spacing. The other six had transcription errors |
| Synthetic blank | Both English modes returned no text and zero regions |

The handwriting check used IAM test rows 0–4 and 406–410 and took about 0.33–0.50 seconds per line on the test CPU. This small sample measures the app's behavior on those lines. [IAM dataset mirror](https://huggingface.co/datasets/anris05/IAM-line)

Windows Chrome exercised real uploads in all three modes. Simulated webcam video verified English mode switching, live recognition, and clearing on a blank frame. The physical webcam sent a frame to the handwriting endpoint, and **Stop Camera** ended its track.

## Limits

- Scan one row with up to 16 separated digits. Touching strokes, multiple rows, characters clipped by the scan box, and small or broken strokes can lead to omissions or extra boxes.
- Use a plain background with strong contrast. The detector handles light gradients, mild blur, and small tilts. Heavy blur, glare, shadows with hard edges, and textured backgrounds can cause errors.
- Some shapes, such as a slanted `1` and a `7`, are ambiguous. Check the displayed result when accuracy matters.
- English works best on a short, single-column block of horizontal text. Handwriting mode processes up to six lines per image. Dense pages, faint writing, heavy cursive, and unusual names can produce omissions or substitutions. Review important text against the source.

## Check with paper

1. Start the camera and hold a handwritten `8` inside the scan box. Confirm one box and `8`.
2. Hold up a printed `007` with clear gaps. Confirm three boxes and `007` after the result settles.
3. Remove the paper against a plain background. Confirm **No digits detected**, then click **Stop Camera** and check that the browser's camera-use indicator clears.
4. Repeat in each English mode with a short printed line and a short handwritten line. Check the words and punctuation against the originals.
