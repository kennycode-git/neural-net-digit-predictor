# Digit-Net Visual Lab

> **[Live demo placeholder]** · [Model Card](#model-card)

<!-- GIF placeholder: draw → predict → heatmap comparison -->
<!-- ![demo](docs/demo.gif) -->

**Browser-native MNIST digit classification with explainability.**

- **Browser inference** — model runs in-browser via [onnxruntime-web](https://github.com/microsoft/onnxruntime); no server round-trip for predictions
- **Explainability comparison** — side-by-side saliency/GradCAM heatmaps for the predicted class vs. runner-up class
- **Calibration + edge-case evaluation** — reliability diagram (ECE), confusion matrix, and a curated failure gallery

---

## Quickstart

### 1. Train the model

```bash
cd ml
pip install -r requirements.txt
python train.py             # → output/model.pt + metrics.json
python export_onnx.py       # → output/model.onnx (copied to frontend/public/artifacts/)
python evaluate.py          # → confusion_matrix.png + misclassified.json
python calibration.py       # → reliability.json
python edge_cases.py        # → edge_cases.json
```

### 2. Run the frontend

```bash
cd frontend
npm install
npm run dev                 # → http://localhost:3000
```

Draw a digit in the browser — inference runs fully client-side.

### 3. Run the backend (optional — for heatmaps)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Once running, click **Explain** on the main page to see saliency or GradCAM heatmaps.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Browser (Next.js)                          │
│                                             │
│  DigitCanvas → preprocess.ts → infer_onnx   │
│                    │               │        │
│               28×28 preview    ONNX model   │
│                            (onnxruntime-web)│
│                                             │
│  ExplainCompare ──────────── optional call  │
└──────────────────────┬──────────────────────┘
                       │ POST /v1/explain
              ┌────────▼────────┐
              │  FastAPI        │
              │  Captum         │
              │  (saliency /    │
              │   GradCAM)      │
              └─────────────────┘

ML pipeline (offline):
  train.py → model.pt → export_onnx.py → model.onnx
  evaluate.py + calibration.py + edge_cases.py → artifacts/
```

---

## Model Card

| Metric | Value |
|---|---|
| Architecture | 2-conv CNN (conv→relu→pool ×2 → fc ×2) |
| Dataset | MNIST (60K train / 10K test) |
| Parameters | ~420K |
| Best val accuracy | see `/model` page after training |
| ONNX opset | 17 |
| Preprocessing version | `1` |

See the in-app [Model Card page](/model) for:
- Training curves (loss + accuracy per epoch)
- Confusion matrix
- Reliability diagram + ECE
- High-confidence mistake gallery
- Edge case gallery (rotations, noise, erosion, ambiguous pairs)

---

## Preprocessing Contract v1

Both `ml/preprocess.py` and `frontend/lib/preprocess.ts` implement identically:

1. Grayscale (`0.299R + 0.587G + 0.114B`)
2. Crop bounding box of drawn pixels + 2px padding
3. Resize 20×20
4. Pad to 28×28 centered
5. Normalize `[0, 1]`
6. Invert (white bg → 0, black stroke → 1)
7. `(x − 0.1307) / 0.3081`

---

## Artifact Storage (Git LFS)

Large binaries (`*.onnx`, `*.pt`, artifact `*.png`) are tracked via **Git LFS**.

```bash
git lfs install
git lfs ls-files   # verify tracking
```

---

## Running Tests

```bash
cd ml
pytest tests/ -v
# test_preprocess: shape/dtype checks (always runs)
# test_onnx_parity: PyTorch vs ONNX logits within 1e-4 (requires model.onnx)
```

---

## Limitations

- Performance degrades for digits written in non-MNIST style (unusual sizing, strokes)
- Not robust to rotations > 30° or heavy noise
- Model can be overconfident on ambiguous inputs (see reliability diagram)

**Future work:** temperature scaling, quantized ONNX, data augmentation, embedding support.

---

## CI

GitHub Actions: `ruff` lint + `pytest` on every push. LFS artifacts are fetched in CI; the ONNX parity test auto-skips if `model.onnx` is not present.
