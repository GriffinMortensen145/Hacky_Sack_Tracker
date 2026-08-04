# Hacky Sack Safari Detector

## 1. Export the trained model

Change `train-4` in `export.py` to your actual run folder, then run:

```bash
python export.py
```

## 2. Add the model

Copy the exported ONNX file into:

```text
hackysack_web/models/best.onnx
```

## 3. Test on your computer

```bash
cd hackysack_web
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## 4. Publish with GitHub Pages

1. Create a GitHub repository.
2. Upload every file inside `hackysack_web`.
3. Open repository Settings.
4. Open Pages.
5. Choose Deploy from a branch.
6. Choose `main` and `/ (root)`.
7. Open the HTTPS Pages address in Safari.
8. Press Start camera and allow camera access.

Notes:
- The website expects `best.onnx` exported at 320 x 320.
- It expects one class named `hackysack`.
- The touch counter is a basic movement estimate; the detector boxes are more reliable.
