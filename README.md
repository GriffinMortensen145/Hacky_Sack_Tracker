# Hacky Sack Website

This folder contains only the website files.

## Add your model

Put your exported ONNX model here:

```text
models/best.onnx
```

The current JavaScript expects:

- Model input size: 320 × 320
- One class: `hackysack`
- Standard Ultralytics YOLOv8 ONNX output

## Test locally

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## GitHub Pages

Upload the contents of this folder to your repository root, enable GitHub Pages,
then open the HTTPS Pages URL directly in Safari.

Camera access requires HTTPS on the phone.
