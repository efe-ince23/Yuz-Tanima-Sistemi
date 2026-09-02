# Face models

The runtime uses two ONNX models when `FACE_ENGINE=yolo_arcface`:

- `yolov8n-face.onnx`: YOLOv8-Face detector with five facial landmarks.
  Source: https://github.com/clibdev/yolov8-face/releases/latest
  SHA-256: `af09683be4937ebe23e9da0346cbf46cd937b73ed5fac986e6205e6c7cdd2c25`
- `arcface-r50-w600k.onnx`: InsightFace R50 recognition model trained on
  WebFace600K. The input supports dynamic batches of aligned 112x112 faces.
  Source: https://github.com/deepinsight/insightface/releases/tag/v0.7
  SHA-256: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`

The YOLOv8-Face repository is GPL-3.0 licensed. InsightFace pretrained models
are provided for non-commercial research use; obtain the appropriate license
before commercial deployment.
