# YOLO learning app (Python)

This project shows how to run object detection with a pretrained YOLOv8 model and how to fine-tune the model on your own classes.

Prerequisites
- Python 3.8+
- GPU recommended (but CPU works)

Quick setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Dataset format
- Use YOLO format: images in `dataset/images/train` and `dataset/images/val`, labels in `dataset/labels/train` and `dataset/labels/val`.
- Each label file is a `.txt` with lines: `class x_center y_center width height` (normalized 0-1).
- Create a `data.yaml` (see `data.yaml.template`) with `train`, `val`, `nc`, and `names`.

Converting annotations
- If you have Pascal VOC XML annotations, use `utils/convert_pascal_voc_to_yolo.py`.

Training (fine-tune)
```bash
python train_custom.py --data data.yaml --weights yolov8n.pt --epochs 50
```

Detection / inference
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source image.jpg --save
```

Web UI (Streamlit)
```bash
streamlit run app.py
```

Notes
- `ultralytics` provides the training loop and model modules. For experimenting, try `yolov8n.pt` (nano), `yolov8s.pt` (small), etc.
- Monitor training outputs in `runs/train/`.
