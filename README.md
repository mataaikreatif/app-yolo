# YOLO learning app (Python)

Description
This repository is a minimal, self-contained learning app for YOLOv8. It includes
scripts to run inference with pretrained weights, fine-tune on your own dataset,
convert simple Pascal VOC annotations, a Streamlit demo UI (`app.py`), and a
`run_demo.py` script that generates a tiny synthetic dataset and performs a
short training run so you can try the full workflow end-to-end.

Key features
- Quick inference with `detect.py`.
- Transfer-learning / fine-tuning via `train_custom.py`.
- Synthetic demo (auto dataset generation + short training) in `run_demo.py`.
- Simple web UI with `app.py` (Streamlit) for uploading images and visualizing detections.

Prerequisites
- Python 3.8+
- GPU recommended for training (CPU works but is slower)

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

Demo script
```bash
python run_demo.py
```
`run_demo.py` will create a tiny synthetic dataset under `dataset/`, train a small
model for a few epochs, and save an example detection visualization as
`demo_output.jpg`.

Example label (YOLO format)
Each label file `.txt` contains one line per object with the format:
```
class_id x_center y_center width height
```
All coordinates are normalized to the image size (values between 0 and 1).

Sample (from `dataset/labels/val/val_005.txt`):
```
0 0.201563 0.357812 0.196875 0.228125
```
This line means: class `0`, center at `(0.201563, 0.357812)`, box width `0.196875`, box height `0.228125`.


Notes
- `ultralytics` provides the training loop and model modules. For experimenting, try `yolov8n.pt` (nano), `yolov8s.pt` (small), etc.
- Monitor training outputs in `runs/train/`.
