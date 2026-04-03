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

**Steps To Add A New Dataset And Train**

1) Prepare image folders
- Create the folders:
	- `dataset/images/train`
	- `dataset/images/val`
	- `dataset/labels/train`
	- `dataset/labels/val`

2) Label images (YOLO format)
- Each image must have a `.txt` file with the same basename in the labels folder.
- Each line in the `.txt` follows: `class_id x_center y_center width height` (normalized 0-1).
- Example (one object):
	```
	0 0.201563 0.357812 0.196875 0.228125
	```

3) Create `classes.txt` (optional helper)
- Put one class name per line in `classes.txt` (order defines `class_id`).

4) Write `data.yaml` at repo root
- Minimal example:
	```yaml
	path: ./dataset
	train: images/train
	val: images/val
	nc: 1
	names: ["your_class_name"]
	```

5) (Optional) Convert other formats
- If you have Pascal VOC XMLs, run the helper:
	```bash
	python utils/convert_pascal_voc_to_yolo.py --xml_dir path/to/xmls --img_dir path/to/images --out_dir dataset/labels --classes classes.txt
	```

6) Quick label sanity check (Python snippet)
```python
from PIL import Image, ImageDraw
img = Image.open('dataset/images/train/000.jpg')
draw = ImageDraw.Draw(img)
with open('dataset/labels/train/000.txt') as f:
		for line in f:
				cls, cx, cy, w, h = map(float, line.split())
				W, H = img.size
				x0 = (cx - w/2) * W
				y0 = (cy - h/2) * H
				x1 = (cx + w/2) * W
				y1 = (cy + h/2) * H
				draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
img.show()
```

7) Train (fine-tune)
- Use the provided trainer. Adjust `--weights`, `--epochs`, `--imgsz`, and `--batch` as needed:
	```bash
	python train_custom.py --data data.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16
	```
- Outputs are saved to `runs/train/<name>/weights/` (look for `best.pt` and `last.pt`).

8) Resume or continue training
- To resume from a checkpoint, set `--weights runs/train/<exp>/weights/last.pt`.

9) Inference / validation
- Run detection on images or folders:
	```bash
	python detect.py --weights runs/train/<name>/weights/best.pt --source dataset/images/val --save
	```
- Or use the Streamlit UI to load the trained weights and test images or webcam.

10) Troubleshooting & tips
- Ensure `nc` in `data.yaml` equals number of classes in `names`/`classes.txt`.
- Labels must be normalized floats in [0,1] and boxes should lie inside image bounds.
- If training is very slow on CPU: reduce `--imgsz` and `--epochs` or train on a GPU.
- For small datasets: use transfer learning with small `--epochs` and data augmentation.
- Inspect `runs/train/<name>/events` and saved plots to check loss/metrics.

11) Next steps after successful training
- Export or convert `best.pt` for deployment, evaluate on a separate test set, or add more classes and repeat.

If you want, I can also add a small `scripts/validate_labels.py` that walks the dataset and reports malformed labels and missing files.

**Labeling Tools & Best Workflow**

Recommended tools
- **LabelImg (desktop)** — simple, local, exports YOLO `.txt` files. Good for small/single-user workflows.
- **MakeSense.ai (web)** — no-install, quick start, export YOLO. Best for one-off labeling without setup.
- **Roboflow (web)** — upload, auto-prelabel, augment, split, export YOLO. Great for repeatable pipelines and small/medium projects.
- **CVAT (server)** — collaborative, advanced tools (interpolation, review). Best for teams and complex workflows.
- **Label Studio** — flexible multi-task labeling; useful when you need custom label types.

Minimal labeling workflow
1. Collect images into a single folder for the task (e.g., `images_to_label/`).
2. Define a short, unambiguous class specification (who labels what, occlusion rules).
3. Choose a tool and create a new project (or open MakeSense.ai for instant use).
4. Draw bounding boxes and assign classes; follow the spec consistently.
5. Export annotations in YOLO format: one `.txt` per image with `class x_center y_center width height` (normalized).
6. Verify a sample of labels visually and with an automated checker (see snippet below).
7. Move images/labels into `dataset/images/{train,val}` and `dataset/labels/{train,val}` and write `data.yaml`.

Quick tool-specific how-tos
- LabelImg: install (`pip install labelImg` or download executable). run Program in terminal `labelImg`. Open image folder, draw boxes, save as YOLO format so `.txt` files appear beside each image.
- MakeSense.ai: open https://www.makesense.ai → Upload images → define classes → label boxes → Export → choose `YOLO`.
- Roboflow: create project → upload images → label or pre-annotate → create a version (optional augment/split) → Export as `YOLOv5/v8`.
- CVAT: deploy with Docker, create task, invite annotators, label collaboratively, export YOLO.

Automated label verification (quick Python snippet)
```python
from PIL import Image
from pathlib import Path

def check_labels(img_dir, lbl_dir):
	img_dir = Path(img_dir)
	for img_path in img_dir.glob('*.jpg'):
		lbl = Path(lbl_dir) / (img_path.stem + '.txt')
		if not lbl.exists():
			print('Missing label for', img_path.name)
			continue
		W, H = Image.open(img_path).size
		for i, line in enumerate(open(lbl)):
			parts = line.split()
			if len(parts) != 5:
				print('Malformed label', lbl, 'line', i+1)
				continue
			cls, cx, cy, w, h = map(float, parts)
			if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
				print('Out of range coords', lbl, 'line', i+1)

# usage: check_labels('dataset/images/train', 'dataset/labels/train')
```

Labeling best practices
- Keep class definitions short and unambiguous; provide labelers examples of edge cases.
- Label the full object extent (not only parts), and be consistent about occlusion rules.
- Use pre-labeling (run an existing model to auto-annotate) and then correct labels — speeds labeling significantly.
- Maintain a balanced train/val split; ensure each class appears in validation.
- Run the verification script above to catch missing/malformed labels before training.
- For small datasets, use transfer learning + augmentation; for large datasets, ensure diverse sampling and review.

Notes
- `ultralytics` provides the training loop and model modules. For experimenting, try `yolov8n.pt` (nano), `yolov8s.pt` (small), etc.
- Monitor training outputs in `runs/train/`.
