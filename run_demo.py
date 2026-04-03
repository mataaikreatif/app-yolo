"""Generate synthetic dataset, train YOLOv8 briefly, and run inference.

Creates `dataset/` with images and labels, writes `data.yaml`, trains for a few epochs,
and saves an inference image `demo_output.jpg`.

Run: python run_demo.py
"""
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
IMG_SIZE = 320


def ensure_dirs():
    for p in [DATA_DIR / 'images' / 'train', DATA_DIR / 'images' / 'val', DATA_DIR / 'labels' / 'train', DATA_DIR / 'labels' / 'val']:
        p.mkdir(parents=True, exist_ok=True)


def write_classes(classes):
    with open(ROOT / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(classes))


def make_box_image(path, label_path, classes, n_boxes=1):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    labels = []
    for _ in range(n_boxes):
        w = random.randint(30, 100)
        h = random.randint(30, 100)
        x0 = random.randint(0, IMG_SIZE - w)
        y0 = random.randint(0, IMG_SIZE - h)
        x1 = x0 + w
        y1 = y0 + h
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        # normalized center x,y,w,h
        cx = (x0 + x1) / 2.0 / IMG_SIZE
        cy = (y0 + y1) / 2.0 / IMG_SIZE
        nw = w / IMG_SIZE
        nh = h / IMG_SIZE
        labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    img.save(path)
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))


def generate_dataset(classes, n_train=30, n_val=6):
    ensure_dirs()
    write_classes(classes)
    for i in range(n_train):
        img_path = DATA_DIR / 'images' / 'train' / f'train_{i:03d}.jpg'
        lbl_path = DATA_DIR / 'labels' / 'train' / f'train_{i:03d}.txt'
        make_box_image(img_path, lbl_path, classes)
    for i in range(n_val):
        img_path = DATA_DIR / 'images' / 'val' / f'val_{i:03d}.jpg'
        lbl_path = DATA_DIR / 'labels' / 'val' / f'val_{i:03d}.txt'
        make_box_image(img_path, lbl_path, classes)


def write_data_yaml(classes):
    data = {
        'path': str(DATA_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes,
    }
    with open(ROOT / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
    return str(ROOT / 'data.yaml')


def train_demo(data_yaml, weights='yolov8n.pt', epochs=5):
    model = YOLO(weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=IMG_SIZE, batch=8, project='runs/train', name='demo')
    return model


def run_inference(model, img_path, out_path='demo_output.jpg', conf=0.25):
    results = model.predict(source=str(img_path), conf=conf, imgsz=IMG_SIZE)
    try:
        vis = results[0].plot()
        vis_img = Image.fromarray(vis)
        vis_img.save(out_path)
        print(f"Saved inference visualization to: {out_path}")
    except Exception as e:
        print("Inference completed but failed to render image:", e)


def main():
    classes = ["box"]
    print("Generating synthetic dataset...")
    generate_dataset(classes)
    print("Writing data.yaml...")
    data_yaml = write_data_yaml(classes)
    print("Starting short training (this may take a few minutes)...")
    model = train_demo(data_yaml, weights='yolov8n.pt', epochs=3)
    val_img = DATA_DIR / 'images' / 'val' / 'val_000.jpg'
    if val_img.exists():
        run_inference(model, val_img, out_path='demo_output.jpg')
    else:
        print("No val image found to run inference on.")


if __name__ == '__main__':
    main()
