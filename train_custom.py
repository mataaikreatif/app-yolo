"""Fine-tune (transfer-learn) YOLOv8 on a custom dataset.

Dataset must be in YOLO format (see README). Provide a `data.yaml` file
that points to `train` and `val` folders and lists `names`.

Example usage:
  python train_custom.py --data data.yaml --weights yolov8n.pt --epochs 30
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--weights", default="yolov8n.pt", help="pretrained weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/train", help="save directory")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
    )


if __name__ == "__main__":
    main()
