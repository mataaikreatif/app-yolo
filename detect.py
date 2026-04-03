"""Run object detection with a YOLO model (Ultralytics).

Usage examples:
  python detect.py --source data/images --weights yolov8n.pt --conf 0.25
  python detect.py --source image.jpg --weights runs/train/exp/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt", help="model weights or .pt file")
    p.add_argument("--source", default="0", help="image/video source, path or camera index")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--save", action="store_true", help="save results to runs/detect/")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=args.save)
    print(f"Ran detection on {args.source} using {args.weights}")


if __name__ == "__main__":
    main()
