"""Convert Pascal VOC XML annotations to YOLO txt format.

Usage:
  python convert_pascal_voc_to_yolo.py --xml_dir dataset/annotations/xmls --img_dir dataset/images --out_dir dataset/labels

Note: This is a minimal helper. It assumes class names are provided in `classes.txt` (one per line)
and that XML filenames match image filenames (except extension).
"""
import os
import argparse
import xml.etree.ElementTree as ET


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_dir", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--classes", default="classes.txt", help="file with class names, one per line")
    return p.parse_args()


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def convert(xml_path, classes, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        b = obj.find('bndbox')
        xmin = float(b.find('xmin').text)
        ymin = float(b.find('ymin').text)
        xmax = float(b.find('xmax').text)
        ymax = float(b.find('ymax').text)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines


def main():
    args = parse_args()
    classes = load_classes(args.classes)
    os.makedirs(args.out_dir, exist_ok=True)
    for xml in os.listdir(args.xml_dir):
        if not xml.endswith('.xml'):
            continue
        xml_path = os.path.join(args.xml_dir, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        img_w = float(size.find('width').text)
        img_h = float(size.find('height').text)
        yolo_lines = convert(xml_path, classes, img_w, img_h)
        txt_name = os.path.splitext(xml)[0] + '.txt'
        with open(os.path.join(args.out_dir, txt_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))


if __name__ == '__main__':
    main()
