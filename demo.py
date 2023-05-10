from time import time 
from glob import glob
import argparse

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapidocr_openvinogpu import RapidOCR

def draw_inference(img_path, boxes, txts, scores=None, text_score=0.5):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (960, 544))
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and float(scores[idx]) < text_score:
            continue
        is_closed = True
        color = (0,255,0)
        thickness = 2
        pts = []
        for i in range(0, len(box)):
            pts.append([box[i][0], box[i][1]])
        pts = np.array(pts, np.int32)
        img = cv2.polylines(img, [pts], is_closed, color, thickness)
        font_scale = 1.5
        text_thickness = 2
        text_org = (pts[0][0], pts[0][1])
        img = cv2.putText(img, txt, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

    return img


def visualize(image_path, result, out_dir=None):
    image = Image.open(image_path)
    boxes, txts, scores = list(zip(*result))

    draw_img = draw_inference(image_path, boxes, txts, scores=None, text_score=0.5)

    draw_img_save = Path("./inference_results/")

    if out_dir is not None:
        draw_img_save = Path(out_dir)

    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True, exist_ok=True)

    image_save = str(draw_img_save / f'infer_{Path(image_path).name}')
    cv2.imwrite(image_save, draw_img)
    print(f'The infer result has saved in {image_save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--in-dir', required=False, help="Images directory")
    parser.add_argument('-f', '--in-file', required=False, help="Image path")
    parser.add_argument('-o', '--out-dir', required=False, help="Image path")
    parser.add_argument('-v', '--vis', required=False, action="store_true",
                        help="visualization")
    args = parser.parse_args()
    out_dir = args.out_dir

    rapid_ocr = RapidOCR()
    
    all_image_files = []

    if args.in_dir is not None:
        inp_dir = args.in_dir
        types = ('*.jpg', '*.jpeg', '*.png')

        for t in types:
            all_image_files.extend(glob(inp_dir+t))
    elif args.in_file is not None:
        all_image_files = [args.in_file]
    else:
        print("Input missing: either provide a folder (-d option) or a image path (-f)")

    for image_path in all_image_files:  
        with open(image_path, 'rb') as f:
            img = f.read()
        result, elapse_list = rapid_ocr(img)
        print(result)
        print(elapse_list)
        if args.vis:
            visualize(image_path, result, out_dir)
