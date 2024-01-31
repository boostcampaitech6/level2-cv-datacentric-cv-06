import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
import os
import sys

import argparse

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def main(args):
    prediction = f'/data/ephemeral/home/level2-cv-datacentric-cv-06/predictions/{args.train_serial}/{args.output}'
    data = read_json(prediction)

    img_dir = '../data/medical/img/test'
    save_dir = f'/data/ephemeral/home/level2-cv-datacentric-cv-06/predictions/{args.train_serial}/img'
    os.makedirs(save_dir, exist_ok=True)
    img_list = os.listdir(img_dir)
    
    for img_name in img_list:
        image = os.path.join(img_dir, img_name)
        origin_img = Image.open(image).convert('RGB')
        draw = ImageDraw.Draw(origin_img)
        
        for id in range(len(data['images'][img_name]['words'])):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = data['images'][img_name]['words'][str(id)]['points']
            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)
            draw.rectangle((x_min, y_min, x_max, y_max), outline=(0,255,0), width = 2)
        
        origin_img.save(os.path.join(save_dir, img_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_serial", type=str, default='20230000_0000', help="output dir"
    )
    parser.add_argument(
        "--output", type=str, default='output.csv', help="output.csv"
    )

    args = parser.parse_args()
    main(args)