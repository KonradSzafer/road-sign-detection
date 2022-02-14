import os
import shutil
from pathlib import Path
from glob import glob
from xml.etree import ElementTree

import pandas as pd
from sklearn.model_selection import train_test_split


def get_abs_path(n_parent: int = 0):
    return Path('../' * n_parent).resolve()


def parse_annotations_xml(filename):
    dataset_path = Path(filename).parent.parent
    images_root = dataset_path / 'images'

    root = ElementTree.parse(filename).getroot()
    annotations = {}
    img_name = root.find("./filename").text
    annotations['name'] = img_name.replace('.png', '')
    annotations['annotations_filename'] = str(filename)
    annotations['img_filename'] = str(images_root) + '/' + img_name
    annotations['width'] = int(root.find("./size/width").text)
    annotations['height'] = int(root.find("./size/height").text)
    annotations['class'] = root.find("./object/name").text
    annotations['xmin'] = int(root.find("./object/bndbox/xmin").text)
    annotations['xmax'] = int(root.find("./object/bndbox/xmax").text)
    annotations['ymin'] = int(root.find("./object/bndbox/ymin").text)
    annotations['ymax'] = int(root.find("./object/bndbox/ymax").text)
    return annotations


def create_annotations_list(annotations_path):
    annotations_file_list = list(annotations_path.iterdir())[:]
    annotations_list = []
    for filename in annotations_file_list:
        annotation = parse_annotations_xml(filename)
        annotations_list.append(annotation)
    return annotations_list
