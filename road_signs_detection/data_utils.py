import os
import shutil
from pathlib import Path
from glob import glob
from xml.etree import ElementTree

from sklearn.model_selection import train_test_split


def get_abs_path(n_parent: int = 0):
    return Path('../' * n_parent).resolve()


def split_dataset(dataset_images_path, dataset_annotations_path,
                    train_images_path, train_annotations_path,
                    test_images_path, test_annotations_path) -> None:

    # get data samples paths
    images_paths = list(dataset_images_path.iterdir())[:]
    annotations_paths = list(dataset_annotations_path.iterdir())[:]

    # split to train and test
    test_size = 0.3

    train_images, test_images = train_test_split(images_paths,
                                                test_size=test_size,
                                                shuffle=False)

    train_annotations, test_annotations = train_test_split( annotations_paths,
                                                            test_size=test_size,
                                                            shuffle=False)

    # copy files to new folders
    for image in train_images:
        shutil.copy(image, train_images_path)
    for annotation in train_annotations:
        shutil.copy(annotation, train_annotations_path)

    for image in test_images:
        shutil.copy(image, test_images_path)
    for annotation in test_annotations:
        shutil.copy(annotation, test_annotations_path)


def parse_annotations_xml(filename):
    root = ElementTree.parse(filename).getroot()
    annotations = {}
    annotations['filename'] = str(filename)
    annotations['width'] = root.find("./size/width").text
    annotations['height'] = root.find("./size/height").text
    annotations['class'] = root.find("./object/name").text
    annotations['xmin'] = int(root.find("./object/bndbox/xmin").text)
    annotations['ymin'] = int(root.find("./object/bndbox/ymin").text)
    annotations['xmax'] = int(root.find("./object/bndbox/xmax").text)
    annotations['ymax'] = int(root.find("./object/bndbox/ymax").text)
    return annotations


if __name__ == '__main__':

    root_dir = get_abs_path(1)
    dataset_images_dir = root_dir / 'data' / 'images'
    dataset_annotations_dir = root_dir / 'data' / 'annotations'
    train_images_dir = root_dir / 'road_signs_detection' / 'train' / 'images'
    test_images_dir = root_dir / 'road_signs_detection' / 'test' / 'images'
    train_annotations_dir = root_dir / 'road_signs_detection' / 'train' / 'annotations'
    test_annotations_dir = root_dir / 'road_signs_detection' / 'test' / 'annotations'
    train_images_dir.mkdir(exist_ok=True, parents=True)
    test_images_dir.mkdir(exist_ok=True, parents=True)
    train_annotations_dir.mkdir(exist_ok=True, parents=True)
    test_annotations_dir.mkdir(exist_ok=True, parents=True)

    split_dataset(  dataset_images_dir, dataset_annotations_dir,
                    train_images_dir, train_annotations_dir,
                    test_images_dir, test_annotations_dir)
