import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from data_utils import get_abs_path


def get_bounding_box(row):
    return np.array([
        row['xmin'],
        row['xmax'],
        row['ymin'],
        row['ymax']])


def create_mask(img_width, img_height, bounding_box, r=255, g=0, b=0):
    bb_x_min, bb_x_max, bb_y_min, bb_y_max = bounding_box
    mask = np.zeros((img_width, img_height, 3))
    cv2.rectangle(mask, (int(bb_x_min), int(bb_y_min)), (int(bb_x_max), int(bb_y_max)), (r,g,b), 2)
    return mask


def get_bb_from_mask(mask):
    cols, rows = np.nonzero(mask[:,:,0])
    bounding_box = np.array([np.min(rows),
                            np.max(rows),
                            np.min(cols),
                            np.max(cols)])
    return bounding_box


def resize_img_and_bb(img_path, bounding_box, new_width, new_height):

    img = cv2.imread(img_path)
    img_width, img_height, _ = img.shape

    img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # plt.imshow(img)
    # plt.show()

    mask = create_mask(img_width, img_height, bounding_box)
    mask = cv2.resize(mask, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # plt.imshow(mask)
    # plt.show()

    bounding_box = get_bb_from_mask(mask)
    return img, bounding_box


def plot_img_with_mask(img, bounding_box, title=''):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = create_mask(img.shape[0], img.shape[1], bounding_box)
    plt.title(title)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    plt.show()


if __name__ == '__main__':

    data_path = get_abs_path(1)
    data_path = data_path / 'data' / 'images'
    img_path = data_path / 'road54.png'

    img, bounding_box = resize_img_and_bb(str(img_path), np.array([20, 60, 10, 100]), 300, 400)
    plot_img_with_mask(img, bounding_box)

    mask = create_mask(300, 400, bounding_box)
    bounding_box = get_bb_from_mask(mask)
    print(bounding_box)
