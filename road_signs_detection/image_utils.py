import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def get_bounding_box(row):
    return np.array([row[4], row[6]], row[5], row[7])


def resize_img_and_bb(img_path, bounding_box, new_width, new_height):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_width, img_height, _ = img.shape
    plt.imshow(img)
    plt.show()

    bb_x_min, bb_x_max, bb_y_min, bb_y_max = bounding_box
    mask = np.zeros((img_width, img_height, 3))
    cv2.rectangle(mask, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (255,0,0), 1)
    plt.imshow(mask)
    plt.show()

    img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST)
    cols, rows = np.nonzero(mask[:,:,0])
    bounding_box = np.array([np.min(rows),
                            np.max(rows),
                            np.min(cols),
                            np.max(cols)])

    return img, bounding_box


if __name__ == '__main__':

    img, bounding_box = resize_img_and_bb('./train/images/road1.png', np.array([20, 60, 10, 100]), 300, 450)
    print(bounding_box)
