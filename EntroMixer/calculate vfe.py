import cv2
import numpy as np


def partial_derivative(img):
    img = np.pad(img, ((0, 1), (0, 1)), constant_values=0)
    h, w = img.shape
    df_gray = np.zeros([h - 1, w - 1])
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray = img[i, j + 1] - img[i, j]
            dy_gray = img[i + 1, j] - img[i, j]
            df_gray[i, j] = np.square(dx_gray) + np.square(dy_gray)
    return df_gray

def calculate_vfe(img: object, size: object, stride: object) -> object:
    image = []
    target = []
    num_h = (img.shape[0] - size) // stride + 1
    num_w = (img.shape[1] - size) // stride + 1
    for h in range(num_h):
        for w in range(num_w):
            img_crop = img[h * stride:h * stride + size, w * stride:w * stride + size]
            image.append(img_crop)
            crop = img_crop - np.mean(img_crop)
            crop = crop * crop
            target.append(crop / (stride * stride - 1))
    entropy = 0
    for crop in image:
        crop = partial_derivative(crop)
        entropy += np.sum(crop)
    entropy = entropy / len(image)
    return entropy, np.mean(target)

if __name__ == '__main__':
    img = cv2.imread("053-0-13.75.jpg")
    vfe = 0
    for i in range(3):
        pe, t = calculate_vfe(inputs[:, :, i], size=7, stride=7)
        vfe += pe / t
    print(VFE)