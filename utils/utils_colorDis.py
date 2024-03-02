import cv2
from pyciede2000 import ciede2000
import numpy as np


def calculate_avg_lab(img_rgb):
    # 读取图像
    # img = cv2.imread(image_path)
    img = img_rgb

    # 转换到 Lab 色彩空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # 计算平均 Lab 值
    avg_lab = np.mean(lab_img, axis=(0, 1))

    avg_lab[0] = (avg_lab[0] * 100) / 255.0  # L*
    avg_lab[1] = avg_lab[1] - 128  # a*
    avg_lab[2] = avg_lab[2] - 128  # b*

    return avg_lab


def calculate_ciede2000_between_images(imgrgb1, imgrgb2):
    lab1 = calculate_avg_lab(imgrgb1)
    lab2 = calculate_avg_lab(imgrgb2)

    # 计算 CIEDE2000 色差
    delta_e = ciede2000(lab1, lab2)

    return delta_e


if __name__ == "__main__":
    # 示例使用
    image_path1 = "path_to_first_image.jpg"
    image_path2 = "path_to_second_image.jpg"

    color_difference = calculate_ciede2000_between_images(image_path1, image_path2)
    print(f"Color difference (CIEDE2000) between images: {color_difference}")

