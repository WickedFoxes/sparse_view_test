import imageio as iio
import os
import numpy as np

def normalize_image(image):
    return ((image - image.min()) / (image.max() - image.min())).astype(np.float32)

def crop_center(image_path, crop_size=640):
    # 이미지 읽기 (imageio는 NumPy 배열 반환)
    img = iio.imread(image_path)

    # 원본 이미지 크기 가져오기
    height, width = img.shape[:2]  # (H, W, C) 또는 (H, W)

    # 중앙 좌표 계산
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # 이미지 자르기 (NumPy 슬라이싱)
    cropped_img = img[top:bottom, left:right]

    return cropped_img


'''==========arg inputs=========='''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default="/home/byeoli0832/sparse_view_test/ld_proj/walnut_19/noisy/", help=' : Please set the dir')
parser.add_argument('--rescale_dir', type=str, default="/home/byeoli0832/sparse_view_test/walnut19_noise", help=' : Please set the dir')
args = parser.parse_args()
print(args)

data_folder_path = args.dataset_dir
rescale_folder_path = args.rescale_dir

if not os.path.exists(rescale_folder_path):
    os.makedirs(rescale_folder_path)
for idx in range(501):
    file_name = f"scan_{idx:06d}.tif"
    img = crop_center(os.path.join(data_folder_path, file_name))
    img = normalize_image(img)
    iio.imsave(os.path.join(rescale_folder_path, file_name), img)