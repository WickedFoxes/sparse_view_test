import os
import shutil
import cv2
import math
import sys
import torch
import numpy as np
import argparse
import imageio as iio
from imageio import mimsave
import os

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from model import feature_extractor
from model import flow_estimation

'''==========arg inputs=========='''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="ours_small", help=' : Please set the name')
parser.add_argument('--div', type=int, default=2, help=' : Please set the num')
parser.add_argument('--dataset_dir', type=str, default="../../ld_proj/walnut_19/good/", help=' : Please set the dir')
parser.add_argument('--interpolation_dir', type=str, default="../../ours_small_walnut19_div2_interpolation", help=' : Please set the dir')
args = parser.parse_args()
print(args)

div = args.div
dataPath=args.dataset_dir
interpolationDir=args.interpolation_dir
os.makedirs(interpolationDir, exist_ok=True)

#################################################################

# EMA-VFI interpolation

##################################################################

'''==========Model setting=========='''
model_name = args.model_name

TTA = True
if model_name == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )

model = Model(cfg.MODEL_CONFIG)
model.load_model()
model.eval()
model.device()


print(f'=========================Start Generating=========================')
def crop_center(image_path, crop_size=640):
    # 이미지 읽기 (imageio는 NumPy 배열 반환)
    img = iio.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # print("변환된 이미지 :", img.shape, img.min(), img.max())  # (H, W, 3)

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
def normalize_image(image):
    return ((image - image.min()) / (image.max() - image.min())).astype(np.float32)


images = []

for proj_id in range(501):
    file0_name = f"scan_{proj_id:06d}.tif"
    I0 = crop_center(os.path.join(dataPath, file0_name))
    I0 = normalize_image(I0)
    images.append(I0)

for proj_id in range(0, 500, div):
    I0 = images[proj_id]
    I2 = images[proj_id%501]

    # I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I0_ = torch.tensor(I0.transpose(2, 0, 1)).cuda().unsqueeze(0)
    I2_ = torch.tensor(I2.transpose(2, 0, 1)).cuda().unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)
    
    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./div) for i in range(div - 1)], fast_TTA=TTA)
    temp_idx = proj_id
    for pred in preds:
        temp_idx += 1
        mid = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0)).astype(np.float32)
        images[temp_idx] = mid

print(len(images))
for idx in range(501):
    I0 = images[idx]
    img = cv2.cvtColor(I0[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    print(idx, " 변환된 이미지 :", img.shape, img.min(), img.max())  # (H, W, 3)
    iio.imsave(os.path.join(interpolationDir, f"scan_{idx:06d}.tif"), img)

print(f'=========================Done=========================')