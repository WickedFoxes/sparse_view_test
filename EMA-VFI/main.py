from pathlib import Path
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "interpolator")) # Add the path to the baseline directory
from interpolator.Trainer import Model
from interpolator.model import feature_extractor, flow_estimation
from interpolator.config import init_model_config
import cv2
import torch
from tqdm import tqdm
from data import dataset
from metrics import batch_psnr, batch_ssim
import wandb

wandb.init(project="CT") # test

MODEL_CONFIG = {
    'LOGNAME': 'ours_small_1ch',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_model_config(
        I = 1,
        F = 16,
        W = 7,
        depth = [2, 2, 2, 2, 2]
    )
}
MODEL_CONFIG = {
    'LOGNAME': 'ours_small',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_model_config(
        I = 3,
        F = 16,
        W = 7,
        depth = [2, 2, 2, 2, 2]
    )
}
MODEL_CONFIG = {
    'LOGNAME': 'ours',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_model_config(
        I = 3,
        F = 32,
        W = 7,
        depth = [2, 2, 2, 4, 4]
    )
}
in_chans = MODEL_CONFIG['MODEL_ARCH'][0]['data_chans']
trainset = dataset.PathDataset( # test
    sorted(list(Path("/home/kcj/nas_aict/dataset/CT/Walnuts/Walnut1/Projections/tubeV2").glob("*scan*.tif"))),
    fold = 8,
    normalize_type="minmax",
    input_channels=in_chans,
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
validset = dataset.PathDataset(
    sorted(list(Path("/home/kcj/nas_aict/dataset/CT/Walnuts/Walnut2/Projections/tubeV2").glob("*scan*.tif"))), 
    fold = 8,
    normalize_type="minmax",
    input_channels=in_chans,
)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False)
model = Model(MODEL_CONFIG)
model.device()
# model.load_model(MODEL_CONFIG['LOGNAME'])
print("Model loaded")
for epoch in range(300):
    pbar = tqdm(trainloader, ncols=100, desc=f"Train {epoch}")
    loss_list = []
    model.train()
    for i, data in enumerate(pbar):
        imgs, gt = dataset.fetcher(data, t=0.5)
        imgs, gt = dataset.pader(imgs), dataset.pader(gt)
        pred, loss = model.update(imgs.cuda(), gt.cuda(), 1e-5, training=True)
        loss_list.append(loss)
        pbar.set_postfix_str(f"loss: {loss:.4f}, psnr: {np.mean(list(batch_psnr(pred.cpu(), gt))):.2f}, ssim: {np.mean(list(batch_ssim(pred.cpu(), gt))):.2f}")
    
    pbar = tqdm(validloader, ncols=100, desc=f"Valid {epoch}")
    model.eval()
    for i, projection_path in enumerate(pbar):
        imgs, gt = dataset.fetcher(projection_path, t=0.5)
        imgs, gt = dataset.pader(imgs), dataset.pader(gt)
        with torch.no_grad():
            pred, _ = model.update(imgs.cuda(), gt.cuda(), 1e-5, training=False)
        pbar.set_postfix_str(f"psnr: {np.mean(list(batch_psnr(pred.cpu(), gt))):.2f}, ssim: {np.mean(list(batch_ssim(pred.cpu(), gt))):.2f}")
    model.save_model()