import torch
import numpy as np
import cv2

def pader(img, unit=32):
    assert len(img.shape) == 4, f"img should have 4 dimensions, but got {len(img.shape)}"
    h, w = img.shape[2:]
    h = (h//unit+1)*unit
    w = (w//unit+1)*unit
    return torch.nn.functional.pad(img, (0, w-img.shape[3], 0, h-img.shape[2]), mode='reflect')

def fetcher(imgs, t):
    assert len(imgs.shape) == 5, f"imgs should have 4 dimensions, but got {len(imgs.shape)}"
    gt_idx = np.clip(round(imgs.shape[1]*t), 1, imgs.shape[1]-1)
    gt = imgs[:,gt_idx]
    return torch.cat([imgs[:,0], imgs[:,-1]], 1), gt

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, projection_path_list:list, fold=4, normalize_type="dtype", input_channels=1):
        assert type(projection_path_list) == list, f"projection_path_list should be list, but got {type(projection_path_list)}"
        self.projection_path_list = np.array(projection_path_list)[:len(projection_path_list)//fold*fold].reshape(-1, fold)
        self.projection_path_list = np.concatenate([
            self.projection_path_list, np.concatenate([
                self.projection_path_list[1:, 0:1],
                self.projection_path_list[0:1, 0:1]
        ], 0)], -1)
        self.input_channels = input_channels
        self.normalize_type = normalize_type
        assert self.normalize_type in ["minmax", "meanstd", "dtype"], f"Unsupported normalize_type: {self.normalize_type}"

    def normalize(self, img):
        if self.normalize_type == "dtype":
            if img.dtype == np.uint16:
                img = img/2**16
            elif img.dtype == np.uint8:
                img = img/2**8
            else:
                raise ValueError(f"Unsupported dtype: {img.dtype}")
        elif self.normalize_type == "minmax":
            img = (img-img.min())/(img.max()-img.min())
        elif self.normalize_type == "meanstd":
            img = (img-img.mean())/img.std()
        return img
    
    def __len__(self):
        return len(self.projection_path_list)
    
    def __getitem__(self, idx):
        imgs = []
        for i, path in enumerate(self.projection_path_list[idx]):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
            img = self.normalize(img)
            img = torch.from_numpy(img).permute(2, 0, 1)
            if self.input_channels == 3 and img.shape[0] == 1:
                img = torch.cat([img] * 3, 0)
            elif self.input_channels == 1 and img.shape[0] == 3:
                img = img[0:1]
            imgs.append(img)
        # return torch.cat(imgs, 0).float()
        return torch.stack(imgs).float()

if __name__ == "__main__":
    from pathlib import Path
    projection_path_list = sorted(list(Path("/home/kcj/nas_aict/dataset/CT/Walnuts/Walnut1/Projections/tubeV2").glob("*scan*.tif")))
    dataset = PathDataset(projection_path_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    t = 0.5
    for i, data in enumerate(dataloader):
        img1, gt, img2 = fetcher(data, t)
        break