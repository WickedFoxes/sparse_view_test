import astra
import os
import shutil
import torch
import numpy as np
import imageio
import time
import scipy.io as io
from torch.autograd import Variable
from torch.nn import functional as F
from model_inter.JDINet_inter import UNet_3D_2D
import argparse

def get_pairData(div, max_num=501):
    num_list = list(range(0, max_num, div))

    pairData = []
    n = len(num_list)
    if div%2==0: n=n-1
    print("pairData 개수 :", n)

    for i in range(n):  # 500을 div 크기로 나눈 만큼 반복
        pair_array = [num_list[i-1], num_list[i], num_list[(i+1)%n], num_list[(i+2)%n]]
        pairData.append(pair_array)

        interp_array = list(range(pair_array[1]+1, pair_array[1]+div))
        interp_array = [x%max_num for x in interp_array]
        pairData[i].extend(interp_array)

    return np.array(pairData)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def crop_center(image_path, crop_size=640):
    # 이미지 읽기 (imageio는 NumPy 배열 반환)
    img = imageio.imread(image_path)

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


parser = argparse.ArgumentParser()
parser.add_argument('--div', type=int, default=1, help=' : Please set the num') 
parser.add_argument('--model_path', type=str, default="./saved_model/JDINet_inter.pth", help=' : Please set the dir')
parser.add_argument('--dataset_dir', type=str, default="../ld_proj/walnut_19/good", help=' : Please set the dir')
parser.add_argument('--interpolation_dir', type=str, default="../full_clean_proj", help=' : Please set the dir')
parser.add_argument('--fdk_dir', type=str, default="../good_fdk_reconstruction", help=' : Please set the dir')

# Model
model_choices = ["unet_18", "unet_34"]
parser.add_argument('--model', choices=model_choices, type=str, default="unet_34")
parser.add_argument('--nbr_frame' , type=int , default=4)
parser.add_argument('--nbr_width' , type=int , default=1)
parser.add_argument('--joinType' , choices=["concat" , "add" , "none"], default="concat")
parser.add_argument('--upmode' , choices=["transpose","upsample"], type=str, default="transpose")
parser.add_argument('--n_outputs' , type=int, default=1, help="For Kx FLAVR, use n_outputs k-1")

args = parser.parse_args()
print(args)

#################################################################

# JDINet interpolation

##################################################################

print('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = UNet_3D_2D(args.model.lower() , 
                   n_inputs=args.nbr_frame, 
                   n_outputs=args.n_outputs, 
                   joinType=args.joinType, 
                   upmode=args.upmode)
model=model.to(device)

model_path=args.model_path
model_dict=model.state_dict()
model.load_state_dict(torch.load(model_path)["state_dict"] , strict=True)
model.eval()

div = args.div
dataPath=args.dataset_dir
interpolationDir=args.interpolation_dir
fdkDir = args.fdk_dir
pairData=get_pairData(div)

print(pairData.shape, pairData[0])
print(pairData.shape, pairData[-1])

# 폴더가 존재하지 않으면 생성
os.makedirs(interpolationDir, exist_ok=True)
os.makedirs(fdkDir, exist_ok=True)

if div > 1:
    with torch.no_grad():
        for idx in range(len(pairData)):
            projId=idx

            rawFrameInput=np.zeros((4,640,640))
            rawFrameGt=np.zeros((1,640,640))

            # load input frames
            for k in range(4):
                index=int(pairData[projId,k])
                img = crop_center(os.path.join(dataPath, f"scan_{index:06d}.tif"))
                img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)

                imageio.imsave(os.path.join(interpolationDir, f"scan_{index:06d}.tif"),img)
                rawFrameInput[k,:,:]=img              
            rawFrameInput=torch.tensor(rawFrameInput)
            rawFrameGt=torch.tensor(rawFrameGt)

            rawFrameInput=rawFrameInput.unsqueeze(0)

            # padding
            h0=rawFrameInput.shape[-2]
            if h0%32!=0:
                pad_h=32-(h0%32)
                rawFrameInput=F.pad(rawFrameInput,(0,0,0,pad_h),mode='reflect')
            
            rawFrameGt=rawFrameGt.unsqueeze(1)
            rawFrameInput=rawFrameInput.unsqueeze(1)
            rawFrameInput=rawFrameInput.float()
            rawFrameGt=rawFrameGt.float()

            rawFrameInput=to_variable(rawFrameInput)
            rawFrameGt=to_variable(rawFrameGt)

            out = model(rawFrameInput)
            img=out.detach().cpu().numpy()
            for j in range(div-1):
                index=int(pairData[projId,j+4])
                print("interpolation:", index)

                data = img[j,0,:,:]
                imageio.imsave(os.path.join(interpolationDir, f"scan_{index:06d}.tif"), data)
else:
    # 파일 복사 및 이름 변경
    for num in range(0, 501):
        filename = f"scan_{num:06d}.tif"  # scan_000000.tif, scan_000004.tif, ...
        source_file = os.path.join(dataPath, filename)
        dst_file = os.path.join(interpolationDir, filename)
        img = crop_center(source_file)
        img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
        
        imageio.imsave(dst_file, img)


#################################################################

# FDK reconstruction

##################################################################
angluar_sub_sampling = 1
voxel_per_mm = 10

t = time.time()
print('load data', flush=True)

theta = np.linspace(0, 2*np.pi,500)
vecs1=io.loadmat('../vectors.mat')['vectors'] 
# print(vecs.shape)
# quit()
vecs = np.zeros((500, 12))
for a in range(500):
    vecs[a,:] = vecs1[a, :]


# projection index
# there are in fact 1201, but the last and first one come from the same angle
projs_idx = range(1,501, angluar_sub_sampling)
projs_rows = 640
projs_cols = 640


# create the numpy array which will receive projection data from tiff files
projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)

trafo = lambda image : np.transpose(np.flipud(image))
# load projection data
for i in range(len(projs_idx)):
    print(projs_idx[i])
    a = imageio.imread(os.path.join(interpolationDir, f"scan_{projs_idx[i]:06d}.tif"))
    projs[i]=(a-a.min())/(a.max()-a.min())

#projs = projs[::-1,...]
projs = np.transpose(projs, (1,0,2))
projs = np.ascontiguousarray(projs)
print(np.round_(time.time() - t, 3), 'sec elapsed')

### compute FDK reconstruction #################################################

t = time.time();
print('compute reconstruction', flush=True)

# size of the reconstruction volume in voxels
vol_sz  = 3*(44 * 10 + 8,)
# size of a cubic voxel in mm
vox_sz  = 1/voxel_per_mm
# numpy array holding the reconstruction volume
vol_rec = np.zeros(vol_sz, dtype=np.float32)

# we need to specify the details of the reconstruction space to ASTRA
# this is done by a "volume geometry" type of structure, in the form of a Python dictionary
# by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
vol_geom = astra.create_vol_geom(vol_sz)
vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

# we need to specify the details of the projection space to ASTRA
# this is done by a "projection geometry" type of structure, in the form of a Python dictionary
proj_geom = astra.create_proj_geom('cone_vec', 640, 640, vecs)

# register both volume and projection geometries and arrays to ASTRA
vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)
proj_id = astra.data3d.link('-sino', proj_geom, projs)

# finally, create an ASTRA configuration.
# this configuration dictionary setups an algorithm, a projection and a volume
# geometry and returns a ASTRA algorithm, which can be run on its own
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = proj_id
cfg['ReconstructionDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)

# run FDK algorithm
astra.algorithm.run(alg_id, 1)

# release memory allocated by ASTRA structures
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)

print(np.round_(time.time() - t, 3), 'sec elapsed')



### save reconstruction ########################################################

t = time.time();
print('save results', flush=True)

# low level plotting
f, ax = plt.subplots(1, 3, sharex=False, sharey=False)
ax[0].imshow(vol_rec[vol_sz[0]//2,:,:])
ax[1].imshow(vol_rec[:,vol_sz[1]//2,:])
ax[2].imshow(vol_rec[:,:,vol_sz[2]//2])
f.tight_layout()
np.transpose(vol_rec,[0,1,2])

for i in range(200):
    a=vol_rec[:,:,i+150]
    a=255.0*(a-a.min())/(a.max()-a.min())
    print(a.min(), a.max())
    a = a.astype(np.uint8)
    # imageio.imsave(recon_path+'%d'%(i+1)+'.png', a)
    imageio.imsave(os.path.join(fdkDir, f"fdk_{(i+1):06d}.png"), a)
print(np.round_(time.time() - t, 3), 'sec elapsed')