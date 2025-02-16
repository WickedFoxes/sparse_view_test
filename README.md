# sparse_view_test

## JDINet
### 라이브러리 설치
    git clone https://github.com/LianyingChao/FusionLowDoseCBCT.git
    pretrained JDINet : https://pan.baidu.com/share/init?surl=pjlDlRAYweXKySwrAfKrXg&pwd=nnix
    
    conda create -n JDINet python=3.7
    conda activate JDINet
    <!-- conda remove pytorch-cpu cpuonly -->

    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    conda install -c astra-toolbox -c nvidia astra-toolbox

    cd FusionLowDoseCBCT
    pip install -r requirements.txt

### 데이터셋
#### walnut
    https://zenodo.org/records/3763412

### interpolation, fdk
    conda activate JDINet
    cd FusionLowDoseCBCT/JDINet
    python interpolation_and_fdk --div 2 --model_path "./saved_model/JDINet_inter.pth" --dataset_dir "../ld_proj/walnut_19/good" --interpolation_dir "../walnut19_div2_interpolation" --fdk_dir "../walnut19_div2_interpolation_fkd"

interpolation 결과 : tif, float 0 ~ 1 정규화값, (640, 640)

fdk 결과 : png, uint8 0 ~ 255, (448, 448)

### postnet
    conda activate JDINet
    cd FusionLowDoseCBCT/PostNet
    python postprocess.py --fdk_dir "../walnut19_div2_interpolation_fkd" --post_dir "../walnut19_div2_interpolation_fkd_post"

postnet 결과 : png, uint8 0 ~ 255, (448, 448)

## InterpAny-Clearer
### 라이브러리 설치
    git clone https://github.com/zzh-tech/InterpAny-Clearer.git
    
    conda create -n InterpAny python=3.8
    conda activate InterpAny
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    
    cd InterpAny-Clearer
    pip install -r requirements.txt