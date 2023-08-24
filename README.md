# MGDN_official
This is official Pytorch implementation of "Mutual-Guided Dynamic Network for Image Fusion".

### Enviroment Configuration
```
pip install -r requirements.txt
```

## Test
**Test Multiple Exposure Fusion (MEF)**

```
python Inference_MGDN_MEF.py --model_path=./Pretrain_model/MEF/ --iter_number=5500 --dataset=MEF_Benchmark --A_dir=testsets/under --B_dir=testsets/over --root_path='./Dataset/MEF'
```
**Test Multiple Focus Fusion (MFF)**

```
python Inference_MGDN_MFF.py --model_path=./Pretrain_model/MFF/ --iter_number=62000 --dataset=Lytro --A_dir=testsets/A --B_dir=testsets/B --root_path='./Dataset/MFF'
```
**Test HDR Deghosting**

```
python Inference_MGDN_Deghosting.py -config ./Pretrain_model/HDR_Deghosting/config.yaml
```
**Test RGB-guided Depth Super Resolution (GDSR)**

- 4x

```
python Inference_MGDN_GDSR.py --save_path ./Results/DepthSR/NYU_4x_MGDN --depth_path ./Dataset/GDSR/Depth --rgb_path ./Dataset/GDSR/RGB --depth_path_valid  ./Dataset/GDSR/Depth --rgb_path_valid  ./Dataset/GDSR/RGB --scale 4 --log_path ./Results/DepthSR --model_path ./Pretrain_model/GDSR --model_name MGDN_8x --statedict_path ./Pretrain_model/GDSR/GDSR_4x_1.496.pth --lr 1e-5 
```

- 8x

```
python Inference_MGDN_GDSR.py --save_path ./Results/DepthSR/NYU_8x_MGDN --depth_path ./Dataset/GDSR/Depth --rgb_path ./Dataset/GDSR/RGB --depth_path_valid  ./Dataset/GDSR/Depth --rgb_path_valid  ./Dataset/GDSR/RGB --scale 8 --log_path ./Results/DepthSR --model_path ./Pretrain_model/GDSR --model_name MGDN_8x --statedict_path ./Pretrain_model/GDSR/GDSR_8x_3.117.pth --lr 1e-5 
```

- 16x

```
python Inference_MGDN_GDSR.py --save_path ./Results/DepthSR/NYU_16x_MGDN --depth_path ./Dataset/GDSR/Depth --rgb_path ./Dataset/GDSR/RGB --depth_path_valid  ./Dataset/GDSR/Depth --rgb_path_valid  ./Dataset/GDSR/RGB --scale 16 --log_path ./Results/DepthSR --model_path ./Pretrain_model/GDSR --model_name MGDN_8x --statedict_path ./Pretrain_model/GDSR/GDSR_16x_5.813.pth --lr 1e-5 
```

## Train

Coming soon.

