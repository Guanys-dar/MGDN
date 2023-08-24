# python inference.py -config /code/HDR/config/HDR_KPN.yaml

import os
import sys
import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
import scipy.io as scio
from torch.nn import init
from utils.running_func import *
from utils.utils import *
import yaml
# from utils.utils import OrderedYaml
import collections
sys.path.append("./models")
from models.MGFF.DYKPN import MGFF_Deghosting

Loader, Dumper = OrderedYaml()

if __name__ == "__main__":
    import torch 
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='MGDN')
    parser.add_argument('-config',default=None)
    args = parser.parse_args()
    print(args)

    with open(args.config, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)

    print(opt)
    dataset_config = opt["datasets"]
    train_config = opt["TrainSetting"]
    network_config = opt["NetworkSetting"]
    save_config = opt["Saveset"]
    if opt["use_cuda"] and torch.cuda.is_available():
        torch.cuda.manual_seed(int(train_config["Seed"]))

    print("==> load data")
    val_loader = torch.utils.data.DataLoader(
        val_data_loader(dataset_config["test_data"], crop=False, patch_size=1),
        batch_size=1, shuffle=False, num_workers=1)

    print("==> load model")
    model = MGFF_Deghosting(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6],window_size=8, kernel_num =[1, 1, 1],mlp_ratio=4.,inference=False)
    model.cuda()
    cuda_index = opt["cuda_idx"]
    model = torch.nn.DataParallel(model.cuda(), device_ids=[cuda_index])

    print("==>load pretrain")
    pretrain_path = "./Pretrain_model/HDR_Deghosting/Psnr_44.434_trained_model_12301.pth"
    state_dict = torch.load(pretrain_path)
    model.load_state_dict(state_dict, strict=True)

    print("==> test cycle")
    eval_result = val(0, model, val_loader)

