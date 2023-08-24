# -*- coding: utf-8 -*
# !/usr/local/bin/python
import argparse, os, sys
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import *
import numpy as np
from utils.dsr_utils import *
from utils.dsr_dataloader import *
sys.path.append("./models")
from models.MGFF.DYKPN import MGFF_GDSR

parser = argparse.ArgumentParser(description="DepthSR")
# data loader
parser.add_argument("--depth_path", type=str,
                    default='/data/guanys3/DepthSR/Data/NYUDepthv2_Train/Depth',
                    help="HyperSet path")

parser.add_argument("--rgb_path", type=str,
                    default='/data/guanys3/DepthSR/Data/NYUDepthv2_Train/RGB',
                    help="RGBSet path")
parser.add_argument("--save_path", type=str, default='')
parser.add_argument('--dataset_name', type=str, default='NYU', help='test dataset name')
parser.add_argument("--depth_path_valid", type=str,
                    default='/data/guanys3/DepthSR/Data/NYUDepthv2_Test/Depth',
                    help="HyperSet path")
parser.add_argument("--rgb_path_valid", type=str,
                    default='/data/guanys3/DepthSR/Data/NYUDepthv2_Test/RGB',
                    help="RGBSet path")
parser.add_argument('--augmentation', type=bool, default=True, help='augmentation')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--scale', type=int, default=16, help='patch_size')

# model&events path
parser.add_argument('--log_path', default='/data/guanys3/DepthSR/log', help='log path')
parser.add_argument('--model_path', default='/data/guanys3/DepthSR/exp/demo', help='model')
parser.add_argument('--statedict_path', default='/data/guanys3/DepthSR/exp/demo', help='model')
parser.add_argument('--model_name', default='Ours_16x', help='model')

# train setting
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1200, help="number of epochs to train for")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--lr", type=float, default=1 * 1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--milestones", type=list, default=[1000, 1100], help="how many epoch to reduce the lr")
parser.add_argument("--gamma", type=int, default=0.5, help="how much to reduce the lr each time")

parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def valid(arg, model):
    torch.cuda.empty_cache()
    val_set = ValidLoader(arg)
    val_set_loader = DataLoader(dataset=val_set, num_workers=arg.threads, batch_size=1, shuffle=False)
    RMSE_epoch = 0
    RMSE_aver_epoch = 0
    SSIM_epoch = 0
    model.eval()
    for iteration, (depth_lr, rgb, depth_hr, depth_min, depth_max,depth_name) in enumerate(val_set_loader):

        if arg.cuda:
            rgb = rgb.cuda()
            depth_lr = depth_lr.cuda()
            depth_hr = depth_hr.cuda()
            depth_min = depth_min.cuda()
            depth_max = depth_max.cuda()

        _, _, H, W = depth_hr.shape
        depth_lr = torch.nn.functional.interpolate(depth_lr, size=(H, W), mode='bicubic')

        result = model(depth = depth_lr, rgb = rgb)
        pred = torch.clamp(result, 0, 1)
        pred = pred * (depth_max - depth_min) + depth_min
        depth_hr = depth_hr * (depth_max - depth_min) + depth_min

        if arg.dataset_name == "NYU":
            pred = pred[..., 6: -6, 6: -6]
            depth_hr = depth_hr[..., 6: -6, 6: -6]

        if arg.dataset_name == "NYU" or arg.dataset_name == "RGBDD":
            pred = pred * 100
            depth_hr = depth_hr * 100
            
        pred_aver = torch.mean(pred,dim=1).unsqueeze(1)
        
        RMSE = computeRMSE(pred, depth_hr)
        RMSE_aver = computeRMSE(pred_aver, depth_hr)
        SSIM = computeSSIM(pred, depth_hr,data_range=float((depth_hr.max()-depth_hr.min()).item()))
        
        RMSE_epoch = RMSE_epoch + RMSE
        RMSE_aver_epoch = RMSE_aver_epoch + RMSE_aver
        SSIM_epoch = SSIM_epoch + SSIM
                
        # save for vision
        print("===>save iteration:",iteration)
        save_path = args.save_path
        mkdir(save_path)
        np.save(f"{save_path}/NYU_{depth_name[0]}",pred.detach().cpu().numpy())
        
        if arg.dataset_name == "NYU" or arg.dataset_name == "RGBDD":
            if iteration % 50 == 0:
                print("VAL===> Val.RMSE: {:.4f}".format(RMSE))
                # print("VAL===> Val.SSIM: {:.4f}".format(SSIM))

    RMSE_valid = RMSE_epoch / (iteration + 1)
    RMSE_aver_valid = RMSE_aver_epoch / (iteration + 1)
    SSIM_epoch = SSIM_epoch / (iteration + 1)
    print("VAL===> Val_Avg.RMSE: {:.4f}".format(RMSE_valid))
    # print("VAL===> Val_Avg.SSIM: {:.4f}".format(SSIM_epoch))
    return  0



def main(arg):
    torch.manual_seed(arg.seed)

    cuda = arg.cuda
    if cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        torch.cuda.manual_seed(arg.seed)
    cudnn.benchmark = True

    print("===> Building model")
    arg.cuda = torch.cuda.is_available()
    
    if not args.scale == 16:
        model = MGFF_GDSR(GF_chans=30,nFeat=48,window_size=4,kernel_num=[1,1,1],embed_dim=30, depths=[3, 3, 3], num_heads=[6, 6, 6],flag=False)
    else:
        model = MGFF_GDSR(GF_chans=60,nFeat=60,window_size=4,kernel_num=[1,1,1],embed_dim=60, depths=[3, 3, 3], num_heads=[6, 6, 6],flag=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        # criterion = criterion.cuda()

    load_model_path = args.statedict_path
    if os.path.isfile(load_model_path):
        print("=> loading checkpoint '{}'".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))

    model_out_path = os.path.join(arg.model_path, arg.model_name)
    print("===> model_path", model_out_path)
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)

    Best = 1000
    Best_val = 1000
    total_iter = 0
    loss_epoch = 0
    RMSE_epoch = 0
    with torch.no_grad():
        RMSE_val = valid(arg, model)
        is_best_val = RMSE_val < Best_val
        Best_val = min(RMSE_val, Best_val)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
