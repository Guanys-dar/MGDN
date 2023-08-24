# -*- coding: utf-8 -*-

from __future__ import division

import torch
import torch.nn as nn
import logging
from scipy.io import loadmat,savemat

from PIL import Image, ImageOps
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from skimage.io import imread, imsave

from torchvision.transforms import Compose
from torchvision import transforms
import random
import numpy as np
import h5py
# import cv2
from skimage.io import imread, imsave


def get_patch(depth, rgb, patch_size):
    ih, iw = depth.shape
    tp = patch_size  # target_patch_size

    tx = random.randrange(0, iw - tp + 1)
    ty = random.randrange(0, ih - tp + 1)

    depth_patch  = depth[ty:ty + tp, tx: tx + tp]
    rgb_patch    = rgb[:, ty:ty + tp, tx: tx + tp]

    return depth_patch, rgb_patch


def augment(depth, rgb):

    if random.random() < 0.5:
        # Random vertical Flip
        rgb = rgb[:, :, ::-1].copy()
        depth = depth[:, ::-1].copy()

    if random.random() < 0.5:
        # Random horizontal Flip
        rgb = rgb[:, ::-1, :].copy()
        depth = depth[::-1, :].copy()

    # if random.random() < 0.5:
    #     # Random rotation
    #     rgb = np.rot90(rgb.copy(), axes=(1, 2))
    #     depth = np.rot90(depth.copy(), axes=(0, 1))

    return depth, rgb


def get_patch_triple(rgb, depth_lr, depth_hr, patch_size):
    ih, iw = depth_hr.shape
    tp = patch_size  # target_patch_size

    tx = random.randrange(0, iw - tp + 1)
    ty = random.randrange(0, ih - tp + 1)

    depth_lr_patch = depth_lr[ty:ty + tp, tx: tx + tp]
    depth_hr_patch = depth_hr[ty:ty + tp, tx: tx + tp]
    rgb_patch = rgb[:, ty:ty + tp, tx: tx + tp]

    return rgb_patch, depth_lr_patch, depth_hr_patch



def augment_triple(rgb, depth_lr, depth_hr):

    if random.random() < 0.5:
        # Random vertical Flip
        rgb = rgb[:, :, ::-1].copy()
        depth_lr = depth_lr[:, ::-1].copy()
        depth_hr = depth_hr[:, ::-1].copy()

    if random.random() < 0.5:
        # Random horizontal Flip
        rgb = rgb[:, ::-1, :].copy()
        depth_lr = depth_lr[::-1, :].copy()
        depth_hr = depth_hr[::-1, :].copy()

    # if random.random() < 0.5:
    #     # Random rotation
    #     rgb = np.rot90(rgb.copy(), axes=(1, 2))
    #     depth_lr = np.rot90(depth_lr.copy(), axes=(1, 2))
    #     depth_hr = np.rot90(depth_hr.copy(), axes=(1, 2))

    return rgb, depth_lr, depth_hr

class Train_NYU(Dataset):
    def __init__(self, args):
        super(Train_NYU, self).__init__()

        self.depth_path = args.depth_path
        self.depth_names = os.listdir(self.depth_path)

        self.rgb_path = args.rgb_path

        self.scale = args.scale
        self.patch_size = args.patch_size # HR patch size
        self.data_augmentation = args.augmentation

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_name = os.path.join(self.depth_path, self.depth_names[index])
        depth = np.load(depth_name) # [H, W]

        rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-4]+".jpg")
        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0        # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # crop patch
        depth_patch, rgb_patch = get_patch(depth, rgb, self.patch_size)

        # data augmentation
        if self.data_augmentation:
            depth_patch, rgb_patch = augment(depth_patch, rgb_patch)

        # normalize
        depth_min = depth_patch.min()
        depth_max = depth_patch.max()
        depth_patch = (depth_patch - depth_min) / (depth_max - depth_min)
        rgb_patch = (rgb_patch - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # depth resize
        h, w = depth_patch.shape[:2]
        depth_patch_lr = np.array(
            Image.fromarray(depth_patch).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        # to tensor
        depth_patch = torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
        depth_patch_lr = torch.from_numpy(depth_patch_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous() # [3, H, W]

        return depth_patch_lr, rgb_patch, depth_patch
    
class Get_NYU_Traindata(Dataset):
    def __init__(self, args):
        super(Get_NYU_Traindata, self).__init__()

        self.depth_path = args.depth_path
        self.depth_names = os.listdir(self.depth_path)

        self.rgb_path = args.rgb_path

        self.scale = args.scale
        self.patch_size = args.patch_size # HR patch size
        self.data_augmentation = args.augmentation

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_name = os.path.join(self.depth_path, self.depth_names[index])
        depth = np.load(depth_name) # [H, W]

        rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-4]+".jpg")
        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0        # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # crop patch
        depth_patch, rgb_patch = get_patch(depth, rgb, self.patch_size)

#         # data augmentation
#         if self.data_augmentation:
#             depth_patch, rgb_patch = augment(depth_patch, rgb_patch)

        # normalize
        depth_min = depth_patch.min()
        depth_max = depth_patch.max()
        depth_patch = (depth_patch - depth_min) / (depth_max - depth_min)
        rgb_patch = (rgb_patch - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # depth resize
        h, w = depth_patch.shape[:2]
        depth_patch_lr = np.array(
            Image.fromarray(depth_patch).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        # # to tensor
        # depth_patch = torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
        # depth_patch_lr = torch.from_numpy(depth_patch_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        # rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous() # [3, H, W]

        return depth_patch_lr, rgb_patch, depth_patch


class ValidLoader(Dataset):
    def __init__(self, args):
        super(ValidLoader, self).__init__()
        self.dataset_name = args.dataset_name

        self.depth_path = args.depth_path_valid
        self.depth_names = os.listdir(self.depth_path)

        self.rgb_path = args.rgb_path_valid

        self.scale = args.scale

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_name = os.path.join(self.depth_path, self.depth_names[index])
        if "NYU" in self.dataset_name:
            depth = np.load(depth_name)  # [H, W]
        elif "Middlebury" in self.dataset_name:
            depth = imread(depth_name).astype(np.float32)  # [H, W]
        elif "Lu" in self.dataset_name:
            depth = imread(depth_name).astype(np.float32)  # [H, W]

        if "NYU" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-4] + ".jpg")
        elif "Middlebury" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "color.png")  # depth.png
        elif "Lu" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "color.png")  # depth.png

        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0  # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, H, W]


        # normalize
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)
        rgb = (rgb - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # depth resize
        h, w = depth.shape[:2]
        depth_lr = np.array(
            Image.fromarray(depth).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        # to tensor
        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
        depth_lr = torch.from_numpy(depth_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        rgb = torch.from_numpy(rgb.astype(np.float32)).contiguous() # [3, H, W]

        return depth_lr, rgb, depth, depth_min, depth_max,self.depth_names[index]


# class Train_RGBDD(Dataset):
#     def __init__(self, args):
#         super(Train_RGBDD, self).__init__()
#
#         self.depth_path = args.depth_path
#         self.depth_names = os.listdir(self.depth_path)
#
#         self.rgb_path = args.rgb_path
#
#         self.scale = args.scale
#         self.patch_size = args.patch_size # HR patch size
#         self.data_augmentation = args.augmentation
#
#     def __len__(self):
#         return len(self.depth_names)
#
#     def __getitem__(self, index):
#         depth_name = os.path.join(self.depth_path, self.depth_names[index])
#         depth = imread(depth_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)
#
#         rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "RGB.jpg")
#         rgb = imread(rgb_name)
#         rgb = rgb.astype(np.float32) / 255.0        # [H, W, 3]
#         rgb = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]
#
#         # crop patch
#         depth_patch, rgb_patch = get_patch(depth, rgb, self.patch_size)
#
#         # data augmentation
#         if self.data_augmentation:
#             depth_patch, rgb_patch = augment(depth_patch, rgb_patch)
#
#         # depth resize
#         h, w = depth_patch.shape[:2]
#         depth_patch_lr = np.array(
#             Image.fromarray(depth_patch).resize((w // self.scale, h // self.scale), Image.BICUBIC))
#
#         # normalize
#         # depth_patch = (depth_patch - depth_patch.min()) / (depth_patch.max() - depth_patch.min())
#         depth_patch_lr = (depth_patch_lr - depth_patch.min()) / (depth_patch.max() - depth_patch.min())
#         rgb_patch = (rgb_patch - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
#             [0.229, 0.224, 0.225]).reshape(3, 1, 1)
#
#         # to tensor
#         depth_patch = torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
#         depth_patch_lr = torch.from_numpy(depth_patch_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
#         rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous() # [3, H, W]
#
#         # follow DCTNet
#         depth_patch = depth_patch * 100
#         depth_min = depth_patch.min()
#         depth_max = depth_patch.max()
#
#         return depth_patch_lr, rgb_patch, depth_patch
#
#
# class Valid_RGBDD(Dataset):
#     def __init__(self, args):
#         super(Valid_RGBDD, self).__init__()
#         self.dataset_name = args.dataset_name
#
#         self.depth_path = args.depth_path_valid
#         self.depth_names = os.listdir(self.depth_path)
#
#         self.rgb_path = args.rgb_path_valid
#
#         self.scale = args.scale
#
#     def __len__(self):
#         return len(self.depth_names)
#
#     def __getitem__(self, index):
#         depth_name = os.path.join(self.depth_path, self.depth_names[index])
#         depth = imread(depth_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)
#
#         rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "RGB.jpg")
#         rgb = imread(rgb_name)
#         rgb = rgb.astype(np.float32) / 255.0  # [H, W, 3]
#         rgb = np.transpose(rgb, [2, 0, 1])  # [3, H, W]
#
#         # crop edge
#         h, w = rgb.shape[1:]
#         scale = 4
#         h = h - h % scale
#         w = w - w % scale
#         rgb = rgb[:, :h, :w]
#         depth = depth[:h, :w]
#
#         # depth resize
#         h, w = depth.shape[:2]
#         depth_lr = np.array(
#             Image.fromarray(depth).resize((w // self.scale, h // self.scale), Image.BICUBIC))
#
#         # normalize
#         # depth = (depth - depth.min()) / (depth.max() - depth.min())
#         depth_lr = (depth_lr - depth.min()) / (depth.max() - depth.min())
#         rgb = (rgb - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
#             [0.229, 0.224, 0.225]).reshape(3, 1, 1)
#
#         # to tensor
#         depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
#         depth_lr = torch.from_numpy(depth_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
#         rgb = torch.from_numpy(rgb.astype(np.float32)).contiguous() # [3, H, W]
#
#         # follow DCTNet
#         depth = depth * 100
#         depth_min = depth.min()
#         depth_max = depth.max()
#
#         return depth_lr, rgb, depth, depth_min, depth_max


class Train_RGBDD_real(Dataset):
    def __init__(self, args):
        super(Train_RGBDD_real, self).__init__()

        self.depth_hr_path = args.depth_hr_path
        self.depth_lr_path = args.depth_lr_path
        self.rgb_path = args.rgb_path

        self.depth_names = os.listdir(self.depth_hr_path)
        self.patch_size = args.patch_size # HR patch size
        self.data_augmentation = args.augmentation
        self.crop_patch = args.crop_patch

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_hr_name = os.path.join(self.depth_hr_path, self.depth_names[index]) # HR_gt.png
        depth_hr = imread(depth_hr_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)
        
        rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9]+"RGB.jpg")
        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0        # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        depth_lr_name = os.path.join(self.depth_lr_path, self.depth_names[index][:-9]+"LR_fill_depth.png")
        depth_lr = imread(depth_lr_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)

        # Resize
        h, w = rgb.shape[1:]
        depth_lr = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))

        # crop edge
        h, w = rgb.shape[1:]
        scale = 4
        h = h - h % scale
        w = w - w % scale
        rgb = rgb[:, :h, :w]
        depth_hr = depth_hr[:h, :w]
        depth_lr = depth_lr[:h, :w]

        # normalize
        # depth_hr = (depth_hr - depth_hr.min()) / (depth_hr.max() - depth_hr.min())
        depth_lr = (depth_lr - depth_hr.min()) / (depth_hr.max() - depth_hr.min())
        rgb = (rgb - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # crop patch
        if self.crop_patch:
            rgb_patch, depth_lr_patch, depth_hr_patch = get_patch_triple(rgb, depth_lr, depth_hr, self.patch_size)
        else:
            rgb_patch, depth_lr_patch, depth_hr_patch = rgb, depth_lr, depth_hr

        # data augmentation
        if self.data_augmentation:
            rgb_patch, depth_lr_patch, depth_hr_patch = augment_triple(rgb_patch, depth_lr_patch, depth_hr_patch)

        # to tensor
        depth_hr_patch = torch.from_numpy(depth_hr_patch.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        depth_lr_patch = torch.from_numpy(depth_lr_patch.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous()  # [3, H, W]

        # follow DCTNet
        depth_hr_patch = depth_hr_patch * 100
        depth_min = depth_hr_patch.min().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        depth_max = depth_hr_patch.max().unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return depth_lr_patch, rgb_patch, depth_hr_patch,  depth_min, depth_max


class Valid_RGBDD_real(Dataset):
    def __init__(self, args):
        super(Valid_RGBDD_real, self).__init__()
        self.depth_hr_path = args.depth_hr_path_valid
        self.depth_lr_path = args.depth_lr_path_valid
        self.depth_names = os.listdir(self.depth_hr_path)

        self.rgb_path = args.rgb_path_valid

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_hr_name = os.path.join(self.depth_hr_path, self.depth_names[index]) # HR_gt.png
        depth_hr = imread(depth_hr_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)

        depth_lr_name = os.path.join(self.depth_lr_path, self.depth_names[index][:-9]+"LR_fill_depth.png")
        depth_lr = imread(depth_lr_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)

        rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9]+"RGB.jpg")
        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0        # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # Resize
        h, w = rgb.shape[1:]
        depth_lr = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))

        # crop edge
        h, w = rgb.shape[1:]
        scale = 4
        h = h - h % scale
        w = w - w % scale
        rgb = rgb[:, :h, :w]
        depth_hr = depth_hr[:h, :w]
        depth_lr = depth_lr[:h, :w]

        # normalize
        # depth_hr = (depth_hr - depth_hr.min()) / (depth_hr.max() - depth_hr.min())
        depth_lr = (depth_lr - depth_hr.min()) / (depth_hr.max() - depth_hr.min())
        rgb = (rgb - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # to tensor
        depth_hr = torch.from_numpy(depth_hr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        depth_lr = torch.from_numpy(depth_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        rgb = torch.from_numpy(rgb.astype(np.float32)).contiguous() # [3, H, W]

        # follow DCTNet
        depth_hr = depth_hr * 100
        depth_min = depth_hr.min()
        depth_max = depth_hr.max()

        return depth_lr, rgb, depth_hr, depth_min, depth_max


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="HSI Rec")
    # data loader
    # parser.add_argument("--depth_path", type=str,
    #                     default='/data/depthSR/RawDatasets/NYUDepthv2_Train/Depth',
    #                     help="HyperSet path")
    #
    # parser.add_argument("--rgb_path", type=str,
    #                     default='/data/depthSR/RawDatasets/NYUDepthv2_Train/RGB',
    #                     help="RGBSet path")
    # #
    # parser.add_argument('--dataset_name', type=str, default='NYU', help='test dataset name')
    # parser.add_argument("--depth_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/NYUDepthv2_Test/Depth',
    #                     help="HyperSet path")
    # parser.add_argument("--rgb_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/NYUDepthv2_Test/RGB',
    #                     help="RGBSet path")

    # parser.add_argument('--dataset_name', type=str, default='Middlebury', help='test dataset name')
    # parser.add_argument("--depth_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/Middlebury/Depth',
    #                     help="HyperSet path")
    # parser.add_argument("--rgb_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/Middlebury/RGB',
    #                     help="RGBSet path")

    # parser.add_argument('--dataset_name', type=str, default='Lu', help='test dataset name')
    # parser.add_argument("--depth_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/Lu/Depth',
    #                     help="HyperSet path")
    # parser.add_argument("--rgb_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/Lu/RGB',
    #                     help="RGBSet path")

    # parser.add_argument('--dataset_name', type=str, default='RGBDD', help='test dataset name')
    # parser.add_argument("--depth_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/RGBDD/Depth',
    #                     help="HyperSet path")
    # parser.add_argument("--rgb_path_valid", type=str,
    #                     default='/data/depthSR/RawDatasets/RGBDD/RGB',
    #                     help="RGBSet path")


    # parser.add_argument('--dataset_name', type=str, default='RGBDD', help='test dataset name')

    parser.add_argument("--depth_hr_path", type=str,
                        default='/data/depthSR/RawDatasets/RGBDD_Train_Realscene/Depth',
                        help="HyperSet path")
    parser.add_argument("--depth_lr_path", type=str,
                        default='/data/depthSR/RawDatasets/RGBDD_Train_Realscene/DepthLR',
                        help="HyperSet path")
    parser.add_argument("--rgb_path", type=str,
                        default='/data/depthSR/RawDatasets/RGBDD_Train_Realscene/RGB',
                        help="RGBSet path")

    parser.add_argument("--depth_hr_path_valid", type=str,
                        default='/data/depthSR/RawDatasets/RGBDD_Test_Realscene/Depth',
                        help="HyperSet path")
    parser.add_argument("--depth_lr_path_valid", type=str,
                        default='/data/depthSR/RawDatasets/RGBDD_Test_Realscene/DepthLR',
                        help="HyperSet path")
    parser.add_argument("--rgb_path_valid", type=str,
                        default='/data/depthSR/RawDatasets/RGBDD_Test_Realscene/RGB',
                        help="RGBSet path")

    parser.add_argument('--crop_patch', type=bool, default=True, help='crop_patch')
    parser.add_argument('--augmentation', type=bool, default=True, help='augmentation')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--scale', type=int, default=4, help='patch_size')

    arg = parser.parse_args()
    #
    # train_set = Train_NYU(arg)
    # valid_set = ValidLoader(arg)
    train_set = Train_RGBDD_real(arg)
    valid_set = Valid_RGBDD_real(arg)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=4, shuffle=False)
    valid_data_loader = DataLoader(dataset=valid_set, num_workers=0, batch_size=1, shuffle=False)

    for i, (depth_lr, rgb, depth_hr, depth_min, depth_max) in enumerate(training_data_loader):
    # for i, (depth_lr, rgb, depth_hr, depth_min, depth_max)  in enumerate(valid_data_loader):

        if i % 2 == 0:
            print(i)
            print(depth_lr.shape, torch.max(depth_lr), torch.min(depth_lr))
            print(rgb.shape, torch.max(rgb), torch.min(rgb))
            print(depth_hr.shape, torch.max(depth_hr), torch.min(depth_hr))
            print(depth_min.shape, depth_min)
            print(depth_max.shape, depth_max)
            print((depth_lr * (depth_max - depth_min) + depth_min).min())