# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------

import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.color import rgb2ycbcr
from skimage.io import imread
from scipy.io import loadmat
try:
    from skimage.metrics import structural_similarity as compare_ssim
except:
    from skimage.measure import compare_ssim
import torch.utils.data as Data
import numpy as np
from glob import glob
import torch
import h5py
import os
import json


def computeRMSE(imageA, imageB):
    # img1, img2: [0, 255]
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    imageA = imageA.cpu().detach().numpy()
    imageB = imageB.cpu().detach().numpy()

    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return np.sqrt(err)

def computeSSIM(im1, im2, data_range):
    img1 = im1.squeeze(0).squeeze(0).cpu()
    img2 = im2.squeeze(0).squeeze(0).cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()

    s = compare_ssim(img1_np, img2_np, gaussian_weights=True, sigma=1.5,
                              use_sample_covariance=False, data_range=data_range)
    return s


def computeRMSE_numpy(imageA, imageB):
    # img1, img2: [0, 255]
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"

    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return np.sqrt(err)


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def save_param(input_dict, path):
    f = open(path, 'w')
    f.write(json.dumps(input_dict))
    f.close()
    print("Hyper-Parameters have been saved!")


# ----------------------------------------------------------------------------
# Dataset & Image Processing
# ----------------------------------------------------------------------------


def normlization(x):
    # x [N,C,H,W]
    N, C, H, W = x.shape
    m = []
    for i in range(N):
        m.append(torch.max(x[i, :, :, :]))
    m = torch.stack(m, dim=0)[:, None, None, None]
    m = m+1e-10
    x = x/m
    return x, m


def inverse_normlization(x, m):
    return x*m


def im2double(img):
    if img.dtype == 'uint8':
        img = img.astype(np.float32)/255.
    elif img.dtype == 'uint16':
        img = img.astype(np.float32)/65535.
    else:
        img = img.astype(np.float32)
    return img


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def imresize(img, size=None, scale_factor=None):
    # img (np.array) - [C,H,W]
    imgT = torch.from_numpy(img).unsqueeze(0)  # [1,C,H,W]
    if size is None and scale_factor is not None:
        imgT = torch.nn.functional.interpolate(imgT,
                                               scale_factor=scale_factor,
                                               mode='bicubic')
    elif size is not None and scale_factor is None:
        imgT = torch.nn.functional.interpolate(imgT,
                                               size=size,
                                               mode='bicubic')
    else:
        print('Neither size nor scale_factor is given.')
    imgT = imgT.squeeze(0).numpy()
    return imgT


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * \
        0.587000 + img[2:3, :, :] * 0.114000
    return y

def output_img(x):
    return x.cpu().detach().numpy()[0, 0, :, :]
