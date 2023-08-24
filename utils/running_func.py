import os
import glob
import random
import numpy as np
import torch
import h5py
import time
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
except:
    from skimage.measure import compare_psnr
    from skimage.measure import compare_ssim
import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import io

print("check")
def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def batch_PSNR(img, imclean, data_range=1):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range=1):
    Img = img.clone().data.cpu().numpy().astype(np.float32)
    Iclean = imclean.clone().data.cpu().numpy().astype(np.float32)
    Img = Img.squeeze(0)
    Iclean = Iclean.squeeze(0)
    SSIM = 0
    SSIM += compare_ssim(Iclean[:,:,:], Img[:,:,:], data_range=data_range,channel_axis=0)
    return SSIM


def model_restore(model, trained_model_dir):
    model_list = glob.glob((trained_model_dir + "/*.pkl"))
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    epoch = np.sort(a)[-1]
    model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model, epoch


class data_loader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)
        
    def __getitem__(self, index):
        sample_path = self.list_txt[index][:-1]
        # print("===>sample_path",sample_path)
        
        if os.path.exists(sample_path):
            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
            crop_size = 128
            data, label = self.imageCrop(data, label, crop_size)
            data, label = self.image_Geometry_Aug(data, label)
            
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        # print("===>self.length",self.length)
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, label, crop_size):
        c, w, h = data.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...

        start_w = self.random_number(w_boder - 1)
        start_h = self.random_number(h_boder - 1)

        crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        return crop_data, crop_label

    def image_Geometry_Aug(self, data, label):
        c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:
            in_data = data
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, index]
            in_label = in_label[:, :, index]

        return in_data, in_label

class val_data_loader(data.Dataset):
    def __init__(self, list_dir,crop,patch_size):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)
        self.crop = crop
        self.patch_size = patch_size

        data = []
        label=[]
#        print("self.length",self.length)
        for i in range(self.length):
            sample_path = self.list_txt[i][:-1]
#            print("INIT:",sample_path)
            f = h5py.File(sample_path, 'r')

            data.append(f['IN'][:])
            label.append(f['GT'][:])
            f.close()

    def __getitem__(self, index):

        sample_path = self.list_txt[index][:-1]
        # print("sample_path",sample_path,index)

        if os.path.exists(sample_path):
            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
            data=data.transpose(0,2,1)
            label=label.transpose(0,2,1)
            return torch.from_numpy(data).float(), torch.from_numpy(label).float()
    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)


def get_lr(epoch, lr, max_epochs):
    epoch=int(epoch)
    max_epochs=int(max_epochs)
    if epoch <= 8000:
        lr = lr
    elif epoch <= 12000:
        lr = 0.5 * lr
    else:
        lr = 0.1 * lr
    return lr

def train(epoch, model, train_loaders, optimizer, opt):
    loss_type = str(opt["TrainSetting"]["Loss"])
    print("train_loss_type====>",loss_type)
    lr = get_lr(epoch,float(opt["TrainSetting"]["LR"]),int(opt["TrainSetting"]["Epochs"]))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    model.train()
    num = 0
    trainloss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loaders):
        
        data, target = data.cuda(), target.cuda()
        end = time.time()

############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data1, data2, data3)

#########  make the loss

        if loss_type == "mapped_L1":
            ## main loss
            output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
            target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
            loss = F.l1_loss(output, target)
            
        elif loss_type == "L1":
            ## linear L1 loss
            loss_linear = F.l1_loss(output, target)
            ## main loss
            output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
            target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
            loss_L1 = F.l1_loss(output, target)
            loss = loss_L1 + 0.1*loss_linear
            
        elif loss_type == "L2":
            ## linear L2 loss
            loss_mse = F.mse_loss(output, target)
            output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
            target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
            loss_L1 = F.l1_loss(output, target)
            loss = loss_L1 + 0.1*loss_mse
        else:
            raise Error
            
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        if (batch_idx +1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx, trainloss.data))
            fname = opt["Saveset"]["Result_dir"] + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss= 0
    #print(type(model),dir(model))
    if hasattr(model.module, 'update_temperature'):
        model.module.update_temperature()
    else:
        pass

def train_setting_changed(epoch, model, train_loaders, optimizer, args):
    lr = get_lr(epoch, args.lr, args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    model.train()
    num = 0
    trainloss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        end = time.time()

############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data1, data2, data3)

#########  make the loss
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        if (batch_idx +1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx, trainloss.data))
            fname = args.trained_model_dir + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss = 0

def val_code(epoch, model, val_loader, args, save_name):
    #f=open(r'/output/result.txt','a+')  #±£¥ÊŒ™txt
    model=model.eval()
    inum = 0
    psnr = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        inum = inum + 1
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        end = time.time()
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)
        #print("debug:",data1.shape)
        output = torch.Tensor(target.shape)
        with torch.no_grad():
            for h_axe in range(0,1000,500):
                for w_axe in range(0,1500,500):
                    input0=data1[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
                    input1=data2[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
                    input2=data3[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
 #                   print(input0.shape)
                    output[:,:,h_axe:h_axe+500,w_axe:w_axe+500] = model(input0, input1, input2)
        
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
         
        Psnr = batch_PSNR(output, target)
        print('Val Epoch {} Psnr: {:.6f}'.format(epoch, Psnr))
        psnr += Psnr

        psnr_pred = torch.squeeze(output.clone())
        psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)

        # if not os.path.exists(rf"/ghome/guanys/LW_Test/{save_name}"):
        #     os.mkdir(rf"/ghome/guanys/LW_Test/{save_name}")
        # io.imsave(os.path.join(rf"/ghome/guanys/LW_Test/{save_name}", str(inum).zfill(3) + f"_{save_name}.tif"), psnr_pred)
        print("saved")
    print(f"Val Epoch {epoch} Psnr",psnr/15.0)
    valres=psnr/15.0
    return valres
    #print(f"Val Epoch {epoch} Psnr",psnr/15.0,file=f)
    #f.close()

def val(epoch, model, val_loader):
    #f=open(r'/output/result.txt','a+')  #±£¥ÊŒ™txt
    model=model.eval()
    num = 0
    psnr_L = 0 
    ssim_L = 0
    psnr = 0
    ssim = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        
        data, target = data.cuda(), target.cuda()
        end = time.time()
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1).cuda()
        data2 = Variable(data2).cuda()
        data3 = Variable(data3).cuda()
        target = Variable(target).cuda()
        #print("debug:",data1.shape)
        # with torch.no_grad():
        #     output = model(data1, data2, data3)
        output = torch.Tensor(target.shape)
        with torch.no_grad():
            for h_axe in range(0,1000,500):
                for w_axe in range(0,1500,500):
                    input0=data1[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
                    input1=data2[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
                    input2=data3[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
                    output[:,:,h_axe:h_axe+500,w_axe:w_axe+500] = model(input0, input1, input2)
                    
        Psnr_linear = batch_PSNR(output, target)
        Ssim_linear = batch_SSIM(output, target)
        psnr_L += Psnr_linear
        ssim_L += Ssim_linear
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        Psnr = batch_PSNR(output, target)
        Ssim = batch_SSIM(output, target)
        psnr += Psnr
        ssim += Ssim
        
        print('Val Epoch {} Psnr_mu: {:.4f}|SSIM_mu: {:.4f}|Psnr_L: {:.4f}|SSIM_mu: {:.4f}'.format(epoch, Psnr,Ssim,Psnr_linear,Ssim_linear))
        #psnr_pred = torch.squeeze(output.clone())
        #psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)

        #if not os.path.exists(rf"/ghome/guanys/LW_Test/ours_{epoch}"):
        #    os.mkdir(rf"/ghome/guanys/LW_Test/ours_{epoch}") 
        #io.imsave(os.path.join(rf"/ghome/guanys/LW_Test/ours_{epoch}", str(inum).zfill(3) + "_ours.tif"), psnr_pred)
        #print("saved")
    print(f"*** Val Epoch {epoch} Result ", psnr/15.0,"\n",ssim/15.0,"\n",psnr_L/15.0,"\n",ssim_L/15.0)
    valres=psnr/15.0
    return valres

def val_simple(epoch, model, val_loader):
    #f=open(r'/output/result.txt','a+')  #±£¥ÊŒ™txt
    model=model.eval()
    num = 0
    psnr = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        
        data, target = data.cuda(), target.cuda()
        end = time.time()
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1).cuda()
        data2 = Variable(data2).cuda()
        data3 = Variable(data3).cuda()
        target = Variable(target).cuda()
        #print("debug:",data1.shape)
        with torch.no_grad():
            output = model(data1, data2, data3)
#        output = torch.Tensor(target.shape)
#        with torch.no_grad():
#            for h_axe in range(0,1000,500):
#                for w_axe in range(0,1500,500):
#                    input0=data1[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
#                    input1=data2[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
#                    input2=data3[:,:,h_axe:h_axe+500,w_axe:w_axe+500].cuda()
# #                   print(input0.shape)
#                    output[:,:,h_axe:h_axe+500,w_axe:w_axe+500] = model(input0, input1, input2)
        
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

        Psnr = batch_PSNR(output, target)
        print('Val Epoch {} Psnr: {:.6f}'.format(epoch, Psnr))
        psnr += Psnr

        #psnr_pred = torch.squeeze(output.clone())
        #psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)

        #if not os.path.exists(rf"/ghome/guanys/LW_Test/ours_{epoch}"):
        #    os.mkdir(rf"/ghome/guanys/LW_Test/ours_{epoch}") 
        #io.imsave(os.path.join(rf"/ghome/guanys/LW_Test/ours_{epoch}", str(inum).zfill(3) + "_ours.tif"), psnr_pred)
        #print("saved")
    print(f"*** Val Epoch {epoch} Psnr",psnr/15.0)
    valres=psnr/15.0
    return valres


def inference_simple(model, val_loader):
    model=model.eval()
    res_dict = {}
    output_list = []
    for batch_idx, (data, target) in enumerate(val_loader):
        
        data, target = data.cuda(), target.cuda()
        end = time.time()
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1).cuda()
        data2 = Variable(data2).cuda()
        data3 = Variable(data3).cuda()
        target = Variable(target).cuda()
        
        with torch.no_grad():
            output_list = model(data1, data2, data3)
            res_dict[f"{str(batch_idx).zfill(3)}"] = output_list
            
    assert len(res_dict.keys())==15
    return res_dict

