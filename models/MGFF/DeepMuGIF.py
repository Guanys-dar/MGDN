import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from einops import rearrange
import sys
print("DYKPN in model base")
sys.path.append("/code/retrain_MEF/")

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, act='LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1,inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)
    
class CrossTransAttention(nn.Module):
    def __init__(self,num_heads,dim):
        super(CrossTransAttention, self).__init__()
        self.num_heads = num_heads
        bias=True
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat_guide,feat_op):
        # feat_ref: Guidance
        # feat_ext: Value
        b,c,h,w = feat_guide.shape
        
        q = self.q_dwconv(self.q(feat_guide))
        kv = self.kv_dwconv(self.kv(feat_op))
        k,v = kv.chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class DeepMuGIF(nn.Module):
    def __init__(self,GF_chans):
        super(DeepMuGIF, self).__init__()
        
        self.kernel_width = 7
        self.channel = GF_chans
        self.kernel_dim = self.kernel_width*self.kernel_width
        self.pks = 3
        
        self.TCA = CrossTransAttention(num_heads=6,dim=self.channel)
        
        self.kernel_predictor_ref = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.kernel_predictor_ext = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.conv_out_1 = conv(self.channel, self.channel, kernel_size=1, stride=1)
        self.conv_out_2 = conv(self.channel, self.channel, kernel_size=1, stride=1)

    def FAC(self, feat_in, kernel, ksize):
        """
        customized FAC
        """
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        if channels ==3 and kernels == ksize*ksize:
            ####
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize)
            kernel = torch.cat([kernel,kernel,kernel],channels)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        else:
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize) 
            kernel = kernel.repeat(1,1,1, self.channel,1,1)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        return feat_out

    def kernelpredict_ref(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_ref(feat_kernel)
        return pre_kernel
    
    def kernelpredict_ext(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_ext(feat_kernel)
        return pre_kernel

    def forward(self,feat_1,feat_2):
        # feat_1,feat_2 = feat_list
        kernel_1 = self.kernelpredict_ref(feat_ref=feat_1,feat_ext=feat_2)
        kernel_2 = self.kernelpredict_ext(feat_ref=feat_2,feat_ext=feat_1)
        
        out_feat_1 = self.FAC(feat_1, kernel_1, self.kernel_width)
        out_feat_2 = self.FAC(feat_2, kernel_2, self.kernel_width)
        
        out_feat_1 = self.conv_out_1(out_feat_1)
        out_feat_2 = self.conv_out_2(out_feat_2)
        
        return [out_feat_1,out_feat_2]
    
class HDR_DeepMuGIF(nn.Module):
    def __init__(self,GF_chans=60):
        super(HDR_DeepMuGIF, self).__init__()
        print("HDR_DeepMuGIF in model base")
        self.kernel_width = 7
        self.channel = GF_chans
        self.kernel_dim = self.kernel_width*self.kernel_width
        self.pks = 3
        
        self.TCA = CrossTransAttention(num_heads=6,dim=self.channel)
                
        self.kernel_predictor_under = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.kernel_predictor_over = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.kernel_dim, kernel_size=1, stride=1,act=None)
        )
        
        self.conv_out_1 = conv(self.channel, self.channel, kernel_size=1, stride=1)
        self.conv_out_2 = conv(self.channel, self.channel, kernel_size=1, stride=1)
        
        self.conv_ref = nn.Sequential(
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
            conv(self.channel, self.channel, kernel_size=self.pks, stride=1),
        )

    def FAC(self, feat_in, kernel, ksize):
        """
        customized FAC
        """
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        if channels ==3 and kernels == ksize*ksize:
            ####
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize)
            kernel = torch.cat([kernel,kernel,kernel],channels)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        else:
            kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize) 
            kernel = kernel.repeat(1,1,1, self.channel,1,1)
            kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        return feat_out

    def kernelpredict_under(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_under(feat_kernel)
        return pre_kernel
    
    def kernelpredict_over(self, feat_ref, feat_ext):
        feat_kernel = self.TCA(feat_guide=feat_ref,feat_op=feat_ext)
        pre_kernel = self.kernel_predictor_over(feat_kernel)
        return pre_kernel


    def forward(self,feat_1,feat_2,feat_3):
        # feat_1,feat_2,feat_3 = feat_list
        
        kernel_1 = self.kernelpredict_under(feat_ref=feat_2,feat_ext=feat_1)
        kernel_3 = self.kernelpredict_over(feat_ref=feat_2,feat_ext=feat_3)
        
        out_feat_1 = self.FAC(feat_1, kernel_1, self.kernel_width)
        out_feat_3 = self.FAC(feat_3, kernel_3, self.kernel_width)
        
        out_feat_1 = self.conv_out_1(out_feat_1)
        out_feat_3 = self.conv_out_2(out_feat_3)
        
        out_feat_2 = self.conv_ref(feat_2)
        
        return [out_feat_1,out_feat_2,out_feat_3]
    
if __name__ == '__main__':
    import sys
    model = DeepMuGIF()

    import logging
    import sys
    logger = logging.getLogger("demo")
    from test_utils.model_summary import get_model_flops, get_model_activation

    input_dim = (60, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    # logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    # logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    # logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
