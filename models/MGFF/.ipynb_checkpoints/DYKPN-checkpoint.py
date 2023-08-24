import os
import sys
sys.path.append("/code/MEF/")
from MGFF.swinir_backbone import *
from MGFF.dykpn import DeepMuGIF as Odconv_Dynamic_Encoder
import torch.nn.functional as F
from MGFF.MaskedMI import Mutual_info_reg as MIloss
# from MGFF.MaskedMI import Normal_Mutual_info_reg as NMIloss

class MGFF_Blocks(nn.Module):

    def __init__(self, input_dim, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 kernel_num=4,inference=False,mid_channel=16,temperature=34,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.MGF = Odconv_Dynamic_Encoder()
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
                )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_embed_MGF = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        
        self.adjust_channel = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_01 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_02 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_03 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.MIloss = MIloss(dim,dim).cuda()

    def forward(self, feat_list):
        # Mutual Guided Fliter
        outlist = self.MGF(feat_list[0],feat_list[1])
        # MI loss

        MIloss = self.MIloss(outlist[0],outlist[1])
        if len(feat_list) == 3:
            MIloss += feat_list[2]
            MIloss = MIloss*0.5
    
        x = torch.cat((outlist[0],outlist[1]),dim=1)
        x = self.adjust_channel(x)
        # Fusion
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_MGF(x) # B L C
        res = self.residual_group(x, x_size) # B L C
        res = self.patch_unembed(res, x_size) # B c H W
        res = self.dilated_conv(res)
        # concat input
        out_out_1 =  self.adjust_channel_01(torch.cat((res,outlist[0]),dim=1))
        out_out_2 =  self.adjust_channel_02(torch.cat((res,outlist[1]),dim=1))

        return [out_out_1,out_out_2,MIloss]

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        flops += self.patch_embed_MGF.flops()
        return flops
    

class MGFF(nn.Module):
    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 kernel_num=4,inference=False,mid_channel=16,temperature=34,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(MGFF, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.inconv1 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=True)
        self.inconv2 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MGFF_Blocks(
                         input_dim=embed_dim,
                         dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         kernel_num=kernel_num[i_layer],
                         inference=inference,
                         mid_channel=mid_channel,
                         temperature=temperature,
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        
        self.conv_after_body = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, 3, 3, 1, 1),
            nn.Sigmoid()
            )          

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # def update_temperature(self):
    #     for m in self.modules():
    #         if isinstance(m, Odconv_Dynamic_Encoder):
    #             m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x[0].shape[2], x[0].shape[3])
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, under, over):
        
        _,_,H,W = under.shape
        
        under = self.check_image_size(under)
        over = self.check_image_size(over)
        
        under_feat = self.inconv1(under)
        over_feat = self.inconv2(over)
        
        x_list = [under_feat,over_feat]
        
        filter_list = self.forward_features(x_list)
        feat = torch.cat([filter_list[0],filter_list[1]],dim=1)
        
        res = self.conv_after_body(feat) + under_feat
        
        output = self.conv_last(res)
        
        return output[:,:,:H,:W],filter_list[2]
    
    def check_image_size(self, x):
        #print("[debug1,check_image_size]==>",x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #print("[debug2,check_image_size]==>",x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops


if __name__=="__main__":
    import argparse
    from thop.profile import profile
    import time

    model = MGFF(embed_dim=30, depths=[6, 6], num_heads=[6, 6],window_size=4, kernel_num = [1,1,1],mlp_ratio=2.,inference=False).cuda()
    
    height=128
    width=128
    
    rgb = torch.randn((1, 3, height, width)).cuda() 
    depth = torch.randn((1, 3, height, width)).cuda()
    name = "our"
    tic=time.time()
    total_ops, total_params = profile(model, (rgb,depth,))
    toc=time.time()
    print("===>time:",toc-tic)
    print("%s         | %.4f(M)      | %.4f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))