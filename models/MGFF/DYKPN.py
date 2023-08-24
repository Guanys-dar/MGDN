import os
import sys

sys.path.append("../MGFF")
from MGFF.swinir_backbone import *
from MGFF.DeepMuGIF import DeepMuGIF as MuGIF
from MGFF.DeepMuGIF import HDR_DeepMuGIF as HDR_MuGIF
import torch.nn.functional as F


class MGFF_Blocks(nn.Module):

    def __init__(self, GF_chans, input_dim, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 kernel_num=4, inference=False, mid_channel=16, temperature=34,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.MGF = MuGIF(GF_chans)
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

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_embed_MGF = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.adjust_channel = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_01 = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_02 = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_03 = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0, bias=True)

    def forward(self, feat_list):
        # Mutual Guided Fliter
        outlist = self.MGF(feat_list[0], feat_list[1])
        x = torch.cat((outlist[0], outlist[1]), dim=1)
        x = self.adjust_channel(x)
        # Fusion
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_MGF(x)  # B L C
        res = self.residual_group(x, x_size)  # B L C
        res = self.patch_unembed(res, x_size)  # B c H W
        res = self.dilated_conv(res)
        # concat input
        out_out_1 = self.adjust_channel_01(torch.cat((res, outlist[0]), dim=1))
        out_out_2 = self.adjust_channel_02(torch.cat((res, outlist[1]), dim=1))
        return [out_out_1, out_out_2]

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        flops += self.patch_embed_MGF.flops()
        return flops
    
class MGFF_HDR_Blocks(nn.Module):

    def __init__(self, GF_chans, input_dim, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 kernel_num=4,inference=False,mid_channel=16,temperature=34,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.MGF = HDR_MuGIF(GF_chans=60)
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

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_embed_MGF = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        
        self.adjust_channel = nn.Conv2d(3*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_01 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_02 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)
        self.adjust_channel_03 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0, bias=True)

    def forward(self, feat_list):
        # Mutual Guided Fliter
        outlist = self.MGF(feat_list[0],feat_list[1],feat_list[2])
        x = torch.cat((outlist[0],outlist[1],outlist[2]),dim=1)
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
        out_out_3 =  self.adjust_channel_03(torch.cat((res,outlist[2]),dim=1))
        return [out_out_1,out_out_2,out_out_3]

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
    def __init__(self, GF_chans=30, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 kernel_num=4, inference=False, mid_channel=16, temperature=34,
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
        self.GF_chans = GF_chans

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
                GF_chans=self.GF_chans,
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
            nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1),
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
    #         if isinstance(m, MuGIF):
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

        _, _, H, W = under.shape

        under = self.check_image_size(under)
        over = self.check_image_size(over)

        under_feat = self.inconv1(under)
        over_feat = self.inconv2(over)

        x_list = [under_feat, over_feat]

        feat = torch.cat(self.forward_features(x_list), dim=1)

        res = self.conv_after_body(feat) + under_feat

        output = self.conv_last(res)

        return output[:, :, :H, :W]

    def check_image_size(self, x):
        # print("[debug1,check_image_size]==>",x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print("[debug2,check_image_size]==>",x.shape)
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


class MGFF_GDSR(nn.Module):
    def __init__(self, GF_chans=30, img_size=128, patch_size=1, in_chans=3,
                embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                kernel_num=4,inference=False,mid_channel=16,temperature=34,
                use_checkpoint=False, resi_connection='1conv',
                **kwargs):
        super(MGFF_GDSR, self).__init__()
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

        # split image into non-depthlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-depthlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.inconv1 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=True)
        self.inconv2 = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MGFF_Blocks(
                        GF_chans = GF_chans,
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
            nn.Conv2d(embed_dim, 1, 3, 1, 1),
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
    #         if isinstance(m, MuGIF):
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

    def forward(self, rgb, depth):
        
        _,_,H,W = rgb.shape
        
        rgb = self.check_image_size(rgb)
        depth = self.check_image_size(depth)
        
        rgb_feat = self.inconv1(rgb)
        depth_feat = self.inconv2(depth)
        
        x_list = [rgb_feat,depth_feat]
        
        feat = torch.cat(self.forward_features(x_list),dim=1)
        
        res = self.conv_after_body(feat) + depth_feat
        
        output = self.conv_last(res)
        
        return output[:,:,:H,:W]
    
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


class MGFF_Deghosting(nn.Module):
    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=48, depths=[3, 3], num_heads=[6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 kernel_num=4,inference=False,mid_channel=16,temperature=34,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(MGFF_Deghosting, self).__init__()
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
        self.conv1 = nn.Conv2d(6, embed_dim, kernel_size=3, padding=1, bias=True)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MGFF_HDR_Blocks(
                         GF_chans=60,
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
            nn.Conv2d(embed_dim*3, embed_dim, 3, 1, 1),
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

    def forward(self, under, ref, over):
        
        _,_,H,W = ref.shape
        
        under = self.check_image_size(under)
        ref = self.check_image_size(ref)
        over = self.check_image_size(over)
        
        under_feat = self.conv1(under)
        ref_feat = self.conv1(ref)
        over_feat = self.conv1(over)
        
        x_list = [under_feat,ref_feat,over_feat]
        
        feat = torch.cat(self.forward_features(x_list),dim=1)
        
        res = self.conv_after_body(feat)+ref_feat
        
        output = self.conv_last(res)
        
        # print("===>",output[:,:,:H,:W].shape)
        return output[:,:,:H,:W]
    
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


if __name__ == "__main__":
    import argparse
    from thop.profile import profile
    import time

    model = MGFF(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], window_size=8, kernel_num=[1, 1, 1], mlp_ratio=4.,
                 inference=False).cuda()
    height = 256
    width = 256
    rgb = torch.randn((1, 3, height, width)).cuda()
    depth = torch.randn((1, 3, height, width)).cuda()

    name = "our"
    tic = time.time()
    total_ops, total_params = profile(model, (depth, rgb,))
    toc = time.time()
    print("===>time:", toc - tic)
    print("%s         | %.4f(M)      | %.4f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))