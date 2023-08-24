import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
print("shared_dyconv in model base")

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,temperature=34):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        assert self.kernel_num == 4
        self.temperature= temperature
        
#         print("Attention temperature",self.temperature)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        #print("debug",x.shape)
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        #print("before view===>",self.channel_fc(x).shape)
        #print("after view===>",channel_attention.shape)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        #print("after view===>",spatial_attention.shape)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        #print("forward====>",x.shape)
        x = self.avgpool(x)
        x = self.fc(x)
        #print(x.shape)
        #x = self.bn(x)
        x = self.relu(x)
        #print("forward2====>",x.shape)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class Mutual_Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,temperature=34):
#    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=32):
        super(Mutual_Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature= temperature

        # assert self.kernel_num == 4
        
#         print("Mutual_Attention temperature",self.temperature)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes//2, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        #print("debug",x.shape)
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        #print("debug",channel_attention.shape)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        #print(x.shape)
        x = self.avgpool(x)
        x = self.fc(x)
        #print(x.shape)
        #x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)



class Mutual_CrossAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,temperature=34):
        super(Mutual_CrossAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.attention_channel = attention_channel
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature= temperature

        self.avgpool = nn.AdaptiveAvgPool2d(1)
#         assert in_planes == 6
        self.map_ref_fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.map_nonref_fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.map_cat_fc = nn.Conv2d(2*in_planes, attention_channel, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        self.fc = nn.Conv2d(attention_channel, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    @staticmethod
    def skip(_):
        return 1.0
    
    def vector_cross_attention(self,v_ref,v_nonref):
        # print("vector_cross_attention input==>",v_ref.shape,v_nonref.shape)
        B_, C, H, W = v_ref.shape
        
        q = self.map_ref_fc(v_ref)
        k = self.map_nonref_fc(v_nonref)
        v = self.map_cat_fc(torch.cat([v_ref,v_nonref],dim=1))
        # print("q k v shape==>",q.shape,k.shape,v.shape)
        
        attn = (q @ k.transpose(-2, -1))
        # print("attn shape==>",attn.shape)
        attn = self.softmax(attn)
        vertor = (attn @ v).transpose(1, 2).reshape(B_, self.attention_channel, H, W)
        return vertor

    def get_channel_attention(self, x):
        #print("debug",x.shape)
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        # print("channel_attention==>",channel_attention.shape)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        # print("filter_attention==>",filter_attention.shape)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        # print("spatial_attention==>",spatial_attention.shape)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        # print("kernel_attention==>",kernel_attention.shape)
        return kernel_attention

    def forward(self, x_ref, x_nonref):
        v_ref = self.avgpool(x_ref)
        v_nonref = self.avgpool(x_nonref)
        v_x = self.vector_cross_attention(v_ref=v_ref,v_nonref=v_nonref)
        x = self.relu(self.bn(self.fc(v_x)))
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=3, padding=0, dilation=1, groups=1,reduction=0.0625, kernel_num=4,temperature=34):
#    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,reduction=0.25, kernel_num=8):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num,temperature = temperature)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self):
        #print("OD UPT")
        self.attention.update_temperature()

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)
        

class Mutual_ODEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,reduction=0.0625, kernel_num=4,temperature = 34, inference=False):
        super(Mutual_ODEncoder, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        # assert kernel_num == 4, f"Wrong kernel number: {kernel_num}, should be 4"

        if not inference:
            temperature = 34
        else:
            temperature = 1
        # assert temperature == 1, f"Wrong temperature: {temperature}"
        self.over_attention = Mutual_Attention(2*in_planes, out_planes, kernel_size, groups=groups,reduction=reduction, kernel_num=kernel_num,temperature = temperature)
        self.ref_attention = Mutual_Attention(2*in_planes, out_planes, kernel_size, groups=groups,reduction=reduction, kernel_num=kernel_num,temperature = temperature)
        self.under_attention = Mutual_Attention(2*in_planes, out_planes, kernel_size, groups=groups,reduction=reduction, kernel_num=kernel_num,temperature = temperature)
        self.shared_weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),requires_grad=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.shared_weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self):
        self.over_attention.update_temperature()
        self.ref_attention.update_temperature()
        self.under_attention.update_temperature()

    def _forward_impl_common(self, x_ref,x_over,x_under):
        
        feat_under = x_under.clone()
        feat_over = x_over.clone()
        
        # under branch
        r1=torch.cat([feat_ref,feat_under],dim=1)
        channel_attention_01, filter_attention_01, spatial_attention_01, kernel_attention_01= self.under_attention(r1)
        batch_size, in_planes, height, width = x_under.size()
        x_under = x_under * channel_attention_01
        x_under = x_under.reshape(1, -1, height, width)
        aggregate_weight_under = spatial_attention_01 * kernel_attention_01 * self.shared_weight.unsqueeze(dim=0)
        aggregate_weight_under = torch.sum(aggregate_weight_under, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output_under = F.conv2d(x_under, weight=aggregate_weight_under, bias=None, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups * batch_size)
        output_under = output_under.view(batch_size, self.out_planes, output_under.size(-2), output_under.size(-1))
        output_under = output_under * filter_attention_01
        # skip connection
        # output_under = short_cut_under + output_under
        
        # over branch
        r3=torch.cat([feat_under,feat_over],dim=1)
        channel_attention_03, filter_attention_03, spatial_attention_03, kernel_attention_03 = self.over_attention(r3)
        batch_size, in_planes, height, width = x_over.size()
        x_over = x_over * channel_attention_03
        x_over = x_over.reshape(1, -1, height, width)
        aggregate_weight_over = spatial_attention_03 * kernel_attention_03 * self.shared_weight.unsqueeze(dim=0)
        aggregate_weight_over = torch.sum(aggregate_weight_over, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output_over = F.conv2d(x_over, weight=aggregate_weight_over, bias=None, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups * batch_size)
        output_over = output_over.view(batch_size, self.out_planes, output_over.size(-2), output_over.size(-1))
        output_over = output_over * filter_attention_03
        # skip connection
        # output_over = short_cut_over + output_over
        
        return [output_under,output_over]

    def forward(self,x_over,x_under):
        return self._forward_impl_common(x_over=x_over,x_under=x_under)

class test_flops(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,reduction=0.0625, kernel_num=4,temperature = 34):
        super(test_flops, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        
    def forward(self,x_ref,x_over,x_under):
        feat1 = self.conv1(x_ref)
        feat2 = self.conv2(x_over)
        feat3 = self.conv3(x_under)
        return feat1,feat2,feat3



class Mutual_CrossAttention_ODEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,reduction=0.0625, kernel_num=4,temperature=34,mid_channel =16,inference=False):
        super(Mutual_CrossAttention_ODEncoder, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        # assert kernel_num == 4, f"Wrong kernel number: {kernel_num}, should be 4"

        if not inference:
            temperature = temperature
        else:
            temperature = 1
        # assert temperature == 1, f"Wrong temperature: {temperature}"
        self.over_attention = Mutual_CrossAttention(in_planes, out_planes, kernel_size, groups=groups,reduction=reduction, min_channel=mid_channel, kernel_num=kernel_num,temperature = temperature)
        self.ref_attention = Mutual_CrossAttention(in_planes, out_planes, kernel_size, groups=groups,reduction=reduction, min_channel=mid_channel, kernel_num=kernel_num,temperature = temperature)
        self.under_attention = Mutual_CrossAttention(in_planes, out_planes, kernel_size, groups=groups,reduction=reduction, min_channel=mid_channel, kernel_num=kernel_num,temperature = temperature)
        self.shared_weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes, kernel_size, kernel_size),requires_grad=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.shared_weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self):
        self.over_attention.update_temperature()
        self.ref_attention.update_temperature()
        self.under_attention.update_temperature()

    def _forward_impl_common(self,x_over,x_under):
        
        feat_under = x_under.clone()
        feat_over = x_over.clone()
        
        # under branch
        channel_attention_01, filter_attention_01, spatial_attention_01, kernel_attention_01= self.under_attention(x_ref=feat_over,x_nonref=feat_under)
        batch_size, in_planes, height, width = x_under.size()
        # print("channel_attention_01==>",channel_attention_01.shape)
        x_under = x_under * channel_attention_01
        x_under = x_under.reshape(1, -1, height, width)
        aggregate_weight_under = spatial_attention_01 * kernel_attention_01 * self.shared_weight.unsqueeze(dim=0)
        aggregate_weight_under = torch.sum(aggregate_weight_under, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output_under = F.conv2d(x_under, weight=aggregate_weight_under, bias=None, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups * batch_size)
        output_under = output_under.view(batch_size, self.out_planes, output_under.size(-2), output_under.size(-1))
        output_under = output_under * filter_attention_01
        # skip connection
        # output_under = short_cut_under + output_under
        
        
        # over branch
        channel_attention_03, filter_attention_03, spatial_attention_03, kernel_attention_03 = self.over_attention(x_ref=feat_under,x_nonref=feat_over)
        batch_size, in_planes, height, width = x_over.size()
        x_over = x_over * channel_attention_03
        x_over = x_over.reshape(1, -1, height, width)
        aggregate_weight_over = spatial_attention_03 * kernel_attention_03 * self.shared_weight.unsqueeze(dim=0)
        aggregate_weight_over = torch.sum(aggregate_weight_over, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output_over = F.conv2d(x_over, weight=aggregate_weight_over, bias=None, stride=self.stride, padding=self.padding,dilation=self.dilation, groups=self.groups * batch_size)
        output_over = output_over.view(batch_size, self.out_planes, output_over.size(-2), output_over.size(-1))
        output_over = output_over * filter_attention_03

        
        return [output_under,output_over]

    def forward(self,x_over,x_under):
        return self._forward_impl_common(x_over=x_over,x_under=x_under)

class test_flops(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,reduction=0.0625, kernel_num=4,temperature = 34):
        super(test_flops, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)
        
    def forward(self,x_ref,x_over,x_under):
        feat1 = self.conv1(x_ref)
        feat2 = self.conv2(x_over)
        feat3 = self.conv3(x_under)
        return feat1,feat2,feat3
