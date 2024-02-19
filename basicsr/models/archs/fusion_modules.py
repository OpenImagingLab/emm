import torch
import torch.nn as nn
import torch.nn.functional as f
# from basicsr.models.archs.recurrent_sub_modules import ConvLayer

class ConvLayer(nn.Module):
    """
    conv norm relu
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu_slope=0.2, norm=None):
        super(ConvLayer, self).__init__()
        self.relu_slope = relu_slope

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.relu_slope is not None:
            out = self.relu(out)

        return out




# def channel_attention(self, rgb_skip, depth_skip, attention):
#     assert rgb_skip.shape == depth_skip.shape, 'rgb skip shape:{} != depth skip shape:{}'.format(rgb_skip.shape, depth_skip.shape)
#     # single_attenton
#     rgb_attention = attention(rgb_skip)
#     depth_attention = attention(depth_skip)
#     rgb_after_attention = torch.mul(rgb_skip, rgb_attention)
#     depth_after_attention = torch.mul(depth_skip, depth_attention)
#     skip_after_attention = rgb_after_attention + depth_after_attention
#     return skip_after_attention

def se_layer(in_channels, out_channels):
    pool_attention = nn.AdaptiveAvgPool2d(1)
    conv_attention = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    activate = nn.Sigmoid()

    return nn.Sequential(pool_attention, conv_attention, activate)

class img_ev_fusion(nn.Module):
    """
    num_channels: the number of channels
    """
    def __init__(self, num_channels, fusion='add'):
        super(img_ev_fusion, self).__init__()
        self.fusion = fusion
        self.se_0 = se_layer(num_channels, num_channels)
        self.se_1 = se_layer(num_channels, num_channels)

    def forward(self, ev, feat_0, feat_1):
        weight_0 = self.se_0(ev)
        weight_1 = self.se_1(ev)
        
        x = feat_0*weight_0 + feat_1*weight_1
        
        return x

class AttenPred(nn.Module):
    """
    num_channels: the number of channels
    """
    def __init__(self, num_channels, mid_channels = 6, last_channels = 3):
        super(AttenPred, self).__init__()
        self.conv0 = ConvLayer(num_channels, mid_channels, kernel_size=3, stride=1, padding=1, relu_slope=0.2, norm=None)
        self.conv_last = nn.Conv2d(mid_channels, last_channels, kernel_size=3,stride=1,padding=1,bias=True)
        self.se = se_layer(num_channels, mid_channels)

    def forward(self, ev, imgs):
        x = self.conv0(ev) + imgs
        weight = self.se(ev)
        x = weight*x
        out = self.conv_last(x)
        
        return out


#############################


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CrossmodalAtten(nn.Module):
    def __init__(self, c, c_out, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv1_e = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2_e = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=2*dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # Channel Attention
        self.se_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_y_side = nn.Conv2d(in_channels=c, out_channels=c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm1_e = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c_out, 1, 1)), requires_grad=True) # !!!

    def forward(self, event_feat, image_feat):
        """
        event_feat: event branch feat  b, c, h, w
        image_feat: image branch feat
        """
        # x = event_feat
        # x = torch.cat((event_feat, image_feat), dim=0) # cat in b
        x = image_feat
        x_e = event_feat

        x = self.norm1(x) # both two modal
        x_e = self.norm1_e(x_e) # both two modal

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x_e = self.conv1_e(x_e)
        x_e = self.conv2_e(x_e)
        x_e = self.gelu(x_e)
        ## split to two modality
        # x_event, x_image = torch.chunk(x, chunks=2, dim=0) # split in b
        ## mutual atten
        x = x*self.se_1(x_e)
        x_e = x_e*self.se_1(x_e)

        x = torch.cat((x, x_e), dim=1) # cat in c
        # x = x * self.se(x)
        x = self.conv3(x) # fuse

        x = self.dropout1(x)

        y = image_feat + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = self.conv_y_side(y)
        # print('DEBUG: y.shape:{}'.format(y.shape))
        # print('DEBUG: x.shape:{}'.format(x.shape))
        # print('DEBUG: gamma.shape:{}'.format(self.gamma.shape))

        return y + x * self.gamma #



class CrossmodalAtten_imgeventalladd(nn.Module):
    def __init__(self, c, c_out, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv1_e = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2_e = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=2*dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # Channel Attention
        self.se_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_y_side = nn.Conv2d(in_channels=c, out_channels=c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm1_e = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c_out, 1, 1)), requires_grad=True) # !!!

    def forward(self, event_feat, image_feat):
        """
        event_feat: event branch feat  b, c, h, w
        image_feat: image branch feat
        """
        # x = event_feat
        # x = torch.cat((event_feat, image_feat), dim=0) # cat in b
        x = image_feat
        x_e = event_feat

        x = self.norm1(x) # both two modal
        x_e = self.norm1_e(x_e) # both two modal

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x_e = self.conv1_e(x_e)
        x_e = self.conv2_e(x_e)
        x_e = self.gelu(x_e)
        ## split to two modality
        # x_event, x_image = torch.chunk(x, chunks=2, dim=0) # split in b
        ## mutual atten
        x = x*self.se_1(x_e)
        x_e = x_e*self.se_1(x_e)

        x = torch.cat((x, x_e), dim=1) # cat in c
        # x = x * self.se(x)
        x = self.conv3(x) # fuse

        x = self.dropout1(x)

        y = event_feat + image_feat + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = self.conv_y_side(y)
        # print('DEBUG: y.shape:{}'.format(y.shape))
        # print('DEBUG: x.shape:{}'.format(x.shape))
        # print('DEBUG: gamma.shape:{}'.format(self.gamma.shape))

        return y + x * self.gamma #
