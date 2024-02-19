import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from basicsr.models.archs.dcn_util import ModulatedDeformConvPack
from basicsr.models.archs.fusion_modules import CrossmodalAtten, CrossmodalAtten_imgeventalladd
import ipdb

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

############## image encoder #####################
class ImageEncoderConvBlock(nn.Module):
    """
    x conv relu conv relu +  conv_down(k=4 s=2 nobias)
    |------conv-----------|
    """
    def __init__(self, in_size, out_size, downsample, relu_slope): # cat
        super(ImageEncoderConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_3 = nn.LeakyReLU(relu_slope, inplace=False)        

        if downsample:
            self.down = conv_down(out_size, out_size, bias=False)


    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out_conv3 = self.relu_3(self.conv_3(out_conv2))
        out = out_conv3 + self.identity(x)
        if self.downsample:
            out = self.down(out)

        return out
            

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
            if type(relu_slope) is str:
                self.relu = nn.ReLU()
            else:
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



############### new event encoder with atten fusion and img+event #########################
# cyt: main encoder
class SimpleRecurrentThenDownAttenfusionmodifiedConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, downsample=True,use_first_dcn=False, use_atten_fuse=False):
        super(SimpleRecurrentThenDownAttenfusionmodifiedConvLayer, self).__init__()
        self.relu_slope = relu_slope
        self.use_atten_fuse = use_atten_fuse
        self.downsample = downsample
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_atten_fuse:
            self.atten_fuse = CrossmodalAtten_imgeventalladd(c=in_channels, c_out = out_channels, DW_Expand=1, FFN_Expand=2)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x, y=None, prev_state=None, bi_direction_state = None):
        # x = self.conv(x)
        # if self.relu_slope is not None:
            # x = self.relu(x)
        if y is not None:
            if self.use_atten_fuse:
                x = self.atten_fuse(x,y)
            else:
                x = x+y
                x = self.conv(x) # increase the c dimension
                if self.relu_slope is not None:
                    x = self.relu(x)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)
        # print('DEBUG: x.shape:{}'.format(x.shape))
        # if prev_state is not None:
            # print('DEBUG: prev_state.shape:{}'.format(prev_state.shape))

        x, state = self.recurrent_block(x, prev_state) # tensor
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)

        if self.downsample:
            x = self.down(x)
        
        return x, state

############### new event encoder with atten fusion and img+event and lstm #########################
class BiLinkRecurrentThenDownAttenfusionmodifiedConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, downsample=True,use_first_dcn=False, use_atten_fuse=False):
        super(BiLinkRecurrentThenDownAttenfusionmodifiedConvLayer, self).__init__()
        self.relu_slope = relu_slope
        self.use_atten_fuse = use_atten_fuse
        self.downsample = downsample
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_atten_fuse:
            self.atten_fuse = CrossmodalAtten_imgeventalladd(c=in_channels, c_out = out_channels, DW_Expand=1, FFN_Expand=2)

        self.recurrent_block = BiLinkRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x, y=None, prev_state=None, prev_state2=None, bi_direction_state = None):
        # x = self.conv(x)
        # if self.relu_slope is not None:
            # x = self.relu(x)
        if y is not None:
            if self.use_atten_fuse:
                x = self.atten_fuse(x,y)
            else:
                x = x+y
                x = self.conv(x) # increase the c dimension
                if self.relu_slope is not None:
                    x = self.relu(x)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)
        # print('DEBUG: x.shape:{}'.format(x.shape))
        # if prev_state is not None:
            # print('DEBUG: prev_state.shape:{}'.format(prev_state.shape))

        x, state = self.recurrent_block(x, prev_state, prev_state2) # tensor
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)

        if self.downsample:
            x = self.down(x)
        
        return x, state

class TransposedConvLayer(nn.Module):
    """
    TransConv norm relu
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None, upsample=True):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True

        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out



class UpsampleConvLayer(nn.Module):
    """
    bilinear conv norm relu
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposeRecurrentConvLayer(nn.Module):
    """
    for decoder
    transposeconv, recurrent conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, norm=None, fuse_two_direction=False, upsample=True):
        super(TransposeRecurrentConvLayer, self).__init__()
        self.hidden_channel = out_channels
        self.fuse_two_direction = fuse_two_direction
        self.upsample = upsample
        if upsample:
            self.transposed_conv2d = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=True)
        else:
            self.transposed_conv2d = nn.Conv2d(
                in_channels, out_channels, 3, stride=1, padding=1, bias=True)
        
        self.forward_trunk = ConvResidualBlocks(out_channels+self.hidden_channel, out_channels, num_block=1)
        if self.fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope=0.2, norm=norm)

    def forward(self, x, prev_state, bi_direction_state=None):
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]
        spatial_size = list(spatial_size)
        for i in range(len(spatial_size)):
            if self.upsample:
                spatial_size[i] *= 2

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_channel] + spatial_size
            prev_state = torch.zeros(state_size).to(x.device)
        out = self.transposed_conv2d(x)

        if self.fuse_two_direction and bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)
        # ipdb.set_trace()
        out = torch.cat([out, prev_state], dim=1)
        out = self.forward_trunk(out)
        state = out

        return out, state




# Residual block
class ResidualBlock(nn.Module):
    """
    x conv (norm) relu conv (norm) +x relu
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state



############### new event encoder #########################
class SimpleNoRecurrentThenDownConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_first_dcn=False):
        super(SimpleNoRecurrentThenDownConvLayer, self).__init__()
        self.relu_slope = relu_slope
        # assert(recurrent_block_type in ['convlstm', 'convgru', 'simpleconv'])
        # self.recurrent_block_type = recurrent_block_type
        # print("DEBUG: in recurrent conv layer:{}".format(recurrent_block_type))
        if use_first_dcn:
            self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.recurrent_block = SimpleNoRecurrentConv(out_channels, 0, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x):
        x = self.conv(x)
        if self.relu_slope is not None:
            x = self.relu(x)
        x = self.recurrent_block(x) # tensor
        x = self.down(x)
        
        return x


class SimpleRecurrentConv(nn.Module):
    """
    SimpleRecurrentConv, borrowed from BasicVSR
    """

    def __init__(self, input_size, hidden_size, num_block=4):
        super().__init__()
        # propagation
        self.hidden_size = hidden_size
        self.forward_trunk = ConvResidualBlocks(input_size + hidden_size, input_size, num_block)
        # self.fusion = nn.Conv2d(input_size * 2, input_size, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, prev_state):

        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(x.device)

        # backward branch
        feat_prop = torch.cat([x, prev_state], dim=1)
        feat_prop = self.forward_trunk(feat_prop)
        state = feat_prop 

        # print("feat_prop type: {}".format(type(feat_prop)))
        # print("state type: {}".format(type(state)))

        return feat_prop, state

class BiLinkRecurrentConv(nn.Module):
    """
    BiLinkRecurrentConv, borrowed from BasicVSR
    """

    def __init__(self, input_size, hidden_size, num_block=4):
        super().__init__()
        # propagation
        self.hidden_size = hidden_size
        self.forward_trunk = ConvResidualBlocks(input_size + 2*hidden_size, input_size, num_block)
        # self.fusion = nn.Conv2d(input_size * 2, input_size, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, prev_state, prev_state2):

        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(x.device)
        
        if prev_state2 is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state2 = torch.zeros(state_size).to(x.device)

        # backward branch
        feat_prop = torch.cat([x, prev_state, prev_state2], dim=1)
        feat_prop = self.forward_trunk(feat_prop)
        state = feat_prop 

        # print("feat_prop type: {}".format(type(feat_prop)))
        # print("state type: {}".format(type(state)))

        return feat_prop, state

class SimpleNoRecurrentConv(nn.Module):
    """
    SimpleNoRecurrentConv
    """

    def __init__(self, input_size, hidden_size=0, num_block=4):
        super().__init__()
        # propagation
        self.hidden_size = hidden_size
        self.forward_trunk = ConvResidualBlocks(input_size + hidden_size, input_size, num_block)
        # self.fusion = nn.Conv2d(input_size * 2, input_size, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        # backward branch
        feat_prop = x
        feat_prop = self.forward_trunk(feat_prop)

        # print("feat_prop type: {}".format(type(feat_prop)))
        # print("state type: {}".format(type(state)))

        return feat_prop





# sub modules
class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)



class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
