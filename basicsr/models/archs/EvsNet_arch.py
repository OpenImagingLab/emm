import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    ResidualBlock, ConvLSTM, ConvGRU, ImageEncoderConvBlock,  BiLinkRecurrentThenDownAttenfusionmodifiedConvLayer,\
        TransposeRecurrentConvLayer, SimpleRecurrentThenDownAttenfusionmodifiedConvLayer
from basicsr.models.archs.dcn_util import ModulatedDeformConvPack
from einops import rearrange
import ipdb

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


############## image encoder #####################
class ImageDecoderConvBlock(nn.Module):
    """
    x conv relu conv relu +  conv_up(transpose conv)
    |------conv-----------|
    """
    def __init__(self, in_size, out_size, upsample, relu_slope): # cat
        super(ImageDecoderConvBlock, self).__init__()
        self.upsample = upsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_3 = nn.LeakyReLU(relu_slope, inplace=False)          

        if upsample:
            self.up = nn.ConvTranspose2d(in_channels=out_size, out_channels=out_size,\
            kernel_size=2, stride=2, padding=0, bias=False) # 64, H*2, W*2


    def forward(self, x):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)
        if self.upsample:
            out = self.up(out)
        return out

class FinalDecoderRecurrentUNet(nn.Module):
    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum', activation='sigmoid',
                 num_encoders=3, num_decoder_MN =2, base_num_channels=32, num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=True):
        super(FinalDecoderRecurrentUNet, self).__init__()

        self.ev_chn = ev_chn
        self.img_chn = img_chn
        self.out_chn = out_chn
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_recurrent_upsample_conv:
            print('Using Recurrent UpsampleConvLayer (slow, but recurrent in decoder)')
            self.UpsampleLayer = TransposeRecurrentConvLayer
        else:
            print('Using No recurrent UpsampleConvLayer (fast, but no recurrent in decoder)')
            self.UpsampleLayer = UpsampleConvLayer

        self.num_encoders = num_encoders
        self.num_decoder_MN = num_decoder_MN
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.ev_chn > 0)
        assert(self.img_chn > 0)
        assert(self.out_chn > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_indexs = []
        for i in range(self.num_encoders):
            self.encoder_indexs.append(i)

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders) ]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for index, input_size in enumerate(decoder_input_sizes):
            upsample = False if index == 0 and len(decoder_input_sizes)>2 else True
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=2, padding=0, norm=self.norm, upsample=upsample)) # kernei_size= 5, padidng =2 before

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.out_chn, kernel_size=3, stride=1, padding=1, relu_slope=None, norm=self.norm)



class EvsMotionMagNet(FinalDecoderRecurrentUNet):

    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=3, num_decoder_MN = 2, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=True, num_block=3, use_first_dcn=False, use_reversed_voxel=False):
        super(EvsMotionMagNet, self).__init__(img_chn, ev_chn, out_chn, skip_type, activation,
                                            num_encoders, num_decoder_MN,base_num_channels, num_residual_blocks, norm,
                                            use_recurrent_upsample_conv)
        self.use_reversed_voxel = use_reversed_voxel

        # self.alpha = 40

        ## event
        self.head = ConvLayer(self.ev_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.encoders_forward = nn.ModuleList()
        # self.encoders_backward = nn.ModuleList()

        for input_size, output_size, encoder_index in zip(self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs):
            # print('DEBUG: input size:{}'.format(input_size))
            # print('DEBUG: output size:{}'.format(output_size))
            print('Using enhanced attention!')
            use_atten_fuse = True if encoder_index == 1 else False
            downsample = True if encoder_index < 2 else False

            self.encoders_forward.append(BiLinkRecurrentThenDownAttenfusionmodifiedConvLayer(input_size, output_size,
                                                    kernel_size=3, stride=1, padding=1, fuse_two_direction=True,
                                                    norm=self.norm, num_block=num_block, use_first_dcn=use_first_dcn,
                                                    use_atten_fuse=use_atten_fuse,downsample=downsample))

        ## img
        self.head_img = ConvLayer(self.img_chn, self.base_num_channels // 2,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.img_encoders = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs):
            downsample = True if encoder_index < 2 else False
            self.img_encoders.append(ImageEncoderConvBlock(in_size=input_size // 2, out_size=output_size // 2,
                                                            downsample=downsample, relu_slope=0.2))
        
        self.img_decoders_M = nn.ModuleList()
        self.img_decoders_V = nn.ModuleList()
        self.evs_decoders_dM = nn.ModuleList()

        channel_current = self.encoder_output_sizes[-1]

        for index in range(self.num_decoder_MN):
            
            if index == 0: 
                self.img_decoders_M.append(ImageDecoderConvBlock(in_size=channel_current // 2, out_size=channel_current,
                                                                upsample=False, relu_slope=0.2))
                self.img_decoders_V.append(ImageDecoderConvBlock(in_size=channel_current // 2, out_size=channel_current,
                                                                upsample=False, relu_slope=0.2))

            else:
                self.img_decoders_M.append(ImageDecoderConvBlock(in_size=channel_current, out_size=channel_current,
                                                                upsample=False, relu_slope=0.2))
                self.img_decoders_V.append(ImageDecoderConvBlock(in_size=channel_current, out_size=channel_current,
                                                                upsample=False, relu_slope=0.2))
            self.evs_decoders_dM.append(ImageDecoderConvBlock(in_size=channel_current, out_size=channel_current,
                                                            upsample=False, relu_slope=0.2))
                                
        self.merge_imgMV = ConvLayer(channel_current*2, channel_current, kernel_size=3, stride=1, padding=1, relu_slope=0.2)
        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, event ,alpha = 40):


        if x.dim()==5:
            x = rearrange(x, 'b t c h w -> (b t) c h w') 
        b, t, num_bins, h, w = event.size()
        event = rearrange(event, 'b t c h w -> (b t) c h w')
        
        # -----------head-----------
        x = self.head_img(x) # image feat
        head = x
        e = self.head(event)   # event feat

        # --------image encoder-----------
        x_blocks = []
        
        for i, img_encoder in enumerate(self.img_encoders):
            x = img_encoder(x)
            x_blocks.append(rearrange(x, '(b t) c h w -> b (t c) h w', b=b,t=2))

        M = x
        V = x
  
        for i in range(len(self.img_decoders_M)):
            M = self.img_decoders_M[i](M)  #(b t), 128, 64,64
            V = self.img_decoders_V[i](V)

        ## prepare for propt 
        M = rearrange(M, '(b t) c h w -> b t c h w', b=b, t=2)
        V = rearrange(V, '(b t) c h w -> b t c h w', b=b, t=2)
        e = rearrange(e, '(b t) c h w -> b t c h w', b=b, t=t)


        out_l = []
        backward_all_states = [] # list of list
        backward_prev_states = [None] * self.num_encoders # prev states for each scale
        forward_prev_states = [None] * self.num_encoders # prev states for each scale
        forward_prev_states2 = [None] * self.num_encoders # prev states for each scale
        prev_states_decoder = [None] * self.num_encoders



        ## forward propt 
        for frame_idx in range(0,t):
            e_blocks = [] # skip feats for each frame
            e_cur = e[:, frame_idx,:,:,:] # b,c,h,w

            # --------event encoder--------
            for i, encoder in enumerate(self.encoders_forward):
                if i==0: # no img feat in first block
                    e_cur, state = encoder(x=e_cur, y=None, prev_state=forward_prev_states[i], prev_state2=forward_prev_states2[i])
                else:
                    e_cur, state = encoder(x=e_cur, y=x_blocks[i-1], prev_state=forward_prev_states[i], prev_state2=forward_prev_states2[i])

                e_blocks.append(e_cur)
                forward_prev_states2[i] = forward_prev_states[i]
                forward_prev_states[i] = state # update state for next frame

            ## calculate dM
            dM = e_cur # b, 128, 64, 64
            for i in range(len(self.evs_decoders_dM)):
                dM = self.evs_decoders_dM[i](dM)

            # --------Manupulator--------
            # ipdb.set_trace()
            M_mag = M[:,0,...] + alpha*dM

            e_cur = torch.cat((M_mag, 1/2*(V[:,0,...]+V[:,1,...])), dim=1)
            e_cur = self.merge_imgMV(e_cur)

            # --------Reconstruction--------
            # residual blocks
            for i in range(len(self.resblocks)):
                if i == 0:
                    e_cur = self.resblocks[i](e_cur+x_blocks[-1])
                else:
                    e_cur = self.resblocks[i](e_cur)

            ## Decoder
            for i, decoder in enumerate(self.decoders):
                
                e_cur, state = decoder(e_cur, prev_states_decoder[i])
                prev_states_decoder[i] = state
        
            # tail
            out = self.pred(e_cur)
            out_l.append(out)  
        ret = {
            'pred':  torch.stack(out_l, dim=1), # b,t,c,h,w [1, 23, 3, 256, 256]
            'M': M, # [1, 2, 128, 64, 64]
            'dM': dM, # last dM [1, 128, 64, 64]
            'V': V,  # [1, 2, 128, 64, 64]
        }
        return ret 

if __name__ == '__main__':
    import time

    model = TuneBidirectionAttenfusion_v4(img_chn=3, ev_chn=2)
    device = 'cuda'
    x = torch.rand(1, 2, 3, 256, 256).to(device)
    event = torch.rand(1, 23, 2, 256, 256).to(device)
    model = model.to(device)

    start_time = time.time()
    result = model(x, event)
    end_time = time.time()

    inference_time = end_time-start_time
    print('Inference time:{}'.format(inference_time))

