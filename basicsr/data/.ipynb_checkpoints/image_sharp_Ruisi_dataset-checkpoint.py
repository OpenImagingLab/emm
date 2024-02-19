from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch
import ipdb

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, events_to_voxel_grid_pytorch, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, voxel2voxeltensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate

import ipdb
### Sun Peng
class RuisiSharpEventRecurrentDataset(data.Dataset):
    """GoPro dataset for training recurrent networks for sharp image interpolation.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    Returns:
        two image (no paired voxels)
        corresponding voxels in the range
    """

    def __init__(self, opt):
        super(RuisiSharpEventRecurrentDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        assert self.m == 1, 'num of frames must be 1 for sharp image interpolation!'
        self.n = opt['num_inter_interpolation']
        self.num_input_blur = 2
        self.num_input_gt = 2*self.m + self.n
        self.num_bins = self.n + 1
        self.split = 'train' if opt['phase']=='train' else 'test' # train or test

        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False) # false for sharp image interpolation
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg

        train_video_list = os.listdir(os.path.join(self.dataroot, 'train'))
        test_video_list = os.listdir(os.path.join(self.dataroot, 'test'))

        video_list = train_video_list if self.split=='train' else test_video_list

        self.setLength = self.n + 2
        self.imageStepSeqsPath = []
        self.imageSeqsPath = [] 
        self.eventSeqsPath = [] # list of lists of event frames
        ### Formate file lists
        # ipdb.set_trace()
        for video in video_list:
            ## frames
            frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'gt'), suffix='.jpg'))  # all sharp frames in one video
            frames_step = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'step'), suffix='.jpg'))
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'event'), suffix='.npz'))  # all sharp frames in one video

            # print('DEBUG: frames:{}'.format(frames))
            # del(frames[0]) # del the first image of the seq, because we want to use the event before first image
            n_sets = (len(frames) - self.setLength)//(self.n + 1)  + 1

            videoInputs = [frames[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(self.dataroot, self.split, video, 'gt', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
            self.imageSeqsPath.extend(videoInputs)# list of lists of paired blur input, e.g.:
            # [['GOPR0372_07_00/blur/000328.png', 'GOPR0372_07_00/blur/000342.png'],
            #  ['GOPR0372_07_00/blur/000342.png', 'GOPR0372_07_00/blur/000356.png']]
            videoStepInputs = [frames_step[(self.n +1)*i:(self.n +1)*i+self.setLength] for i in range(n_sets)]
            videoStepInputs = [[os.path.join(self.dataroot, self.split, video, 'step', f) for f in group] for group in videoStepInputs]
            self.imageStepSeqsPath.extend(videoStepInputs)
            # events
            eventInputs = [event_frames[(self.n +1)*i :(self.n +1)*i+self.setLength -1] for i in range(n_sets)]
            eventInputs = [[os.path.join(self.dataroot, self.split, video, 'event', f) for f in group] for group in eventInputs] # GOPR0372_07_00/xxx.png ...
            self.eventSeqsPath.extend(eventInputs)

        assert len(self.imageSeqsPath)==len(self.eventSeqsPath), 'The number of sharp/interpo: {}/{} does not match.'
        # ipdb.set_trace()
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        all_image_paths = self.imageSeqsPath[index]
        step_image_paths = self.imageStepSeqsPath[index]
        event_paths = self.eventSeqsPath[index]
        input_idx = [0,-1]
        gt_idx = list(range(1, self.setLength-1))
        image_paths = [step_image_paths[idx] for idx in input_idx]
        gt_paths = [all_image_paths[idx] for idx in gt_idx]
        # print("event",event_paths)
        # print("gt",gt_paths)
        # print("input",image_paths)
        
        assert len(event_paths) == len(gt_paths)+1, 'The length error' # cyt
        
        # print("[DEBUG]: len of event_paths:{}".format(len(event_paths))) # 8
        # print("[DEBUG]: len of gt_paths:{}".format(len(gt_paths))) # 7
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            image_paths.reverse()
            gt_paths.reverse()
            # TODO: reverse event

        ## Read blur and gt sharps
        img_lqs = []
        img_gts = []
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        for gt_path in gt_paths:
            # get GT
            img_bytes = self.file_client.get(gt_path)    # 'gt'
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
        
        h_lq, w_lq, _ = img_lqs[0].shape
        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]
        # npz -> ndarray
        voxels = []
        if self.one_voxel_flg:
            # all_quad_event_array = np.zeros((0,4)).astype(np.float32)
            all_quad_event_array = []
            for event in events:
                ### IMPORTANT: dataset mistake x and y !!!!!!!!
                ###            Switch x and y here !!!!
                # y = event['x'].astype(np.float32)  
                # # print('x.max:{}'.format(x.max()))      
                # # print('x.min:{}'.format(x.min()))             
                # x = event['y'].astype(np.float32)
                # # print('y.max:{}'.format(y.max()))            
                # # print('y.min:{}'.format(y.min()))             
                # t = event['timestamp'].astype(np.float32)
                # p = event['polarity'].astype(np.float32)

                # print(p)
                # # t[-1] += 10 ## Avoid the error of event to voxel function
                # # print('DEBUG: t:{}'.format(t))

                # this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4

                # cyt: read motion mag data
                this_quad_event_array = event['event_array']

                all_quad_event_array.append(this_quad_event_array)
            voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            # Voxel Norm
            # if self.norm_voxel:
            #     voxel = voxel_norm(voxel)
            # print("voxel",voxel.shape)
            voxels.append(voxel) # len=1, shape:h,w,num_bins

            # num_bins,h,w
        else:
            for i in range(len(events)):
                ### IMPORTANT: dataset mistake x and y !!!!!!!!
                ###            Switch x and y here !!!!
                # y = event['x'].astype(np.float32)  
                # # print('x.max:{}'.format(x.max()))      
                # # print('x.min:{}'.format(x.min()))             
                # x = event['y'].astype(np.float32)
                # # print('y.max:{}'.format(y.max()))            
                # # print('y.min:{}'.format(y.min()))             
                # t = event['timestamp'].astype(np.float32)
                # p = event['polarity'].astype(np.float32)
                # # t[-1] += 10 ## Avoid the error of event to voxel function
                # # print('DEBUG: t:{}'.format(t))
                
                this_quad_event_array = event['event_array']
                if i == 0:
                    last_quad_event_array = this_quad_event_array
                elif i >=1:
                    two_quad_event_array = np.concatenate((last_quad_event_array, this_quad_event_array), axis=0)
                    sub_voxel = events_to_voxel_grid(two_quad_event_array, num_bins=2,width=w_lq, height=h_lq, return_format='HWC')
                    voxels.append(sub_voxel)
                    last_quad_event_array = this_quad_event_array
                # len=2m+n+1, each with shape: h,w,2
        # Voxel: list of chw or hwc
        # randomly crop
        # voxel shape: h,w,c
        if gt_size is not None:
            img_gts, img_lqs, voxels = triple_random_crop(img_gts, img_lqs, voxels, gt_size, scale, gt_paths[0])

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_gt = len(img_gts) if isinstance(img_gts, list) else 1
        num_voxel = len(voxels) if isinstance(voxels, list) else 1

        img_lqs.extend(img_gts)
        img_lqs.extend(voxels) if isinstance(voxels,list) else img_lqs.append(voxels) # [img_lqs, img_gts, voxels]

        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        img_gts = torch.stack(img_results[num_lq:num_lq+num_gt], dim=0)
        voxels_list = img_results[num_lq+num_gt:]
        ## Norm voxel
        if self.norm_voxel:
            for voxel in voxels_list:
                voxel = voxel_norm(voxel)

        voxels = torch.stack(voxels_list, dim=0) # t,c,h,w

        # reshape of the voxel tensor   1, num_bins, h, w -> t, 2, h, w
        if self.one_voxel_flg:
            voxels = voxels.squeeze(0)
            all_voxel = []
            for i in range(voxels.shape[0]-1):
                sub_voxel = voxels[i:i+2, :, :]
                all_voxel.append(sub_voxel)
            voxels = torch.stack(all_voxel, dim=0)

        # print('DEBUG: lq.shape:{}'.format(img_lqs.shape))
        # print('DEBUG: gt.shape:{}'.format(img_gts.shape))
        # print('DEBUG: voxel.shape:{}'.format(voxels.shape))

        # img_lqs: (t, c, h, w)
        # img_lqs: (3*2+(m-1)*2, h, w) if return_deblur_voxel

        # img_gts: (t, c, h, w)
        # voxels: (t, num_bins (2), h, w)
        blur0_path = image_paths[0]
        # print('blur0_path:{}'.format(blur0_path))
        seq = blur0_path.split(f'{self.split}/')[1].split('/')[0]
        origin_index = os.path.basename(blur0_path).split('.')[0]

        if self.split == 'train':
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}
        else:
            return {'lq': img_lqs, 'gt': img_gts, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}

    def __len__(self):
        return len(self.imageSeqsPath)


