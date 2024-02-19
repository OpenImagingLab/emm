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

# import ipdb


class TestEventMagRecurrentDataset(data.Dataset):
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
        super(TestEventMagRecurrentDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        assert self.m == 1, 'num of frames must be 1 for sharp image interpolation!'
        self.n = opt['num_inter_interpolation']
        self.num_input_gt = 2*self.m + self.n
        self.num_bins = self.n + 1

        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False) # false for sharp image interpolation
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg
        
        # ipdb.set_trace()
        test_video_list = os.listdir(self.dataroot)

        self.setLength = self.n + 2
        self.imageStepSeqsPath = []
        self.imageSeqsPath = [] 
        self.eventSeqsPath = [] # list of lists of event frames
        ### Formate file lists
        # ipdb.set_trace()
        for video in test_video_list:
            ## frames
            frames  = sorted(recursive_glob(rootdir=os.path.join(self.dataroot,  video, 'rgb'), suffix='.png')\
                + recursive_glob(rootdir=os.path.join(self.dataroot,video, 'rgb'), suffix='.jpg'))  # all sharp frames in one video
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, video, 'evs'), suffix='.npz'))  # all sharp frames in one video
            
            # print('DEBUG: frames:{}'.format(frames))
            # del(frames[0]) # del the first image of the seq, because we want to use the event before first image
            n_sets = (len(frames) - self.setLength)//(self.n + 1)  + 1
            
            videoInputs = [frames[i:i+2] for i in range(len(frames)-1)]
            videoInputs = [[os.path.join(self.dataroot, video, 'rgb', f) for f in group] for group in videoInputs] # GOPR0372_07_00/xxx.png ...
            self.imageSeqsPath.extend(videoInputs)# list of lists of paired blur input, e.g.:
            # [['GOPR0372_07_00/blur/000328.png', 'GOPR0372_07_00/blur/000342.png'],
            #  ['GOPR0372_07_00/blur/000342.png', 'GOPR0372_07_00/blur/000356.png']]
            # events
            # eventInputs = [event_frames[i:i+1] for i in range(len(event_frames))]
            # eventInputs = [event_frames[i:i+1] for i in range(1)]
            eventInputs = [[os.path.join(self.dataroot, video, 'evs', f) for f in event_frames]] # GOPR0372_07_00/xxx.png ...
            self.eventSeqsPath.extend(eventInputs)
            # ipdb.set_trace()
        assert len(self.imageSeqsPath)==len(self.eventSeqsPath), f'The number of sharp/interpo: {self.imageSeqsPath}/{self.eventSeqsPath} does not match.'
        # ipdb.set_trace()
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')

    def shift_event(self,event):
        """
            modify our real testset to 't x y p'
        """
        # event = [list(element) for element in event ]
        event = np.array(event)
        x = event[:,0]
        mask = x>580
        x = x[mask]
        y = event[:,1]
        y = y[mask]
        pols = event[:,2]
        pols = pols[mask]
        t = event[:,3]
        t = t[mask]
        pols[pols==0] = -1

        tmax = t.max()
        tmin = t.min()
        
        event = np.stack((t,x,y,pols),-1)

        # evs_len = event.shape[0]
        # event_list = [event[int(evs_len/self.num_bins*i):int(evs_len/self.num_bins*(i+1)),:] for i in range(self.num_bins)]
        event_list = [event[np.logical_and( event[:,0]>=(tmax-tmin)/self.num_bins*i + tmin, event[:,0]<(tmax-tmin)/self.num_bins*(i+1)+tmin)] for i in range(self.num_bins)]
        return event_list

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        all_image_paths = self.imageSeqsPath[index]
        event_paths = self.eventSeqsPath[index]
        input_idx = [0,1]
        image_paths = [all_image_paths[idx] for idx in input_idx]

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            image_paths.reverse()
            # TODO: reverse event

        ## Read two end images
        img_lqs = []
        for image_path in image_paths:
            # get LQ
            img_bytes = self.file_client.get(image_path)  # 'lq'
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)
        
        h_lq, w_lq, _ = img_lqs[0].shape
        ## Read event and convert to voxel grid:
        events = [np.load(event_path,allow_pickle=True) for event_path in event_paths]
        # npz -> ndarray
        voxels = []
        # ipdb.set_trace()
        if self.one_voxel_flg:
            # all_quad_event_array = np.zeros((0,4)).astype(np.float32)
            all_quad_event_array = []
            # events = [events[0],events[0]]
            for event in events:
                # cyt: read motion mag data
                this_quad_event_array = event['event_array']
                this_quad_event_array = [list(element) for element in this_quad_event_array ]
                
                all_quad_event_array += this_quad_event_array
            # this_quad_event_array = event['event_array']
            all_quad_event_array = self.shift_event(all_quad_event_array)
            
            voxel = events_to_voxel_grid(all_quad_event_array, num_bins=self.num_bins, width=w_lq, height=h_lq, return_format='HWC')
            # ipdb.set_trace()
            # Voxel Norm
            # if self.norm_voxel:
            #     voxel = voxel_norm(voxel)
            # print("voxel",voxel.shape)
            voxels.append(voxel) # len=1, shape:h,w,num_bins

            # num_bins,h,w
        else:
            for i in range(len(events)):
                
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
            img_lqs, voxels = triple_random_crop(img_lqs, voxels, gt_size, scale, gt_paths[0])

        # augmentation - flip, rotate
        num_lq = len(img_lqs) if isinstance(img_lqs, list) else 1
        num_voxel = len(voxels) if isinstance(voxels, list) else 1

        img_lqs.extend(voxels) if isinstance(voxels,list) else img_lqs.append(voxels) # [img_lqs, img_gts, voxels]

        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results) # hwc -> chw
        img_lqs = torch.stack(img_results[:num_lq], dim=0) # t,c,h,w
        voxels_list = img_results[num_lq:] # 1,60,256,256
        ## Norm voxel
        if self.norm_voxel:
            for voxel in voxels_list:
                voxel = voxel_norm(voxel)

        voxels = torch.stack(voxels_list, dim=0) # t,c,h,w
        voxels = voxels.squeeze(0)
        voxels = voxels.reshape((self.num_bins,2,voxels.shape[1],voxels.shape[2]))
        # voxels = voxels[:self.num_bins-1,...]

        blur0_path = image_paths[0]
        # print('blur0_path:{}'.format(blur0_path))
        seq = blur0_path.split('/')[-4] +  '__' + blur0_path.split('/')[-3]
        origin_index = os.path.basename(blur0_path).split('.')[0]

        return {'lq': img_lqs, 'voxel': voxels, 'seq': seq, 'origin_index': origin_index}

    def __len__(self):
        return len(self.imageSeqsPath)


