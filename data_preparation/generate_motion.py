import os
from PIL import Image, ImageEnhance
import numpy as np
import torch
import xml.dom.minidom as xmldom
import glob
import random
import yaml
import argparse
from tqdm import tqdm
# import ipdb
import cv2
import threading
import ipdb

def paste_foreground(background,foreground, mask, corner_x, corner_y,scale):
    """
    Paste foreground to background

    :param background: Image of background
    :param foreground: A list [N, 1] of foreground
    :param mask: A list [N, 1] of mask
    :param corner_x, corner_y: An array [N, 1] of corner of the paste area
    """
    w_b, h_b = background.size
    for i in range(len(foreground)):
        w,h = foreground[i].size
        background.paste(foreground[i],(corner_x[i], corner_y[i], corner_x[i]+w, corner_y[i]+h),mask = mask[i])
    # background = background.resize((int(w_b*downsampling_factor),int(h_b*downsampling_factor)),Image.Resampling.BICUBIC)
    # w = int(256*scale); h = int(256*scale)
    w = 256; h = 256
    background = background.resize((w,h),Image.Resampling.LANCZOS)
    # background = background.crop((0,0,256,256))
    return background




def main(args,rank,num_thread):

    f = open(args.config)
    config = yaml.load(f,Loader=yaml.FullLoader)

    scene_num = 10000
    root = os.getcwd()

    random.seed(0)
    np.random.seed(0)
    

    mask_list = glob.glob(os.path.join(root,"foreground", "mask","*.jpg")) + glob.glob(os.path.join(root,"foreground","mask","*.png"))
    mask_list.sort()
    foreground_list = glob.glob(os.path.join(root,"foreground", "object","*.jpg")) + glob.glob(os.path.join(root,"foreground","object","*.png"))
    foreground_list.sort()
    background_list = glob.glob(os.path.join(root, "background","*.jpg")) + glob.glob(os.path.join(root,"background","*.png"))
    background_list.sort()

    train_dir = os.path.join(root,"trainset0109",'train')
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(root,"trainset0109",'test')
    os.makedirs(test_dir, exist_ok=True)

    
    foregrounds = []
    masks = []
    foreground_len = len(foreground_list)
    for i in range(foreground_len):
        # enhancer = ImageEnhance.Brightness(Image.open(foreground_list[i]).convert('RGB'))
        foregrounds.append(Image.open(foreground_list[i]).convert('RGB'))
        masks.append(Image.open(mask_list[i]).convert('1'))
    
    foreground_num = np.random.randint(config["foreground_min"],config["foreground_max"],size=(scene_num),dtype='int')
    train_sample_range = range(foreground_len-100)  # last 100 for test
    test_sample_range = range(foreground_len-100,foreground_len)
    
    # for i in tqdm(range(5000,scene_num)):
    for i in tqdm(range(int(scene_num/num_thread*rank),int(scene_num/num_thread*(rank+1)))):
        

        # enhancer = ImageEnhance.Brightness(Image.open(background_list[i]).convert('RGB'))
        background = Image.open(background_list[i]).convert('RGB')
        background = background.resize((4096,4096),Image.Resampling.BICUBIC)

        w_bg, h_bg = background.size
        bg_scale = 1
        # if w_bg<1024 or h_bg<1024:
        #     continue

        #motion and manification
        motion_num = config["motion_num"]
        # mag_factor = random.randint(1,config["mag_max"]) # for random magnification factor
        mag_factor = np.random.randint(config["mag_min"],config["mag_max"])
        
        if i < scene_num-50: #training
            foreground_index = random.sample(train_sample_range,foreground_num[i])
            save_dir = os.path.join(train_dir,"train"+str(i).zfill(5)+"_{}".format(mag_factor))

        else:#testing
            foreground_index = random.sample(test_sample_range,foreground_num[i])
            save_dir = os.path.join(test_dir,"test"+str(i).zfill(5)+"_{}".format(mag_factor))
        image_a = background.copy()
        image_b = background.copy()

        #origin xy for foreground
        corner_x_ori = np.random.randint(100,w_bg-100,len(foreground_index))
        corner_y_ori = np.random.randint(100,h_bg-100,len(foreground_index))

        # motion_x = np.random.randint(-config["motion_max"],config["motion_max"],(motion_num,len(foreground_index)))
        # motion_y = np.random.randint(-config["motion_max"],config["motion_max"],(motion_num,len(foreground_index)))
        motion_x = np.random.choice([-2,-1,1,2],(motion_num,len(foreground_index)))
        motion_y = np.random.choice([-2,-1,1,2],(motion_num,len(foreground_index)))

        motion_x_final = np.sum(motion_x,axis=0)
        motion_y_final = np.sum(motion_y,axis=0)

        corner_x_motion = motion_x_final + corner_x_ori
        corner_y_motion = motion_y_final + corner_y_ori

        corner_x_step = np.nancumsum(motion_x,axis=0) + corner_x_ori
        corner_y_step = np.nancumsum(motion_y,axis=0) + corner_y_ori

        corner_x_step_mag = np.nancumsum(motion_x,axis=0) * mag_factor + corner_x_ori
        corner_y_step_mag = np.nancumsum(motion_y,axis=0) * mag_factor + corner_y_ori
        
        corner_x_mag = motion_x_final * mag_factor + corner_x_ori
        corner_y_mag = motion_y_final * mag_factor + corner_y_ori
        # randomly resize
        mask_resized = []
        foreground_resized = []

        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir,"step"), exist_ok=True)
        # os.makedirs(os.path.join(save_dir,"flow"), exist_ok=True)
        os.makedirs(os.path.join(save_dir,"gt"), exist_ok=True)
        for j in foreground_index:
            mask = masks[j]
            # print(np.asarray(mask).shape)
            foreground = foregrounds[j]
            scale = random.uniform(config["foreground_scale_min"],config["foreground_scale_max"])
            _w,_h= mask.size
            w = int(_w*scale)
            h = int(_h*scale)
            foreground = foreground.resize((w,h),Image.Resampling.BICUBIC) 
            mask = mask.resize((w,h),Image.Resampling.BICUBIC) 
            foreground_resized.append(foreground)
            mask_resized.append(mask)

        image_a = background.copy()
        image_a = paste_foreground(image_a, foreground_resized, mask_resized, corner_x_ori, corner_y_ori,bg_scale)
        image_a.save(os.path.join(save_dir,"image_a.png"))
        image_a.save(os.path.join(save_dir,"step","00.png"))
        image_a.save(os.path.join(save_dir,"gt","00.png"))

        image_b = background.copy()
        image_b = paste_foreground(image_b, foreground_resized, mask_resized, corner_x_motion,
                                                            corner_y_motion,bg_scale)
        image_b.save(os.path.join(save_dir,"image_b.png"))
        # cv2.imwrite(os.path.join(save_dir,"image_b_flow.png"),flow_img)
        # flowwrite(flow,os.path.join(save_dir,"image_b_flow.flo"))

        image_flow = background.copy()
        image_flow = background.copy()
        image_flow = paste_foreground(image_flow, foreground_resized, mask_resized, corner_x_mag, corner_y_mag,bg_scale)
        # image_flow.save(os.path.join(save_dir,"image_flow.png
        for j in range(len(corner_x_step)):
            image_step = background.copy()
            image_step_mag = background.copy()
            # print(corner_x_step[j,:])
            image_step= paste_foreground(image_step, foreground_resized, mask_resized, corner_x_step[j,:],
                                        corner_y_step[j,:],bg_scale)
            image_step.save(os.path.join(save_dir,"step",str(j+1).zfill(2)+".png"))
            # cv2.imwrite(os.path.join(save_dir,"flow",str(j+1).zfill(2)+".png"),flow_img)
            # flowwrite(flow,os.path.join(save_dir,"flow",str(j+1).zfill(2)+".flo"))

            image_step_mag = paste_foreground(image_step_mag, foreground_resized, mask_resized, corner_x_step_mag[j,:], corner_y_step_mag[j,:],bg_scale)
            image_step_mag.save(os.path.join(save_dir,"gt",str(j+1).zfill(2)+".png"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yml")
    args = parser.parse_args()
    num_thread = 40
    main(args,0,1)
    # accelaration
    # for rank in range(num_thread):
    #     t = threading.Thread(target=main,args=(args,rank,num_thread))
    #     t.start()

