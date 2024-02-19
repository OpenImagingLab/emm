import os
from PIL import Image
import numpy as np
import xml.dom.minidom as xmldom
import argparse
from tqdm import tqdm

def read_file_list(root, is_train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        filenames = f.read().split()
    images = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in filenames]
    labels = [os.path.join(root, 'SegmentationObject', i + '.png') for i in filenames]
    annotations = [os.path.join(root, 'Annotations', i + '.xml') for i in filenames]
    return images, labels,annotations  # file list

def main():
    root = os.getcwd()
    foregnd_pth = os.path.join(root,"foreground")
    os.makedirs(foregnd_pth,exist_ok=True)
    os.makedirs(os.path.join(foregnd_pth,"mask"),exist_ok=True)
    os.makedirs(os.path.join(foregnd_pth,"object"),exist_ok=True)

    images, labels, annotations = read_file_list(os.path.join(root,"VOC2011"), True)

    for i in tqdm(range(len(images))):
        img = Image.open(images[i]).convert('RGB')
        label = Image.open(labels[i]).convert('RGBA')
        annotation = xmldom.parse(annotations[i])

        eles = annotation.documentElement
        objectlist = (eles.getElementsByTagName('object'))
        objects = objectlist[0] # get the red one (first one)
                
        namelist = objects.getElementsByTagName('name')
        bndbox = objects.getElementsByTagName('bndbox')
        xmin = int(bndbox[0].getElementsByTagName("xmin")[0].childNodes[0].data)
        ymin = int(bndbox[0].getElementsByTagName("ymin")[0].childNodes[0].data)
        xmax = int(bndbox[0].getElementsByTagName("xmax")[0].childNodes[0].data)
        ymax = int(bndbox[0].getElementsByTagName("ymax")[0].childNodes[0].data)
        img = img.crop((xmin,ymin,xmax,ymax))
        label = label.crop((xmin,ymin,xmax,ymax))

        label_m = np.asarray(label)
        color = [128,0,0]
        mask = (label_m[:,:,0]==color[0])&(label_m[:,:,1]==color[1])&(label_m[:,:,2]==color[2])
        mask2 = Image.fromarray(mask)
        mask2.save("./foreground/mask/"+str(i).zfill(4)+".png")
        img.save("./foreground/object/"+str(i).zfill(4)+".png")
    
if __name__ == "__main__":
    main()
