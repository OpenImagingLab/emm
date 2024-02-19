# Dataset Preparation

## 1. Generate foreground object and mask 
We use [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html) for the foreground. 

The data is organised as follows:
```
├── background
└── VOC2010
    ├── Annotations
    ├── ImageSets
    │   ├── Action
    │   ├── Layout
    │   ├── Main
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject

```

Run this to generate foreground. 
```
python generate_foreground.py 
```
## 2. Prepare Background
We crop [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) to 256*256 for background images.

## 3. Generate Datasets
Run this to generate RGB datasets (./trainset0908).

```
python generate_motion.py --config ./config.yml 
```

+ step: the original images with sub-pixel motions

+ gt: the ground-truth of the alpha times maginified images

+ The number behind underscore is magnification factor.