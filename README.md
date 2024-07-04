
# Event-Based Motion Magnification
### [Project Page](https://openimaginglab.github.io/emm/) | [Video](https://youtu.be/WmI7bv9nqjI) | [Paper](https://arxiv.org/pdf/2402.11957.pdf) | [Data](https://drive.google.com/drive/folders/1ssEE1wvnBt4EZjxoCcfoIX8FzOsTIKN8?usp=drive_link) <br>

### ECCV 2024

Yutian Chen, Shi Guo, Fangzheng Yu, Feng Zhang, Jinwei Gu, Tianfan Xue <br><br>



<p align="left" width="100%">
    <img src="docs/static/images/teaser.gif"  width="90%" >
</p>

<!-- - [ ] Release the real-captured dataset
- [ ] Release the synthetic testset.
- [x] Release the training and testing code.
- [x] Release the pretrained model. -->

## Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 


```
git clone https://github.com/OpenImagingLab/emm.git
cd emm
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## Running code
### Dataset
Refer to the [REAME.md](https://github.com/OpenImagingLab/emm/blob/main/data_preparation/README.md) for instructions on preparing training data.  We also provide a mini batch of train set and real-captured test set on [Google Drive](https://drive.google.com/drive/folders/1tzD2PRbpTJfiI9VqxovT6pB-HkVuxN-y?usp=drive_link) as an example.

### Train
To train the EMM model:
```bash
# Modify the dataroot options/train.yml 
bash train.sh
```

### Test
To test the EMM model with real-captured video:
```bash
# Modify the dataroot and the temporal filter parameters in options/train.yml 
bash test.sh
```  



## Citations
```
@article{chen2024eventbased,
      title={Event-Based Motion Magnification}, 
      author={Yutian Chen and Shi Guo and Fangzheng Yu and Feng Zhang and Jinwei Gu and Tianfan Xue},
      journal={arXiv preprint arXiv:2402.11957},
      year={2024}
}
```


