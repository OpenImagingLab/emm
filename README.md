
# Event-Based Motion Magnification
### [Project Page](https://openimaginglab.github.io/emm/) | [Video](https://youtu.be/WmI7bv9nqjI) | [Paper]() | [Data]() <br>

Yutian Chen, Shi Guo, Fangzheng Yu, Feng Zhang, Jinwei Gu, Tianfan Xue <br><br>



<p align="left" width="100%">
    <img src="docs/static/images/teaser.gif"  width="90%" >
</p>

- [ ] Release the real-captured dataset
- [ ] Release the synthetic testset.
- [x] Release the training and testing code.
- [x] Release the pretrained model.

## Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 


```
git clone https://github.com/yutian10/EMM
cd EMM
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## Running code
### Train
Refer to the [REAME.md](https://github.com/yutian10/EMM/blob/main/data_preparation/README.md) for instructions on preparing training data. We also provide a tiny [training set](https://github.com/yutian10/EMM/tree/main/datasets/trainset_tiny) as an example.

To train the EMM model:
```
bash train.sh
```

### Test
To test the EMM model with real-captured video:
```
bash test.sh
```  



## Citations
```

```


