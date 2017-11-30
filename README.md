# myDRGAN
Pytorch implimentation of《2017CVPR-Representation Learning by Rotating Your Faces》

CVPR2017: [Representation Learning by Rotating Your Faces](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)
Reference: [kayamin's implement of DR-GAN](https://github.com/kayamin/DR-GAN)

we trained the network using multiPIE and evaluated it for face recognition.

## Usage
look at document tree:
```
F:.
│  generate_multi.sh
│  generate_single.sh
│  iden_multi.sh
│  iden_single.sh
│  main.py
│  README.md
│  train_multi.sh
│  train_single.sh
│
├─.vscode
├─data
│      datdaset guide.pdf
│      mydataset.py
│      __init__.py
│
├─inference
│      generate_image.py
│      representation_learning.py
│      __init__.py
│
├─model
│      multiple_DR_GAN_model.py
│      single_DR_GAN_model.py
│      __init__.py
│
├─train
│      train_multiple_DRGAN.py
│      train_single_DRGAN.py
│      __init__.py
│
└─util
        mylog.py
        __init__.py
```

### scripts
there are scripts to run DRGAN just as their name, we can set parameters in scripts:

generate images
```
generate_multi.sh
generate_single.sh
```
identity recognition
```
iden_multi.sh
iden_single.sh
```
train model
```
train_multi.sh
train_single.sh
```

### dataset
use `mydataset.py` to create image list for dataloader, and set porper preprocess.
```
mydataset.py
```

### utils
`mylog.py` provide some useful functions to log learning information, plot curve of loss and generate gif.

### model
`multiple_DR_GAN_model.py` is for multi_DRGAN and `single_DR_GAN_model.py` is for single_DRGAN.

### inference
`generate_image.py` is for generate images with specific pose and `representation_learning.py` is for face recognition.

### train
`train_multiple_DRGAN.py` and `train_single_DRGAN.py` contain details for training process, we can modify training policy in them.

## Experiments
### single DRGAN

batch_size:
```
64
```
prepeocess:
```
transforms.CenterCrop(160)
transforms.Scale(110)
transforms.RandomCrop(96)
```
policy:
```
if epoch<2**1+3: ratio=1+1 # 1:1
elif epoch<2**2+3: ratio=2+1 # 1:2
elif epoch<2**3+3: ratio=3+1 # 1:3
elif epoch<2**4+3: ratio=4+1 # 1:4
elif epoch<2**5+3: ratio=5+1 # 1:5
elif epoch<2**6+3: ratio=6+1 # 1:6
```
total epoches:
```
40
```

### multi DRGAN
imagesperID:
```
6
```
batch_size:
```
60
```
preprocess:
```
transforms.CenterCrop(160)
transforms.Scale(110)
transforms.RandomCrop(96)
```
policy:
```
if epoch<2**1+1: ratio=2**0+1 # 1:1
elif epoch<2**2+1: ratio=2**1+1 # 1:2
elif epoch<2**3+1: ratio=2**2+1 # 1:3
elif epoch<2**4+1: ratio=2**3+1 # 1:4
elif epoch<2**5+1: ratio=2**4+1 # 1:8
```
total epoches:
```
40
```