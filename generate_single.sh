#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
SNAPSHOT=/home/lixinpeng/myDRGAN/snapshot/Single/2017-11-27_08-52-05/epoch40
DATA=/home/lixinpeng/myDRGAN/data/trainsingle_DR_GAN_list.txt

nohup $PYDIR/python main.py \
-mode gensingle -batch_size 2 -snapshot $SNAPSHOT -data_path $DATA \
2>&1 | tee $LOG&
