#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
SNAPSHOT=/home/lixinpeng/myDRGAN/snapshot/Multi/2017-11-22_21-55-28/epoch20
DATA=/home/lixinpeng/myDRGAN/data/trainmulti_DR_GAN_list.txt
DEVICE=1

nohup $PYDIR/python main.py \
-mode trainmulti -batch_size 60 -epochs 40 -save_freq 1 -images_perID 6 -data_path $DATA -device $DEVICE \
2>&1 | tee $LOG&
