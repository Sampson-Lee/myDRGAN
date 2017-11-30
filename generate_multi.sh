#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
SNAPSHOT=/home/lixinpeng/myDRGAN/snapshot/Multi/2017-11-27_08-57-38/epoch40
DATA=/home/lixinpeng/myDRGAN/data/trainmulti_DR_GAN_list.txt
DEVICE=1

nohup $PYDIR/python main.py \
-mode genmulti -batch_size 12 -snapshot $SNAPSHOT -images_perID 6 -data_path $DATA -device $DEVICE \
2>&1 | tee $LOG&
