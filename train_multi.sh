#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
SNAPSHOT=/home/lixinpeng/myDRGANv3/snapshot/Multi/2017-12-09_09-22-51/epoch39
DATA=/home/lixinpeng/myDRGAN/data/trainmulti_DR_GAN_list.txt
DEVICE=4

nohup $PYDIR/python main.py \
-mode trainmulti -batch_size 60 -epochs 10 -save_freq 1 -images_perID 6 -data_path $DATA -device $DEVICE -snapshot $SNAPSHOT \
-use_softlabel -use_noiselabel -use_switch \
2>&1 | tee $LOG&
