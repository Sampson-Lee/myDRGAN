#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
DATA=/home/lixinpeng/myDRGAN/data/trainsingle_DR_GAN_list.txt
DEVICE=4

nohup $PYDIR/python main.py \
-mode trainsingle -batch_size 64 -epochs 40 -save_freq 1 -data_path $DATA -device $DEVICE \
-use_history_epoch \
2>&1 | tee $LOG&
