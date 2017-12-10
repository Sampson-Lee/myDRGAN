#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
MODELDIR=/home/lixinpeng/myDRGANv3/snapshot/bestmodel/
DATA=/home/lixinpeng/myDRGAN/data/trainsingle_DR_GAN_list.txt
DEVICE=0

nohup $PYDIR/python main.py \
-mode gensingle -batch_size 64 -modeldir ${MODELDIR} -data_path $DATA -device $DEVICE \
2>&1 | tee $LOG&
