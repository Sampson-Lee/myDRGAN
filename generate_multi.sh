#!/usr/bin/env sh
LOG=./log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/data5/lixinpeng/anaconda2/bin
MODELDIR=/home/lixinpeng/myDRGANv3/snapshot/bestmodel/
DATA=/home/lixinpeng/myDRGAN/data/trainmulti_DR_GAN_list.txt
DEVICE=1

nohup $PYDIR/python main.py \
-mode genmulti -batch_size 48 -modeldir ${MODELDIR} -images_perID 6 -data_path ${DATA} -device $DEVICE \
2>&1 | tee $LOG&
