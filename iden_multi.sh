#!/usr/bin/env sh
PYDIR=/data5/lixinpeng/anaconda2/bin
MODELDIR=/home/lixinpeng/myDRGANv3/snapshot/Multi/2017-12-09_09-15-29/
DATA=/home/lixinpeng/myDRGAN/data/
PROBE=("051" "050" "041" "190" "200")
DEVICE=4

for probeindex in $( seq 0 4)
do
    LOG=${MODELDIR}multi_iden_log-${PROBE[probeindex]}-`date +%Y-%m-%d-%H:%M:%S`.log
    $PYDIR/python main.py \
    -mode idenmulti -batch_size 192 -modeldir ${MODELDIR} -images_perID 6 -device $DEVICE \
    -data_path ${DATA}iden_probe${PROBE[probeindex]}_DR_GAN_multi_list.txt 2>&1 | tee -a $LOG
    sleep 1
done