#!/usr/bin/env sh
PYDIR=/data5/lixinpeng/anaconda2/bin
MODELDIR=/home/lixinpeng/myDRGAN/snapshot/Single/2017-12-18_10-59-28/
DATA=/home/lixinpeng/myDRGAN/data/
PROBE=("051" "050" "041" "190" "200")
DEVICE=3

for probeindex in $( seq 0 4)
do
    LOG=${MODELDIR}single_iden_log-${PROBE[probeindex]}-`date +%Y-%m-%d-%H:%M:%S`.log
    $PYDIR/python main.py \
    -mode idensingle -batch_size 250 -modeldir ${MODELDIR} -device $DEVICE \
    -data_path ${DATA}iden_probe${PROBE[probeindex]}_DR_GAN_single_list.txt 2>&1 | tee -a $LOG
    sleep 1
done
