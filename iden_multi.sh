#!/usr/bin/env sh
PYDIR=/data5/lixinpeng/anaconda2/bin
SNAPSHOT=/home/lixinpeng/myDRGAN/snapshot/Multi/2017-11-29_15-37-30/epoch
DATA=/home/lixinpeng/myDRGAN/data/
PROBE=("051" "050" "041" "190" "200")
DEVICE=0

for probeindex in $( seq 0 4)
do
    LOG=./iden_log-`date +%Y-%m-%d-%H:%M:%S`.log
    for epochnum in $( seq 25 40 )
    do
    $PYDIR/python main.py \
    -mode idenmulti -batch_size 180 -snapshot ${SNAPSHOT}${epochnum} -images_perID 6 -device $DEVICE \
    -data_path ${DATA}iden_probe${PROBE[probeindex]}_DR_GAN_list.txt 2>&1 | tee -a $LOG
    tac $LOG | sed -n '1p' | grep -o '0\.[0-9]\+$' | tee -a multiacc.txt
    echo ${epochnum} | tee -a multiepoch.txt
    sleep 2
    done
    paste -d: multiepoch.txt multiacc.txt > ${PROBE[$probeindex]}multiacc.txt
    rm multiacc.txt multiepoch.txt
    echo 'find the best acc in '${PROBE[probeindex]}
    cat ${PROBE[probeindex]}multiacc.txt | sort -n -r -t ':' -k 2 | tee ${PROBE[probeindex]}multiacc.txt
done