#!/usr/bin/env sh

PYDIR=/data5/lixinpeng/anaconda2/bin
SNAPSHOT=/home/lixinpeng/myDRGAN/snapshot/Single/2017-11-29_21-40-18/epoch
DATA=/home/lixinpeng/myDRGAN/data/
PROBE=("051" "050" "041" "190" "200")
DEVICE=5

for probeindex in $( seq 1 4)
do
    LOG=./iden_log-`date +%Y-%m-%d-%H:%M:%S`.log
    for epochnum in $( seq 20 40 )
    do
    $PYDIR/python main.py \
    -mode idensingle -batch_size 300 -snapshot ${SNAPSHOT}${epochnum} -device $DEVICE \
    -data_path ${DATA}iden_probe${PROBE[probeindex]}_DR_GAN_list.txt 2>&1 | tee -a $LOG
    tac $LOG | sed -n '1p' | grep -o '0\.[0-9]\+$' | tee -a acc.txt
    echo ${epochnum} | tee -a epoch.txt
    sleep 2
    done
    paste -d: epoch.txt acc.txt > ${PROBE[$probeindex]}acc.txt
    rm acc.txt epoch.txt
    echo 'find the best acc in '${PROBE[probeindex]}
    cat ${PROBE[probeindex]}acc.txt | sort -n -r -t ':' -k 2 | tee ${PROBE[probeindex]}acc.txt
done
