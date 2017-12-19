#!/bin/bash
sleep 7h
PID=25720
while : ;do
	ps -fe|grep $PID |grep -v grep
	if [ $? -ne 0 ]
	then
		echo "start to test"
		/home/lixinpeng/myDRGAN/iden_single.sh
		break
	else
		echo "still training....."
	fi
	sleep 15m
done
