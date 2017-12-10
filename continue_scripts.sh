#!/bin/bash
sleep 6h
PID=38493
while : ;do
	ps -fe|grep $PID |grep -v grep
	if [ $? -ne 0 ]
	then
		echo "start to test"
		/home/lixinpeng/myDRGANv3/iden_single.sh
		break
	else
		echo "still training....."
	fi
	sleep 30m
done
