#!/bin/bash -f
NEWPATH=$1
if [ -e params.txt ]
then
	rm params.txt
fi

if [ $NEWPATH = "default" ]
then

TECH=`grep "PARM_TECH_POINT" SIM_port.h | awk '{print $3}' `
B=`grep "PARM_in_buf_set" SIM_port.h | awk '{print $3}' `
V=`grep "PARM_v_channel" SIM_port.h | awk '{print $3}' `
P=`grep "PARM_in_port" SIM_port.h | awk '{print $3}' `
F=`grep "PARM_flit_width" SIM_port.h | awk '{print $3}' `


echo $TECH >> params.txt
echo $B $V $P $F >> params.txt

else

TECH=`grep "PARM_TECH_POINT" SIM_port.h | awk '{print $3}' `
echo $TECH >> params.txt
cat $NEWPATH >> params.txt

fi