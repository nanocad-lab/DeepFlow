#!/bin/bash -f
TRAINPATHA=$1
TRAINPATHP=$2
DIR=`pwd`
rm selected_area_65.txt selected_area_45.txt
if [ $TRAINPATHA = "default" ]
then
cp $DIR/default_selected_area_65.txt selected_area_65.txt
cp $DIR/default_selected_area_45.txt selected_area_45.txt
else
cp $TRAINPATHA selected_area_65.txt
cp $TRAINPATHA selected_area_45.txt
fi
if [ $TRAINPATHP = "default" ]
then
cp $DIR/default_selected_power_65.txt selected_power_65.txt
cp $DIR/default_selected_power_45.txt selected_power_45.txt
else
cp $TRAINPATHP selected_power_65.txt
cp $TRAINPATHP selected_power_45.txt
fi
rm int_pow.txt leak_pow.txt sw_pow.txt total_area.txt total_pow.txt
matlab -nodisplay -nodesktop -r RBF
while [ ! -f total_pow.txt ]
do
	sleep .001s
done
STRING=`awk 'END {print $1}' total_pow.txt`
while [[ "ABKGROUP_ORION" != $STRING ]]
do
	sleep .001s
	STRING=`awk 'END {print $1}' total_pow.txt`
done
file_end=`awk 'END { print NR }' total_pow.txt`
echo -e "B\tV\tP\tF\tTotal Area:\tInternal Power:\tSwitching Power:\tLeakage Power:\tTotal Power:"
for ((i=1; i<$file_end;i++))
do 
	B=`awk -v j=$((i+1)) 'FNR == j {print $1}' params.txt`
	V=`awk -v j=$((i+1)) 'FNR == j {print $2}' params.txt`
	P=`awk -v j=$((i+1)) 'FNR == j {print $3}' params.txt`
	F=`awk -v j=$((i+1)) 'FNR == j {print $4}' params.txt`
	TOTAL_AREA=`awk -v j=$i 'FNR == j' total_area.txt`
	INT=`awk -v j=$i 'FNR == j' int_pow.txt`
	SW=`awk -v j=$i 'FNR == j' sw_pow.txt`
	LEAK=`awk -v j=$i 'FNR == j' leak_pow.txt`
	TOTAL_POWER=`awk -v j=$i 'FNR == j' total_pow.txt`
	echo -e "$B\t$V\t$P\t$F\t$TOTAL_AREA\t$INT\t$SW\t$LEAK\t$TOTAL_POWER"
done 