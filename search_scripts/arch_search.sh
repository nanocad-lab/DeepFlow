exp_root="$PWD/hw_architecture_search/membw-batch"

#infinit memory bandwidth
mem_energy="6e-12"
l2_energy="6e-30" 
internode_energy="10e-30"
bank_size="1000 TB"
bank_bw="322 GB/s"
l2_bank_size="1000 TB"
l2_bank_bw="200000 GB/s" 
kernel_launch_overhead="10e-30"
precision=4
#Starting TDP
#Play with this number to strat from a reasonable point for search
min_TDP=300
###############
kernel_launch_overhead=0
line_latency=0

data_scale=100


init() {
  	exp_dir=$1
	cp ../configs/exp_config.yaml $exp_dir/exp_config.yaml
}

collect_time() {
    _tot_batch_size=$1
	_mem_per=$2
	_TDP=$3

  	exp_dir="$exp_root/data_scale${data_scale}/${_tot_batch_size}_${_mem_per}"
	
	rm -rf $exp_dir
	mkdir -p $exp_dir
	
	init $exp_dir

	_mem_percent=`echo "${_mem_per}/10000" | bc -l | awk {'printf"%1.4f",$1'}` 
	#core_th=`echo "2400000/${_TDP}" | bc -l | awk {'printf"%1.2f",$1'}` 

        #echo $_mem_percent
	
	sed -i "s/batch_size: .*/batch_size: ""${_tot_batch_size}""/" $exp_dir/exp_config.yaml
	#sed -i "s/core: .*/core: ""$core_th""/" $exp_dir/exp_config.yaml
	sed -i "s/DRAM: .*/DRAM: ""${_mem_percent}""/" $exp_dir/exp_config.yaml
	sed -i "s/DRAM_energy_per_bit_trans: .*/DRAM_energy_per_bit_trans: ""$mem_energy""/" $exp_dir/exp_config.yaml
	sed -i "s/L2_energy_per_bit: .*/L2_energy_per_bit: ""$l2_energy""/" $exp_dir/exp_config.yaml
	sed -i "s/internode_energy_per_bit: .*/internode_energy_per_bit: ""$internode_energy""/" $exp_dir/exp_config.yaml
	sed -i "s/HBM_stack_capacity: .*/HBM_stack_capacity: ""$bank_size""/" $exp_dir/exp_config.yaml
	sed -i "s#HBM_stack_bw: .*#HBM_stack_bw: ""$bank_bw""#" $exp_dir/exp_config.yaml
	sed -i "s/L2_bank_capacity: .*/L2_bank_capacity: ""$l2_bank_size""/" $exp_dir/exp_config.yaml
	sed -i "s#L2_bank_bw: .*#L2_bank_bw: ""$l2_bank_bw""#" $exp_dir/exp_config.yaml
	sed -i "s#kernel_launch_overhead: .*#kernel_launch_overhead: ""$kernel_launch_overhead""#" $exp_dir/exp_config.yaml
	sed -i "s#precision: .*#precision: ""$precision""#" $exp_dir/exp_config.yaml
	sed -i "s#TDP: .*#TDP: ""${_TDP}""#" $exp_dir/exp_config.yaml
	
	
	python ../perf.py --exp_config $exp_dir/exp_config.yaml --debug False >| $exp_dir/summary.txt 
	exec_time=`cat $exp_dir/summary.txt | grep "Time:" | awk {'print $2'}`
	time_limit=`bash time_limit.sh $data_scale ${_tot_batch_size} | grep "time_per_step" | awk {'print $2'}`
	core_throughput=`cat $exp_dir/summary.txt | grep "Throughput" | awk {'print $2'} | sed "s/Tflops,//" | awk {'printf"%5.2f",$1'}`
	mem_bw=`cat $exp_dir/summary.txt | grep "Memory Bandwidth" | awk {'print $3'} | sed "s#GB/s,##" | awk {'printf"%5.2f",$1'}`
	echo ${_tot_batch_size} ${mem_bw} $core_throughput $exec_time $time_limit ${_TDP} >| $exp_dir/result
	#echo "core_th:" $core_throughput "(TFlops/s)"	
	if (( $(echo "$exec_time < $time_limit" | bc -l) ));
	then
	  	found=true
	#else
	#    echo $_TDP $time_limit $exec_time | awk {'printf"**** %d\t %5.8f\t %5.8f\n",$1,$2,$3'}
	fi
}

tot_batch_size=1
found=false
#TDP=6072300
#TDP=201072

#while  true;
#do
#  	tot_batch_size=1
#	mem_per=10000
#	TDP=$(($TDP + $base_tdp))
#	#echo $TDP
#	collect_time $tot_batch_size $mem_per $TDP
#	if [[ $found = true ]];then
#	   echo "==========="
#	   break
#	fi
#done
#min_TDP=$TDP
found=false
first=true

xlist=()
ylist=()

echo "Batch | MemBw (%) | MemBw (GB/s) | exec_time (sec) | time_limit (sec) | TDP"
for tot_batch_size_power in `seq 1 5`
do
	#for mem_per in `seq 1 99` `seq 100 100 10000` `seq 11000 1000 100000` `seq 110000 10000 1000000` `seq 1100000 100000 10000000` `seq 11000000 1000000 100000000`
  	for mem_per in `seq 1000 2000`
        do
		collect_time $tot_batch_size $mem_per ${min_TDP}

		if [[ ( $found == true ) && ( $first == true ) ]];then
			xlist+=($mem_bw)
		  	ylist+=($exec_time)	
			first=false
		fi
		echo ${tot_batch_size} ${mem_per} ${mem_bw} ${exec_time} ${time_limit} ${_TDP} | awk '{printf"%d\t%d\t%1.2f\t%1.8f\t%1.8f\t%d\n",$1,$2,$3,$4,$5,$6}'
	 done
	found=false	
	first=true
	tot_batch_size=$(($tot_batch_size*32))
done

echo -n "["
for i in  "${xlist[@]}";
do
  echo -n $i,
done
echo "]"

echo -n "["
for i in  "${ylist[@]}";
do
  echo -n $i,
done
echo "]"
