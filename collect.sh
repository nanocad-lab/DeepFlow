 #exp_dir="/mnt/scratch/newsha/MechaFlow/arch_search/exp_config_10nm_Sep23"
 exp_dir="/mnt/scratch/newsha/MechaFlow/arch_search/exp_config_SiIF_14nm_Sep24"
 for dp in 1 32; 
 do 
   for b in 32 512; 
   do 
	 for i in `seq 0 99` 
	 do
	   if [ -f $exp_dir/dp$dp/b$b/r$i/best.txt ]; 
	   then 
	   	best_time=`cat $exp_dir/dp$dp/b$b/r$i/best.txt | grep "Best" | awk {'printf"%1.2f",$3'}`; 
	   	time_limit=`cat $exp_dir/dp$dp/b$b/r$i/best.txt | grep "Limit" | awk {'printf"%1.2f",$3'}`; 
	   	dir="dp$dp/b$b/r$i"
	   	design=`cat $exp_dir/dp$dp/b$b/r$i/best.txt | grep "area_breakdown"`
		echo $dir $best_time $time_limit | awk {'printf"%s \t %1.2f \t %1.2f \t",$1,$2,$3'}
		echo $design
	  fi
	 done; 
   done; 
 done
