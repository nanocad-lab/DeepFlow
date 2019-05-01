#11 epochs at 10% dataset size at batchsize=128 and seq=80 and nodes=2003
#therefore
#11 epochs at full dataset at batchsize=128 and seq=80 and nodes=2003
#Predicting number of epochs at datasize $dataScale times larger and given batchsize
data_scale="$1"
batch_size="$2"
seq_len=20
day_limit=1 #days
time_limit=$(($day_limit * 24 * 60 * 60)) #seconds

num_epochs=1
num_steps_per_epoch=$(($data_scale * 768648884 / ($batch_size * $seq_len)))
num_steps=$(echo "$num_epochs * $num_steps_per_epoch" | bc)
time_per_step=$(echo "$time_limit/$num_steps" | bc -l | awk {'printf"%1.8f\n",$1'})

echo BatchSize: $batch_size
echo num_epochs: $num_epochs
echo num_steps_per_epoch: $num_steps_per_epoch
echo $num_steps | awk {'printf"num_steps: %d \n",$1'}
echo time_per_step: $time_per_step
