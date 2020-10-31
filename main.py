import click
import subprocess
import os
import shutil
import re
import math
from deviceMapping import Projection

par2cross_default = {'kp1':False, 'kp2':False, 'dp':False, 'lp':False}

def run_command(exp_config, exp_dir, mode = 'standalone', debug=False, no_launch=False, index=0, dp=1, lp=1, kp_type=-1, kp1=1, kp2=1, batch_size=256, hidden_dim=19968, data_scale=1, inter_derate=1, intra_derate=1, par2cross=par2cross_default, wafer_dim=1):
    command = create_sbatch_enqueue_command(exp_config, 
                                            exp_dir, 
                                            mode, 
                                            index, 
                                            batch_size=batch_size,
                                            hidden_dim=hidden_dim,
                                            data_scale=data_scale, 
                                            dp=dp, 
                                            lp=lp, 
                                            kp_type=kp_type, 
                                            kp1=kp1, 
                                            kp2=kp2, 
                                            inter_derate=inter_derate, 
                                            intra_derate=intra_derate, 
                                            par2cross=par2cross, 
                                            wafer_dim=wafer_dim,
                                            debug=debug)
    
    output_file = "{}/command.txt".format(exp_dir)
    with open(output_file, "a+") as f:
      f.write(command)
    
    if no_launch:
        print('WARN: Not running... printing command\n    {}'
              .format(command), flush=True)
    else:
        run_slurm_command(command)
 

def create_sbatch_enqueue_command(exp_config, exp_dir, mode, index, batch_size=256, hidden_dim=19968, data_scale=1, dp=1, lp=1, kp_type=-1, kp1=1, kp2=1, inter_derate=1, intra_derate=1, par2cross=par2cross_default, wafer_dim=1, debug=False):
    """Create a job SLURM enqueue command
    Args:
        exp_dir (str): Path to model directory
    """
    num_nodes = 1
    num_tasks_per_node = 1
    exp_name = 'hw_search'
    num_tasks = 1
    cpus_per_task = 1
    gpus_per_node = 1
    partition = '1080Ti,TitanXx8,M40x8,2080Ti,P100'
    exp_config = exp_config

    tmp_dir = re.sub("/mnt/scratch","/tmp", exp_dir)
    slurm_dir = re.sub("scratch","home", exp_dir)

    print(exp_dir)
    if os.path.exists(exp_dir):
      shutil.rmtree(exp_dir, ignore_errors=True)
    if os.path.exists(tmp_dir):
      shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(slurm_dir):
      shutil.rmtree(slurm_dir, ignore_errors=True)
    
    os.makedirs(exp_dir)
    os.makedirs(tmp_dir)
    os.makedirs(slurm_dir)

    if mode=='standalone':
        script = 'perf.py'
        script_args = '--exp_config {exp_config} --exp_dir {temp_dir} --debug {debug}'.format(exp_config=exp_config, tmp_dir=tmp_dir, debug=debug)
        exp_name = 'perf'
    else:
        script='GD_search.py'
        script_args = '--exp_config {exp_config} --exp_dir {tmp_dir} --debug {debug} --index {index} --batch_size {batch_size} --hidden_dim {hidden_dim} --data_scale {data_scale} --dp {dp} --lp {lp} --kp_type {kp_type} --kp1 {kp1} --kp2 {kp2} --inter_derate {inter_derate} --intra_derate {intra_derate} --kp1_inter {kp1_inter} --kp2_inter {kp2_inter} --dp_inter {dp_inter} --lp_inter {lp_inter} --wafer_dim {wafer_dim}'.format(exp_config=exp_config, tmp_dir=tmp_dir, debug=debug, index=index, batch_size=batch_size, hidden_dim=hidden_dim, data_scale=data_scale, dp=dp, lp=lp, kp1=kp1, kp2=kp2, kp_type=kp_type, inter_derate=inter_derate, intra_derate=intra_derate, kp1_inter=par2cross['kp1'], kp2_inter=par2cross['kp2'], dp_inter=par2cross['dp'], lp_inter=par2cross['lp'], wafer_dim=wafer_dim)
        exp_name = 'GD_search'

    command = (
        'sbatch -N {num_nodes} '
        '--job-name={exp_name} '
        '--ntasks {num_tasks} '
        '--ntasks-per-node {num_tasks_per_node} '
        '--cpus-per-task {cpus_per_task} '
        '--gres=gpu:{gpus_per_node} '
        '--wrap "srun stdbuf -i0 -o0 -e0 '
        'python {script} ' +
        script_args +
        '" '
        '--partition={partition} '
        '-o {slurm_dir}/slurm-%j.out').format(
            num_nodes=num_nodes, exp_name=exp_name,
            cpus_per_task=cpus_per_task, gpus_per_node=gpus_per_node,
            script=script, slurm_dir=slurm_dir, partition=partition,
            script_args=script_args,
            num_tasks=num_tasks,
            num_tasks_per_node=num_tasks_per_node
    )
    return command

def run_slurm_command(command):
    """ Run slurm command.
    Args:
        command (str): Slurm command to execute.

    Returns:
        job_id (int): The slurm-assigned ID of the job.
    """
    output = subprocess.check_output(
        command, stderr=subprocess.STDOUT, shell=True,
    ).decode("utf-8")
    job_id = re.search(r"[Jj]ob [0-9]+", output).group(0)
    job_id = int(job_id.split(' ')[1])
    
    print("JOB {} Submitted!".format(job_id), flush=True)
    
    return job_id

def findMultipliers(n, curr_depth, results, result, max_depth):
    if curr_depth == max_depth:
        result.append(n)
        results.append(result)
        return
    i = 1
    while i <= n:
        r = result[:]
        r.append(i)
        findMultipliers(n//i, curr_depth+1, results, r, max_depth)
        i = i * 2


def outerloop_sweep(exp_config, exp_dir, debug, num_search, no_launch, num_wafer=1, wafer_dim=8, num_worker=1, batch_size=256, hidden_dim=19968):
    #TODO: datascale can be optimized out of the GD search code
    data_scale = 1
    #kp_type=1 #CR
    #kp_type=2 #RC
    #wafer_dim = 16
    #num_wafer = 1
    #num_gpus  = num_wafer * wafer_dim * wafer_dim
    num_gpus = num_worker
    #batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    #batch_sizes = [256]
    #batch_sizes=[1, 16, 4096]
    exp_root = exp_dir
    #Column-Row
    #We only need to specify kp and dp
    #lp can only be 1 or 4
    for kp_type in [1, 2]:
      if kp_type == 1:
        for lp in [1, 4]:
            parallelism_strategies = []
            result = []
            findMultipliers(num_gpus // lp, 1, parallelism_strategies, result, max_depth=2)
            for ps in parallelism_strategies:
                kp1 = ps[0]
                kp2 = 1
                dp = ps[1]
                if dp > batch_size:
                  continue
                print("kp1: {}, kp2: {}, dp : {}, lp: {}".format(kp1, kp2, dp, lp))
                p = Projection(dp = dp, kp1 = kp1, kp2 = kp2, lp = lp, wafer_dim = wafer_dim, num_wafer = num_wafer)
                for layout_id in range(0, len(p.order)):
                    p.project(layout_id)
                    inter_derate, intra_derate, par2cross = p.get_derate_factors(layout_id)
                    #print(layout_id, batch_size, dp, kp1)
                    par2cross_encoded = ''
                    for val in par2cross.values():
                        par2cross_encoded += ('T' if val else 'F') 
                    
                    parallel_strategy = "{}-k{}-k{}-d{}-l{}".format("CR" if kp_type == 1 else "RC" if kp_type == 2 else "None", kp1,kp2,dp,lp)
                    layout = "layout{}-x{}-i{}-{}".format(layout_id, inter_derate, intra_derate, par2cross_encoded)
                    exp_dir='{exp_root}/{parallel_strategy}/{layout}/b{batch_size}'.format(exp_root=exp_root, 
                                                                                           data_scale=data_scale, 
                                                                                           batch_size=batch_size,
                                                                                           parallel_strategy = parallel_strategy, 
                                                                                           layout= layout)
                    #print(exp_dir)
                    GD_search(exp_config = exp_config, 
                              exp_dir = exp_dir, 
                              debug = debug, 
                              num_search = num_search, 
                              batch_size = batch_size,
                              hidden_dim = hidden_dim,
                              data_scale = data_scale, 
                              dp = dp, 
                              lp = lp, 
                              kp_type = kp_type, 
                              kp1 = kp1, 
                              kp2 = kp2, 
                              inter_derate = inter_derate, 
                              intra_derate = intra_derate, 
                              par2cross = par2cross,
                              wafer_dim = wafer_dim,
                              no_launch = no_launch)
      elif kp_type == 2:
        for lp in [1, 4]:
            parallelism_strategies = []
            result = []
            #for CR max_depth = 2
            #for RC max_depth = 3
            findMultipliers(num_gpus // lp, 1, parallelism_strategies, result, max_depth=3)
            for ps in parallelism_strategies:
                kp1 = ps[0]
                kp2 = ps[1]
                dp = ps[2]
                print("kp1: {}, kp2: {}, dp : {}, lp: {}".format(kp1, kp2, dp, lp))
                p = Projection(dp = dp, kp1 = kp1, kp2 = kp2, lp = lp, wafer_dim = wafer_dim, num_wafer = num_wafer)
                for layout_id in range(0, len(p.order)):
                    p.project(layout_id)
                    inter_derate, intra_derate, par2cross = p.get_derate_factors(layout_id)
                    #print(layout_id, batch_size, dp, kp1)
                    par2cross_encoded = ''
                    for val in par2cross.values():
                        par2cross_encoded += ('T' if val else 'F') 
                    
                    parallel_strategy = "{}-k{}-k{}-d{}-l{}".format("CR" if kp_type == 1 else "RC" if kp_type == 2 else "None", kp1,kp2,dp,lp)
                    layout = "layout{}-x{}-i{}-{}".format(layout_id, inter_derate, intra_derate, par2cross_encoded)
                    exp_dir='{exp_root}/{parallel_strategy}/{layout}/b{batch_size}'.format(exp_root=exp_root, 
                                                                                           data_scale=data_scale, 
                                                                                           batch_size=batch_size,
                                                                                           parallel_strategy = parallel_strategy, 
                                                                                           layout= layout)
                    #print(exp_dir)
                    GD_search(exp_config = exp_config, 
                              exp_dir = exp_dir, 
                              debug = debug, 
                              num_search = num_search, 
                              batch_size = batch_size,
                              hidden_dim = hidden_dim,
                              data_scale = data_scale, 
                              dp = dp, 
                              lp = lp, 
                              kp_type = kp_type, 
                              kp1 =kp1, 
                              kp2 = kp2, 
                              inter_derate = inter_derate, 
                              intra_derate = intra_derate, 
                              par2cross = par2cross, 
                              wafer_dim = wafer_dim,
                              no_launch = no_launch)

      else:
          NotImplemented

def GD_search(exp_config, exp_dir, debug, num_search, batch_size, hidden_dim, data_scale, dp, lp, kp_type, kp1, kp2, inter_derate, intra_derate, par2cross, wafer_dim, no_launch):
    for i in range(num_search):
        run_command(exp_config = exp_config, 
                    exp_dir=exp_dir + "/r" + str(i),
                    mode='GD_search', 
                    debug=debug,
                    no_launch=no_launch,
                    index=i,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    data_scale=data_scale,
                    dp=dp,
                    lp=lp,
                    kp_type=kp_type,
                    kp1=kp1,
                    kp2=kp2,
                    inter_derate=inter_derate,
                    intra_derate=intra_derate,
                    par2cross=par2cross,
                    wafer_dim=wafer_dim)

@click.group()
def main():
    pass

@main.command("standalone")        
@click.option("--exp_config", help="Path to template config", required=True)
@click.option("--exp_dir", help="Checkpoint/log directory", required=True)
@click.option("--debug", help="debug", default=False, type=bool)
@click.option("--no_launch", help="Don't launch job, just print command", default=False)
def standalone(exp_config, exp_dir, debug, no_launch):
    run_command(exp_config = exp_config, 
                exp_dir=exp_dir, 
                mode='standalone', 
                debug=debug, 
                no_launch=no_launch)

@main.command("arch_search")        
@click.option("--exp_config", help="path to experiment config", required=True)
@click.option("--exp_dir", help="Checkpoint/log directory", required=True)
@click.option("--debug", help="debug", default=False, type=bool)
@click.option("--no_launch", help="Don't launch job, just print command", default=False, type=bool)
@click.option("--num_search", help="Number of times to search the space from different starting point", default=100)
@click.option("--num_wafer", help="Number of wafers", default=1)
@click.option("--wafer_dim", help="Wafer dimension", default=8)
@click.option("--num_worker", help="Number of parallel workers", default=-1)
@click.option("--batch_size", help="Batch Size", default=256)
@click.option("--hidden_dim", help="Dimensions of hidden layers", default=19968)
def arch_search(exp_config, exp_dir, debug, no_launch, num_search, num_wafer, wafer_dim, num_worker, batch_size, hidden_dim):
    #if number of parallel workers is not specified (default), 
    #we assume that all resources in the systems are parallel workers.
    #Specify num_workers only if number of parallel workers 
    #is less than the available resources in the system.
    #eg. use only 8 GPUs out of 64 GPUs on a wafer
    if num_worker == -1:
      num_worker = num_wafer * wafer_dim * wafer_dim
    outerloop_sweep(exp_config = exp_config,
                    exp_dir = exp_dir, 
                    debug = debug, 
                    num_search = num_search,
                    no_launch = no_launch,
                    num_wafer = num_wafer,
                    wafer_dim = wafer_dim,
                    num_worker = num_worker,
                    batch_size = batch_size,
                    hidden_dim = hidden_dim)

def call_arch_search(exp_config, exp_dir, debug, no_launch, num_search, num_wafer, wafer_dim, num_worker, batch_size, hidden_dim):
    #if number of parallel workers is not specified (default), 
    #we assume that all resources in the systems are parallel workers.
    #Specify num_workers only if number of parallel workers 
    #is less than the available resources in the system.
    #eg. use only 8 GPUs out of 64 GPUs on a wafer
    if num_worker == -1:
      num_worker = num_wafer * wafer_dim * wafer_dim
    outerloop_sweep(exp_config = exp_config,
                    exp_dir = exp_dir, 
                    debug = debug, 
                    num_search = num_search,
                    no_launch = no_launch,
                    num_wafer = num_wafer,
                    wafer_dim = wafer_dim,
                    num_worker = num_worker,
                    batch_size = batch_size,
                    hidden_dim = hidden_dim)




if __name__ == "__main__":
    main()
