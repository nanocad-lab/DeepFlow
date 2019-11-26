import click
import subprocess
import os
import re
import math
from deviceMapping import Projection

par2cross_default = {'kp1':False, 'kp2':False, 'dp':False, 'lp':False}

def run_command(exp_config, exp_dir, mode = 'standalone', debug=False, no_launch=False, index=0, dp=1, lp=1, kp_type=-1, kp1=1, kp2=1, batch_size=32, data_scale=1, inter_derate=1, intra_derate=1, par2cross=par2cross_default):
    command = create_sbatch_enqueue_command(exp_config, 
                                            exp_dir, 
                                            mode, 
                                            index, 
                                            batch_size=batch_size, 
                                            data_scale=data_scale, 
                                            dp=dp, 
                                            lp=lp, 
                                            kp_type=kp_type, 
                                            kp1=kp1, 
                                            kp2=kp2, 
                                            inter_derate=inter_derate, 
                                            intra_derate=intra_derate, 
                                            par2cross=par2cross, 
                                            debug=debug)
    if no_launch:
        print('WARN: Not running... printing command\n    {}'
              .format(command), flush=True)
    else:
        run_slurm_command(command)
 

def create_sbatch_enqueue_command(exp_config, exp_dir, mode, index, batch_size=32, data_scale=1, dp=1, lp=1, kp_type=-1, kp1=1, kp2=1, inter_derate=1, intra_derate=1, par2cross=par2cross_default, debug=False):
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


    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if mode=='standalone':
        script = 'perf.py'
        script_args = '--exp_config {exp_config} --exp_dir {exp_dir} --debug {debug}'.format(exp_config=exp_config, exp_dir=exp_dir, debug=debug)
        exp_name = 'perf'
    else:
        script='GD_search.py'
        script_args = '--exp_config {exp_config} --exp_dir {exp_dir} --debug {debug} --index {index} --batch_size {batch_size} --data_scale {data_scale} --dp {dp} --lp {lp} --kp_type {kp_type} --kp1 {kp1} --kp2 {kp2} --inter_derate {inter_derate} --intra_derate {intra_derate} --kp1_inter {kp1_inter} --kp2_inter {kp2_inter} --dp_inter {dp_inter} --lp_intra {lp_inter}'.format(exp_config=exp_config, exp_dir=exp_dir, debug=debug, index=index, batch_size=batch_size, data_scale=data_scale, dp=dp, lp=lp, kp1=kp1, kp2=kp2, kp_type=kp_type, inter_derate=inter_derate, intra_derate=intra_derate, kp1_inter=par2cross['kp1'], kp2_inter=par2cross['kp2'], dp_inter=par2cross['dp'], lp_inter=par2cross['lp'])
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
        '-o {exp_dir}/slurm.out').format(
            num_nodes=num_nodes, exp_name=exp_name,
            cpus_per_task=cpus_per_task, gpus_per_node=gpus_per_node,
            script=script, exp_dir=exp_dir, partition=partition,
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

def outerloop_sweep(exp_config, exp_dir, debug, num_search, no_launch):
    #TODO: datascale can be optimized out of the GD search code
    data_scale = 1
    kp_type=1 #CR
    #kp_type=2 #RC
    kp1_list=[2,4,8,16,32,64]
    kp2 = 1 #kp2=1 means that we only need one row of gpu for RC
    hlp = 2 #hidden layer parallelism
    lp = hlp + 2 #layer parallelism
    wafer_dim = 8
    dp_list=[1, 32]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 258, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    exp_root = exp_dir
    print("in outerloop_sweep")
    for batch_size in batch_sizes:
        for dp in dp_list:
            for kp1 in kp1_list:
                num_gpus = lp * kp1 * dp
                num_wafer = int(math.ceil(num_gpus / float(wafer_dim * wafer_dim)))
                layouts = Projection(dp = dp, kp1 = kp1, kp2 = kp2, lp = lp, wafer_dim = wafer_dim, num_wafer = num_wafer)
                layouts.project()
                derate_factor_inter, derate_factor_intra, par2cross = layouts.get_derate_factors()
                for layout_id, (inter_derate,intra_derate, p2x) in enumerate(zip(derate_factor_inter, derate_factor_intra, par2cross)):
                    print(layout_id, batch_size, dp, kp1)
                    par2cross_encoded = ''
                    for val in p2x.values():
                        par2cross_encoded += ('T' if val else 'F') 
                    
                    parallel_strategy = "{}-k{}-k{}-d{}-l{}".format("CR" if kp_type == 1 else "RC" if kp_type == 2 else "None", dp,kp1,kp2,lp)
                    layout = "r{}-x{}-i{}-{}".format(layout_id, inter_derate, intra_derate, par2cross_encoded)
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
                              data_scale = data_scale, 
                              dp = dp, 
                              lp = lp, 
                              kp_type = kp_type, 
                              kp1 =kp1, 
                              kp2 = kp2, 
                              inter_derate = inter_derate, 
                              intra_derate = intra_derate, 
                              par2cross = p2x, 
                              no_launch = no_launch)

def GD_search(exp_config, exp_dir, debug, num_search, batch_size, data_scale, dp, lp, kp_type, kp1, kp2, inter_derate, intra_derate, par2cross, no_launch):
    for i in range(num_search):
        run_command(exp_config = exp_config, 
                    exp_dir=exp_dir + "/r" + str(i),
                    mode='GD_search', 
                    debug=debug,
                    no_launch=no_launch,
                    index=i,
                    batch_size=batch_size,
                    data_scale=data_scale,
                    dp=dp,
                    lp=lp,
                    kp_type=kp_type,
                    kp1=kp1,
                    kp2=kp2,
                    inter_derate=inter_derate,
                    intra_derate=intra_derate,
                    par2cross=par2cross)

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
def arch_search(exp_config, exp_dir, debug, no_launch, num_search):
    outerloop_sweep(exp_config = exp_config,
                    exp_dir = exp_dir, 
                    debug = debug, 
                    num_search = num_search,
                    no_launch = no_launch)


if __name__ == "__main__":
    main()
