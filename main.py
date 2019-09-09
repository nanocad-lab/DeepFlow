import click
import subprocess
import os
import re

def run_command(exp_config, exp_dir, mode = 'standalone', debug=False, no_launch=False, index=0):
    command = create_sbatch_enqueue_command(exp_config, exp_dir, mode, index, debug)
    if no_launch:
        print('WARN: Not running... printing command\n    {}'
              .format(command), flush=True)
    else:
        run_slurm_command(command)
 

def create_sbatch_enqueue_command(exp_config, exp_dir, mode, index, debug):
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
    partition = '1080Ti,TitanXx8,M40x8'
    exp_config = exp_config


    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if mode=='standalone':
        script = 'perf.py'
        script_args = '--exp_config {exp_config} --exp_dir {exp_dir} --debug {debug}'.format(exp_config=exp_config, exp_dir=exp_dir, debug=debug)
        exp_name = 'perf'
    else:
        script='GD_search.py'
        script_args = '--exp_config {exp_config} --exp_dir {exp_dir} --debug {debug} --index {index}'.format(exp_config=exp_config, exp_dir=exp_dir, debug=debug, index=index)
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
    batch_sizes = [32, 512]
    data_scales = [1, 100]
    exp_root = exp_dir

    for data_scale in data_scales:
        for batch_size in batch_sizes:
            exp_dir='{exp_root}/d{data_scale}/b{batch_size}'.format(exp_root=exp_root, data_scale=data_scale, batch_size=batch_size)
            GD_search(exp_config, exp_dir, debug, num_search, batch_size, data_scale, no_launch)

def GD_search(exp_config, exp_dir, debug, num_search, batch_size, data_scale, no_launch):
    for i in range(num_search):
        run_command(exp_config = exp_config, 
                    exp_dir=exp_dir + "/r" + str(i),
                    mode='GD_search', 
                    debug=debug,
                    no_launch=no_launch,
                    index=i)

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
