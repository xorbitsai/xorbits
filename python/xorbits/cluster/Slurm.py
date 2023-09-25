import os
import subprocess
import time
import atexit
import xorbits
import xorbits.numpy as np

DEFAULT_JOB_NAME = "default_job"
DEFAULT_NUMBER_NODES = 2
DEFAULT_PARTITION_OPTION = "batch"
DEFAULT_LOAD_ENV = "LOAD_ENV"

class SLURMCluster:
    def __init__(self,
                 job_name=None,
                 num_nodes=None,
                 partition_option=None,
                 load_env=None,
                 output_dir=None,
                 error_dir=None,
                 work_dir=None,
                 time=None,
                 processes=None,
                 cores=None,
                 memory=None,
                 account=None,):

        commands = ["#!/bin/bash"]

        self.job_name = job_name
        self.num_nodes = num_nodes
        self.partition_option = partition_option
        self.output_dir = output_dir
        self.work_dir = work_dir
        self.walltime = time
        self.processes = processes
        self.cores = cores
        self.memory = memory
        self.account = account
        self.load_env = load_env
        self.commands = None

        slurm_params = {
            "job-name": self.job_name,
            "nodes": self.num_nodes,
            "partition": self.partition_option,
            "output": self.output_dir,
            "workdir": self.work_dir,
            "time": self.walltime,
            "ntasks": self.processes,
            "cpus-per-task": self.cores,
            "mem": self.memory,
        }

        for param, value in slurm_params.items():
            if value is not None:
                commands.append(f"#SBATCH --{param}={value}")

        if self.load_env:
            commands.append(f"source activate {self.load_env}")

        commands +=  [
                    "set -x",
                    'nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")',
                    "nodes_array=($nodes)",
                    "head_node=${nodes_array[0]}",
                    "port=16380",
                    "web_port=16379",
                    'echo "Starting SUPERVISOR at ${head_node}"',
                    'srun --nodes=1 --ntasks=1 -w "${head_node}" \\',
                    '    xorbits-supervisor -H "${head_node}" -p "${port}" -w "${web_port}"&',
                    "sleep 10",
                    'worker_num=$((SLURM_JOB_NUM_NODES - 1))',
                    'for ((i = 1; i <= worker_num; i++)); do',
                    '    node_i=${nodes_array[$i]}',
                    '    port_i=$((port + i))',
                    '    echo "Starting WORKER $i at ${node_i}"',
                    '    srun --nodes=1 --ntasks=1 -w "${node_i}" \\',
                    '        xorbits-worker -H "${node_i}"  -p "${port_i}" -s "${head_node}":"${port}"&',
                    'done',
                    "sleep 5",
                    'address=http://"${head_node}":"${web_port}"',
                ]

        self.commands = "\n".join(commands)

    def run(self):
        shell_commands = self.commands
        with open("slurm.sh", 'w') as f:
            f.write(shell_commands)

        os.chmod("slurm.sh", 0o770)

        result = subprocess.run(["sbatch", "slurm.sh"], capture_output=True, text=True)

        if result.returncode == 0:
            print("Job submitted successfully.")
            self.job_id = self.get_job_id(result.stdout)
            if self.job_id:
                print(f"Job ID is {self.job_id}.")
                atexit.register(self.cancel_job)
            else:
                print("Could not get job ID. Cleanup upon exit is not possible.")
            return self.get_job_address()
        else:
            print("Job submission failed.")
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
            return None

    def get_job_id(self, sbatch_output):
        job_id = None
        for line in sbatch_output.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split(" ")[-1]
        return job_id

    def cancel_job(self):
        if self.job_id:
            print(f"Cancelling job {self.job_id}")
            subprocess.run(["scancel", self.job_id])

    def update_head_node(self):
        try:
            if self.job_id:
                command = ["scontrol", "show", "job", self.job_id]
                result = subprocess.run(command, capture_output=True, text=True)
                node_list = None
                if result.returncode == 0:
                    output = result.stdout
                    print(output)
                    for line in output.split('\n'):
                        if line.startswith("NodeList="):
                            node_list = line[len("NodeList="):].strip()
                            break
                    if node_list is None:
                        raise ValueError(f"Job {self.job_id} not found or NodeList information not available.")
                # 提取头节点（第一个节点）
                    self.head_node = node_list.split()[0]
        except subprocess.CalledProcessError as e:
            print(f"Error executing scontrol: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        self.head_node = None
        print("Failed to retrieve head node.")
    
    def get_job_address(self, retry_attempts=10, sleep_interval=30):
        # We retry several times to get job data
        for attempt in range(retry_attempts):
            try:
                self.update_head_node()
                if self.head_node is not None:
                    self.head_node = "eval"
                    address=f"http://{self.head_node}:{self.web_port}"
                    return address
                else:
                    print(f"Attempt {attempt + 1} failed, retrying after {sleep_interval}s...")
                    time.sleep(sleep_interval)
            except Exception as e:
                print(str(e))

if __name__ == "__main__":
    exp = SLURMCluster(job_name="xorbits",num_nodes=1,output_dir="/shared_space/output.out",time="00:30:00")
    adress = exp.run()
    print(adress)
    time.sleep(5)
    adress = "http://c1:16379"
    xorbits.init(adress)
    print(np.random.rand(100, 100).mean())