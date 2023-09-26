import atexit
import logging
import os
import re
import subprocess
import time

import xorbits
import xorbits.numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLURMCluster:
    def __init__(
        self,
        job_name=None,
        num_nodes=None,
        partition_option=None,
        load_env=None,
        output_dir=None,
        error_dir=None,
        work_dir=None,
        time=None,
        walltime=None,
        processes=None,
        cores=None,
        memory=None,
        account=None,
        webport=16379,
    ):
        commands = ["#!/bin/bash"]

        self.job_name = job_name
        self.num_nodes = num_nodes
        self.partition_option = partition_option
        self.output_dir = output_dir
        self.work_dir = work_dir
        self.walltime = walltime
        self.time = time
        self.processes = processes
        self.cores = cores
        self.memory = memory
        self.load_env = load_env
        self.error_dir = error_dir
        self.account = account
        slurm_params = {
            "J": self.job_name,
            "nodes": self.num_nodes,
            "partition": self.partition_option,
            "output": self.output_dir,
            "workdir": self.work_dir,
            "time": self.time,
            "ntasks": self.processes,
            "cpus-per-task": self.cores,
            "mem": self.memory,
            "A": self.account,
        }
        self.commands = None
        self.web_port = webport
        for param, value in slurm_params.items():
            if value is not None:
                if len(str(param)) > 1:
                    commands.append(f"#SBATCH --{param}={value}")
                else:
                    commands.append(f"#SBATCH -{param} {value}")

        if self.load_env:
            commands.append(f"source activate {self.load_env}")

        commands += [
            "set -x",
            'nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            "nodes_array=($nodes)",
            "head_node=${nodes_array[0]}",
            "port=16380",
            f"web_port={self.web_port}",
            'echo "Starting SUPERVISOR at ${head_node}"',
            'srun --nodes=1 --ntasks=1 -w "${head_node}" \\',
            '    xorbits-supervisor -H "${head_node}" -p "${port}" -w "${web_port}"&',
            "sleep 30",
            "worker_num=$((SLURM_JOB_NUM_NODES - 1))",
            "for ((i = 1; i <= worker_num; i++)); do",
            "    node_i=${nodes_array[$i]}",
            "    port_i=$((port + i))",
            '    echo "Starting WORKER $i at ${node_i}"',
            '    srun --nodes=1 --ntasks=1 -w "${node_i}" \\',
            '        xorbits-worker -H "${node_i}"  -p "${port_i}" -s "${head_node}":"${port}"&',
            "done",
            "sleep 300",
            'address=http://"${head_node}":"${web_port}"',
        ]

        self.commands = "\n".join(commands)

    def run(self):
        shell_commands = self.commands
        with open("slurm.sh", "w") as f:
            f.write(shell_commands)

        os.chmod("slurm.sh", 0o770)

        result = subprocess.run(["sbatch", "slurm.sh"], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Job submitted successfully.")
            self.job_id = self.get_job_id(result.stdout)
            if self.job_id:
                logger.info(f"Job ID is {self.job_id}.")
                atexit.register(self.cancel_job)
            else:
                logger.error("Could not get job ID. Cleanup upon exit is not possible.")
            return self.get_job_address()
        else:
            logger.error("Job submission failed.")
            logger.error("Output: {}".format(result.stdout))
            logger.error("Errors: {}".format(result.stderr))

            return None

    def get_job_id(self, sbatch_output):
        job_id = None
        for line in sbatch_output.split("\n"):
            if "Submitted batch job" in line:
                job_id = line.split(" ")[-1]
        return job_id

    def cancel_job(self):
        if self.job_id:
            logger.info(f"Cancelling job {self.job_id}")
            subprocess.run(["scancel", self.job_id])

    def update_head_node(self):
        try:
            if self.job_id:
                time.sleep(5)
                command = ["scontrol", "show", "job", self.job_id]
                result = subprocess.run(command, capture_output=True, text=True)
                node_list = None
                if result.returncode == 0:
                    job_info = result.stdout
                    node_list_pattern = r"NodeList=(c\[\d+-\d+\]|c\d)"
                    matches = re.search(node_list_pattern, job_info)

                if matches:
                    node_list = matches.group(1)
                    logger.info(f"NodeList:{node_list}")
                    if node_list is None:
                        raise ValueError(
                            f"Job {self.job_id} not found or NodeList information not available."
                        )
                    # get_head_node from nodelist
                    if "[" in node_list:
                        head_node = node_list.split("-")[0].replace("[", "")
                    else:
                        # when only one node
                        head_node = node_list

                    self.head_node = head_node
                    return head_node
                else:
                    logger.warning("NodeList not found in the string.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing scontrol: {e}")
        except ValueError as e:
            logger.error(f"Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

        self.head_node = None
        logger.warning("Failed to retrieve head node.")

    def get_job_address(self, retry_attempts=10, sleep_interval=30):
        # We retry several times to get job data
        for attempt in range(retry_attempts):
            try:
                self.update_head_node()
                if self.head_node is not None:
                    address = f"http://{self.head_node}:{self.web_port}"
                    return address
                else:
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying after {sleep_interval}s..."
                    )
                    time.sleep(sleep_interval)
            except Exception as e:
                logger.error(str(e))


if __name__ == "__main__":
    exp = SLURMCluster(
        job_name="xorbits",
        num_nodes=2,
        output_dir="/shared_space/output.out",
        time="00:30:00",
    )
    address = exp.run()
    logger.info(address)
    xorbits.init(address)
    test = np.random.rand(100, 100).mean()
