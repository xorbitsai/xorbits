# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import logging
import os
import re
import subprocess
import time

# Configure logging
logger = logging.getLogger(__name__)


class SLURMCluster:
    def __init__(
        self,
        job_name=None,
        num_nodes=None,
        partition_option=None,
        load_env=None,
        output_path=None,
        error_path=None,
        work_dir=None,
        time=None,
        processes=None,
        cores=None,
        memory=None,
        account=None,
        webport=16379,
        **kwargs,
    ):
        """
        The entrance of deploying a SLURM cluster.

        Parameters
        ----------
        job_name : str, optional
            Name of the Slurm job, by default None
        num_nodes : int, optional
            Number of nodes in the Slurm cluster, by default None
        partition_option : str, optional
            Request a specific partition for the resource allocation, by default None
        load_env : str, optional
            Conda Environment to load, by default None
        output_path : str, optional
            Path for Log output, by default None
        error_path : str, optional
            Path for Log error, by default None
        work_dir : str, optional
            Slurm‘s Working directory,the default place to receive the logs and result, by default None
        time : str, optional
            Minimum time limit on the job allocation, by default None
        processes : int, optional
            Number of processes, by default None
        cores : int, optional
            Number of cores, by default None
        memory : str, optional
            Specify the real memory required per node. Default units are megabytes, by default None
        account : str, optional
            Charge resources used by this job to specified account, by default None
        webport : int, optional
            Xorbits' Web port, by default 16379
        If user have some specifics needing for can just follow the slurm interface we add it at the end automatically
        """
        commands = ["#!/bin/bash"]

        self.job_name = job_name
        self.num_nodes = num_nodes
        self.partition_option = partition_option
        self.output_path = output_path
        self.work_dir = work_dir
        self.time = time
        self.processes = processes
        self.cores = cores
        self.memory = memory
        self.load_env = load_env
        self.error_path = error_path
        self.account = account
        slurm_params = {
            "J": self.job_name,
            "nodes": self.num_nodes,
            "partition": self.partition_option,
            "error": self.error_path,
            "output": self.output_path,
            "chdir": self.work_dir,
            "time": self.time,
            "ntasks": self.processes,
            "cpus-per-task": self.cores,
            "mem": self.memory,
            "A": self.account,
            **kwargs,
        }
        self.commands = None
        self.web_port = webport
        for param, value in slurm_params.items():
            if value is not None:
                # there are two modes of sbatch, one is like --time, the other one is like -A，so i just judge it by using len
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
        # here I give a very long sleep time to avoid when supervisor nodes don't start, and the other node can't find the supervisor node
        self.commands = "\n".join(commands)
        self.sbatch_out = ""

    def run(self):
        shell_commands = self.commands
        with open("slurm.sh", "w") as f:
            f.write(shell_commands)

        os.chmod("slurm.sh", 0o770)

        result = subprocess.run(["sbatch", "slurm.sh"], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Job submitted successfully.")
            self.sbatch_out = result.stdout
            self.job_id = self.get_job_id()
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

    def get_job_id(self):
        sbatch_output = self.sbatch_out
        job_id = None
        for line in sbatch_output.split("\n"):
            if "Submitted batch job" in line:
                job_id = line.split(" ")[-1]
        return job_id

    def get_sbatch_out(self):
        logging.info(f"getting batch_out:{self.sbatch_out}")
        return self.sbatch_out

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
