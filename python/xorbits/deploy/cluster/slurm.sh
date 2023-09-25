#!/bin/bash
#SBATCH --job-name=xorbits
#SBATCH --nodes=2
#SBATCH --output=/shared_space/output.out
#SBATCH --time=00:30:00
set -x
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=16380
web_port=16379
echo "Starting SUPERVISOR at ${head_node}"
srun --nodes=1 --ntasks=1 -w "${head_node}" \
    xorbits-supervisor -H "${head_node}" -p "${port}" -w "${web_port}"&
sleep 30
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    port_i=$((port + i))
    echo "Starting WORKER $i at ${node_i}"
    srun --nodes=1 --ntasks=1 -w "${node_i}" \
        xorbits-worker -H "${node_i}"  -p "${port_i}" -s "${head_node}":"${port}"&
done
sleep 30
address=http://"${head_node}":"${web_port}"
