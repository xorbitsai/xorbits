#!/usr/bin/env bash

function jobqueue_before_install {
    docker version
    docker-compose version

    # start slurm cluster
    cd ./CI/slurm
    docker-compose pull
    ./start-slurm.sh
    cd -

    #Set shared space permissions
    docker exec slurmctld /bin/bash -c "chmod -R 777 /shared_space"

    docker ps -a
    docker images
    show_network_interfaces
}

function show_network_interfaces {
    for c in slurmctld c1 c2; do
        echo '------------------------------------------------------------'
        echo docker container: $c
        docker exec $c python -c 'import psutil; print(psutil.net_if_addrs().keys())'
        echo '------------------------------------------------------------'
    done
}

function jobqueue_install {
    docker exec slurmctld /bin/bash -c "cd xorbits/python/; pip install -e ."
}

function jobqueue_script {
    docker exec c1 /bin/bash -c "pip install xorbits"
    docker exec c2 /bin/bash -c "pip install xorbits"
    docker exec slurmctld /bin/bash -c "python /xorbits/python/xorbits/deploy/cluster/Slurm.py"
    docker exec slurmctld /bin/bash -c "cat /shared_space/output.out"
}

function jobqueue_after_script {
    docker exec slurmctld bash -c 'sinfo'
    docker exec slurmctld bash -c 'squeue'
    docker exec slurmctld bash -c 'sacct -l'
}
