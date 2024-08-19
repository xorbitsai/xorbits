#!/usr/bin/env bash
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

function jobqueue_before_install {
    docker version
    docker compose version

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
    docker exec slurmctld /bin/bash -c \
          "pytest --ignore xorbits/_mars/ --timeout=1500 \
            -W ignore::PendingDeprecationWarning \
            --cov-config=setup.cfg --cov-report=xml --cov=xorbits xorbits/deploy/slurm"
}

function jobqueue_after_script {
    docker exec slurmctld bash -c 'sinfo'
    docker exec slurmctld bash -c 'squeue'
    docker exec slurmctld bash -c 'sacct -l'
}