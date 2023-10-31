#!/bin/bash
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
docker-compose up -d --no-build

while [ `./register_cluster.sh 2>&1 | grep "sacctmgr: error" | wc -l` -ne 0 ]
  do
    echo "Waiting for SLURM cluster to become ready";
    sleep 2
  done
echo "SLURM properly configured"

# On some clusters the login node does not have the same interface as the
# compute nodes. The next three lines allow to test this edge case by adding
# separate interfaces on the worker and on the scheduler nodes.
docker exec slurmctld ip addr add 10.1.1.20/24 dev eth0 label eth0:scheduler
docker exec c1 ip addr add 10.1.1.21/24 dev eth0 label eth0:worker
docker exec c2 ip addr add 10.1.1.22/24 dev eth0 label eth0:worker
