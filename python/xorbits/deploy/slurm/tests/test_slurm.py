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

from .... import init
from ... import pandas as pd
from .. import SLURMCluster


def test_header():
    with SLURMCluster(time="00:02:00", processes=4, cores=8, memory="28G") as cluster:
        assert "#SBATCH" in cluster.commands
        assert "#SBATCH -n 1" in cluster.commands
        assert "#SBATCH --cpus-per-task=8" in cluster.commands
        assert "#SBATCH --mem28G" in cluster.commands
        assert "#SBATCH -t 00:02:00" in cluster.commands
        assert "#SBATCH -p" not in cluster.commands
        # assert "#SBATCH -A" not in cluster.commands

    with SLURMCluster(
        queue="regular",
        account="XorbitsOnSlurm",
        processes=4,
        cores=8,
        memory="28G",
    ) as cluster:
        assert "#SBATCH --cpus-per-task=8" in cluster.commands
        assert "#SBATCH --mem=28G" in cluster.commands
        assert "#SBATCH -t " in cluster.commands
        assert "#SBATCH -A XorbitsOnSlurm" in cluster.commands
        assert "#SBATCH --partion regular" in cluster.commands


def test_jobscript():
    exp = SLURMCluster(
        job_name="xorbits",
        num_nodes=2,
        output_dir="/shared_space/output.out",
        time="00:30:00",
    )
    address = exp.run()
    assert address == "http://c1:16379"
    init(address)
    assert repr(pd.Series([1, 2, 3]).sum()) == "6"
