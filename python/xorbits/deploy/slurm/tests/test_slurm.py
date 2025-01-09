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
import shutil

import pytest

from .... import init
from .... import pandas as pd
from .. import SLURMCluster

slurm_available = shutil.which("sbatch") is not None


def test_header_core_process_memory():
    cluster = SLURMCluster(time="00:02:00", processes=4, cores=8, memory="28G")
    assert "#SBATCH" in cluster.commands
    assert "#SBATCH --cpus-per-task=8" in cluster.commands
    assert "#SBATCH --mem=28G" in cluster.commands
    assert "#SBATCH --time=00:02:00" in cluster.commands
    assert "#SBATCH -A" not in cluster.commands


def test_header_partition_account():
    cluster = SLURMCluster(
        partition_option="regular",
        account="XorbitsOnSlurm",
        processes=4,
        cores=8,
        memory="28G",
    )
    assert "#SBATCH --cpus-per-task=8" in cluster.commands
    assert "#SBATCH --mem=28G" in cluster.commands
    assert "#SBATCH -A XorbitsOnSlurm" in cluster.commands
    assert "#SBATCH --partition=regular" in cluster.commands


def test_header_work_outputdir_web():
    # Test additional parameters
    cluster = SLURMCluster(
        job_name="my_job",
        num_nodes=10,
        output_path="/path/to/output",
        work_dir="/path/to/work",
        error_path="/path/to/error",
        webport=8080,
        load_env="xorbits",
    )
    assert "#SBATCH -J my_job" in cluster.commands
    assert "#SBATCH --nodes=10" in cluster.commands
    assert "#SBATCH --output=/path/to/output" in cluster.commands
    assert "#SBATCH --chdir=/path/to/work" in cluster.commands
    assert "#SBATCH --error=/path/to/error" in cluster.commands
    assert "web_port=8080" in cluster.commands
    assert "source activate xorbits" in cluster.commands


# Construct slurm in a docker environment, so this test could only be exec when there is sbatch command supported
@pytest.mark.skipif(not slurm_available, reason="Cannot run without slurm cluster")
def test_jobscript():
    exp = SLURMCluster(
        job_name="xorbits",
        num_nodes=2,
        output_path="/shared_space/output.out",
        time="00:30:00",
    )
    address = exp.run()
    assert address == "http://c1:16379"
    init(address)
    assert repr(pd.Series([1, 2, 3]).sum()) == "6"
