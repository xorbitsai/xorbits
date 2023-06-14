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

import logging

import pytest

from ...._mars.services.cluster import WebClusterAPI

logger = logging.getLogger(__name__)

from ..utils.utils import (
    _start_kube_cluster,
    juicefs_available,
    kube_available,
    simple_job,
)


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(juicefs_available, reason="Do not need run with juicefs")
def test_run_in_kubernetes():
    with _start_kube_cluster(
        supervisor_cpu=0.5,
        supervisor_mem="1G",
        worker_cpu=0.5,
        worker_mem="1G",
        worker_cache_mem="64m",
        use_local_image=True,
        pip=["Faker"],
    ):
        simple_job()
        import pandas as pd

        def gen_data(n=100):
            from faker import Faker

            df = pd.DataFrame()
            faker = Faker()
            df["name"] = [faker.name() for _ in range(n)]
            df["address"] = [faker.address() for _ in range(n)]
            return df

        import xorbits.remote as xr

        res = xr.spawn(gen_data).to_object()
        print(res)


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(juicefs_available, reason="Do not need to run with juicefs")
@pytest.mark.asyncio
async def test_request_workers():
    with _start_kube_cluster(
        supervisor_cpu=0.2,
        supervisor_mem="1G",
        worker_cpu=0.2,
        worker_mem="1G",
        worker_cache_mem="64m",
        use_local_image=True,
    ) as cluster_client:
        cluster_api = WebClusterAPI(address=cluster_client.endpoint)
        with pytest.raises(
            ValueError, match="Please specify a `timeout` that is greater than zero"
        ):
            await cluster_api.request_workers(worker_num=1, timeout=-10)
        with pytest.raises(
            ValueError, match="Please specify a `worker_num` that is greater than zero"
        ):
            await cluster_api.request_workers(worker_num=-10, timeout=1)
        with pytest.raises(
            ValueError, match="Please specify a `worker_num` that is greater than zero"
        ):
            await cluster_api.request_workers(worker_num=0, timeout=1)
        new_workers = await cluster_api.request_workers(worker_num=1, timeout=300)
        assert len(new_workers) == 1
        with pytest.raises(TimeoutError, match="Request worker timeout"):
            await cluster_api.request_workers(worker_num=1, timeout=0)
        simple_job()


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(juicefs_available, reason="Do not need to run with juicefs")
@pytest.mark.asyncio
async def test_request_workers_insufficient():
    with _start_kube_cluster(
        supervisor_cpu=0.5,
        supervisor_mem="1G",
        worker_cpu=0.5,
        worker_mem="1G",
        worker_cache_mem="64m",
        use_local_image=True,
    ) as cluster_client:
        cluster_api = WebClusterAPI(address=cluster_client.endpoint)
        with pytest.raises(SystemError, match=r".*Insufficient cpu.*"):
            await cluster_api.request_workers(worker_num=1, timeout=30)
