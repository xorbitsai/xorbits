# -*- coding: utf-8 -*-
# Copyright 2022 XProbe Inc.
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

import glob
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from distutils.spawn import find_executable

import numpy as np
import pytest

from .... import numpy as xnp
from .. import new_cluster

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
except ImportError:
    k8s_client = k8s_config = None

logger = logging.getLogger(__name__)

XORBITS_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(xnp.__file__)))
)
TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCKER_ROOT = os.path.join((os.path.dirname(os.path.dirname(TEST_ROOT))), "docker")

kube_available = (
    find_executable("kubectl") is not None
    and find_executable("docker") is not None
    and k8s_config is not None
)


def _collect_coverage():
    dist_coverage_path = os.path.join(XORBITS_ROOT, ".dist-coverage")
    if os.path.exists(dist_coverage_path):
        # change ownership of coverage files
        if find_executable("sudo"):
            proc = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "chown",
                    "-R",
                    f"{os.geteuid()}:{os.getegid()}",
                    dist_coverage_path,
                ],
                shell=False,
            )
            proc.wait()

        # rewrite paths in coverage result files
        for fn in glob.glob(os.path.join(dist_coverage_path, ".coverage.*")):
            if "COVERAGE_FILE" in os.environ:
                new_cov_file = os.environ["COVERAGE_FILE"] + os.path.basename(
                    fn
                ).replace(".coverage", "")
            else:
                new_cov_file = fn.replace(".dist-coverage" + os.sep, "")
            shutil.copyfile(fn, new_cov_file)
        shutil.rmtree(dist_coverage_path)


def _build_docker_images():
    print(XORBITS_ROOT)
    print(DOCKER_ROOT)
    image_name = "xorbits-test-image:" + uuid.uuid1().hex
    print(os.path.exists(os.path.join(DOCKER_ROOT, "Dockerfile")))
    docker_file_path = os.path.join(DOCKER_ROOT, "Dockerfile").removeprefix(XORBITS_ROOT + "/")
    build_proc = subprocess.run(
        [
            "minikube",
            "image",
            "build",
            "-f",
            docker_file_path,
            "-t",
            image_name,
            ".",
        ],
        cwd=XORBITS_ROOT,
        check=True,
        capture_output=True
    )
    print(build_proc.stdout)
    print(build_proc.stderr)
    print('Build Done')
    print(build_proc.returncode)
    # try:
    #     build_proc = subprocess.run(
    #         [
    #             "minikube",
    #             "image",
    #             "build",
    #             "-f",
    #             os.path.join(DOCKER_ROOT, "Dockerfile"),
    #             "-t",
    #             image_name,
    #             ".",
    #         ],
    #         cwd=XORBITS_ROOT,
    #         check=True,
    #         capture_output=True,
    #         shell=True
    #     )
    #     logger.info(build_proc.stdout)
    #     logger.warning(build_proc.stderr)
    #     # if proc.wait() != 0:
    #     #     raise SystemError("Executing docker build failed.")
    # except:  # noqa: E722
    #     _remove_docker_image(image_name)
    #     raise
    return image_name


def _remove_docker_image(image_name, raises=True):
    if "CI" not in os.environ:
        # delete image iff in CI environment
        return
    proc = subprocess.Popen(["docker", "rmi", "-f", image_name])
    if proc.wait() != 0 and raises:
        raise SystemError("Executing docker rmi failed.")


def _load_docker_env():
    if os.path.exists("/var/run/docker.sock") or not shutil.which("minikube"):
        # enable nginx ingress
        ingress = subprocess.run(
            ["minikube", "addons", "enable", "ingress"], capture_output=True, check=True
        )
        print(ingress.stdout)
        print(ingress.stderr)
        logger.info(f"Stdout for ingress enable: {ingress.stdout}")
        logger.info(f"Stderr for ingress enable: {ingress.stderr}")
        return

    # proc = subprocess.Popen(["minikube", "docker-env"], stdout=subprocess.PIPE)
    # proc.wait(30)
    # for line in proc.stdout:
    #     line = line.decode().split("#", 1)[0]
    #     line = line.strip()  # type: str | bytes
    #     export_pos = line.find("export")
    #     if export_pos < 0:
    #         continue
    #     line = line[export_pos + 6 :].strip()
    #     var, value = line.split("=", 1)
    #     os.environ[var] = value.strip('"')

    # # enable nginx ingress
    # ingress = subprocess.run(
    #     ["minikube", "addons", "enable", "ingress"], capture_output=True, check=True
    # )
    # print(ingress.stdout)
    # print(ingress.stderr)
    # logger.info(f"Stdout for ingress enable: {ingress.stdout}")
    # logger.info(f"Stderr for ingress enable: {ingress.stderr}")

    # subprocess.run(
    #     ["eval", "$(minikube -p minikube docker-env)"], capture_output=True, check=True, shell=True
    # )


@contextmanager
def _start_kube_cluster(**kwargs):
    _load_docker_env()
    image_name = _build_docker_images()

    temp_spill_dir = tempfile.mkdtemp(prefix="test-xorbits-k8s-")
    api_client = k8s_config.new_client_from_config()
    kube_api = k8s_client.CoreV1Api(api_client)

    cluster_client = None
    try:
        cluster_client = new_cluster(
            api_client,
            image=image_name,
            worker_spill_paths=[temp_spill_dir],
            timeout=600,
            log_when_fail=True,
            **kwargs,
        )

        assert cluster_client.endpoint is not None

        pod_items = kube_api.list_namespaced_pod(cluster_client.namespace).to_dict()

        log_processes = []
        for item in pod_items["items"]:
            log_processes.append(
                subprocess.Popen(
                    [
                        "kubectl",
                        "logs",
                        "-f",
                        "-n",
                        cluster_client.namespace,
                        item["metadata"]["name"],
                    ]
                )
            )

        yield

        [p.terminate() for p in log_processes]
    finally:
        shutil.rmtree(temp_spill_dir)
        if cluster_client:
            try:
                cluster_client.stop(wait=True, timeout=20)
            except TimeoutError:
                pass
        _collect_coverage()
        _remove_docker_image(image_name, False)


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
def test_run_in_kubernetes():
    with _start_kube_cluster(
        supervisor_cpu=0.5,
        supervisor_mem="1G",
        worker_cpu=0.5,
        worker_mem="1G",
        worker_cache_mem="64m",
        use_local_image=True,
    ):
        a = xnp.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = xnp.ones((100, 100), chunk_size=20) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()
        print(c)

        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        np.testing.assert_array_equal(c.to_numpy(), expected.sum())
