# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

from abc import ABC, abstractmethod
from typing import Optional

from .juicefs.config import (
    PersistentVolumeClaimConfig,
    PersistentVolumeConfig,
    SecretConfig,
)

try:
    from kubernetes import client as kube_client
    from kubernetes.client import ApiClient
except ImportError:  # pragma: no cover
    client = None
    ApiClient = None


class ExternalStorage(ABC):
    def __init__(
        self,
        namespace: Optional[str],
        api_client: ApiClient,
    ):
        self._namespace = namespace
        self._api_client = api_client
        self._core_api = kube_client.CoreV1Api(self._api_client)

    @abstractmethod
    def build(self):
        """
        Build the external storage
        """


class JuicefsK8SStorage(ExternalStorage):
    def __init__(
        self,
        namespace: Optional[str],
        api_client: ApiClient,
        external_storage_config: Optional[dict],
    ):
        super().__init__(namespace=namespace, api_client=api_client)
        self._external_storage_config = external_storage_config

    def build(self):
        """
        Create pv, secret, and pvc
        """
        secret_config = SecretConfig(
            metadata_url=self._external_storage_config["metadata_url"],
            bucket=self._external_storage_config["bucket"],
        )

        persistent_volume_config = PersistentVolumeConfig(namespace=self._namespace)

        persistent_volume_claim_config = PersistentVolumeClaimConfig(
            namespace=self._namespace
        )
        self._core_api.create_namespaced_secret(self._namespace, secret_config.build())
        self._core_api.create_persistent_volume(persistent_volume_config.build())
        self._core_api.create_namespaced_persistent_volume_claim(
            self._namespace, persistent_volume_claim_config.build()
        )
