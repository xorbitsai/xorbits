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
        metadata_url: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        self._namespace = namespace
        self._api_client = api_client
        self._core_api = kube_client.CoreV1Api(self._api_client)
        self._metadata_url = metadata_url
        self._bucket = bucket

    def build(self):
        """
        Create pv, secret, and pvc
        """
        secret_config = SecretConfig(
            metadata_url=self._metadata_url, bucket=self._bucket
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
