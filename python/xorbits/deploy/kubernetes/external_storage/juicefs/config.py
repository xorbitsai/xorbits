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

import uuid
from typing import Optional

from ...config import KubeConfig


class SecretConfig(KubeConfig):
    """
    Configuration builder for Juicefs secret
    """

    _default_kind = "Secret"
    api_version = "v1"

    def __init__(
        self,
        metadata_url: str,
        bucket: str,
        kind: Optional[str] = None,
    ):
        self._kind = kind or self._default_kind
        self._metadata_url = metadata_url
        self._bucket = bucket

    def build(self):
        return {
            "kind": self._kind,
            "metadata": {"name": "juicefs-secret"},
            "type": "Opaque",
            "stringData": {
                "name": "jfs",
                "metaurl": self._metadata_url,
                "storage": "file",
                "bucket": self._bucket,
            },
        }


class PersistentVolumeConfig(KubeConfig):
    """
    Juicefs persistent volume builder
    """

    _default_kind = "PersistentVolume"
    api_version = "v1"

    def __init__(
        self,
        namespace: str,
        kind: Optional[str] = None,
    ):
        self._namespace = namespace
        self._kind = kind or self._default_kind

    def build(self):
        return {
            "kind": self._kind,
            "metadata": {
                "name": f"juicefs-pv-{self._namespace}",
                "labels": {"juicefs-name": f"juicefs-fs-{self._namespace}"},
            },
            "spec": {
                "capacity": {
                    "storage": "200M"
                },  # For now, JuiceFS CSI Driver doesn't support setting storage capacity. Fill in any valid string is fine. ( Reference: https://juicefs.com/docs/csi/guide/pv/ )
                "volumeMode": "Filesystem",
                "mountOptions": [
                    "subdir=/data/{ns}/{id}".format(
                        ns=self._namespace, id=uuid.uuid4().hex
                    )
                ],  # Mount in sub directory to achieve data isolation. See https://juicefs.com/docs/csi/guide/pv/#create-storage-class for more references.
                "accessModes": [
                    "ReadWriteMany"
                ],  # accessModes is restricted to ReadWriteMany because it's the most suitable mode for our system. See https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes for more reference
                "persistentVolumeReclaimPolicy": "Retain",  # persistentVolumeReclaimPolicy is restricted to Retain for Static provisioning. See https://juicefs.com/docs/csi/guide/resource-optimization/#reclaim-policy for more references.
                "csi": {
                    "driver": "csi.juicefs.com",
                    "volumeHandle": f"juicefs-pv-{self._namespace}",
                    "fsType": "juicefs",
                    "nodePublishSecretRef": {
                        "name": "juicefs-secret",
                        "namespace": self._namespace,
                    },
                },
            },
        }


class PersistentVolumeClaimConfig(KubeConfig):
    """
    Juicefs persistent volume claim builder
    """

    _default_kind = "PersistentVolumeClaim"
    api_version = "v1"

    def __init__(
        self,
        namespace: str,
        kind: Optional[str] = None,
    ):
        self._namespace = namespace
        self._kind = kind or self._default_kind

    def build(self):
        return {
            "kind": self._kind,
            "metadata": {"name": "juicefs-pvc", "namespace": self._namespace},
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "volumeMode": "Filesystem",
                "storageClassName": "",
                "resources": {"requests": {"storage": "200M"}},
                "selector": {
                    "matchLabels": {"juicefs-name": f"juicefs-fs-{self._namespace}"}
                },
            },
        }
