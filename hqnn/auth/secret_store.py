import os
from typing import Optional
from enum import Enum


class BackendType(Enum):
    LOCAL = "local"
    VAULT = "vault"
    AWS = "aws"
    AZURE = "azure"


class SecretStore:
    def __init__(self, backend: Optional[str] = None):
        backend_type = backend or os.environ.get(
            "HQNN_SECRET_BACKEND", "local")

        if backend_type == BackendType.VAULT.value:
            from hqnn.auth.vault_backend import VaultSecretBackend
            self._backend = VaultSecretBackend()

        elif backend_type == BackendType.AWS.value:
            from hqnn.auth.aws_backend import AWSSecretBackend
            self._backend = AWSSecretBackend()

        elif backend_type == BackendType.AZURE.value:
            from hqnn.auth.azure_backend import AzureSecretBackend
            self._backend = AzureSecretBackend()

        else:
            from hqnn.auth.api_key_manager import HQNNKeyManager
            self._backend = HQNNKeyManager(
                master_password=os.environ.get(
                    "HQNN_MASTER_PASSWORD", ""))

    def store(self, name: str, data: dict) -> None:
        self._backend.store_secret(name, data)

    def get(self, name: str) -> Optional[dict]:
        return self._backend.get_secret(name)

    def delete(self, name: str) -> None:
        self._backend.delete_secret(name)