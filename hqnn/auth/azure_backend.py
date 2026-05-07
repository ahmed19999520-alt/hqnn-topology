import json
import os
from typing import Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


class AzureSecretBackend:
    def __init__(self, vault_url: Optional[str] = None):
        url = vault_url or os.environ["AZURE_KEYVAULT_URL"]
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=url,
                                    credential=credential)
        self.prefix = "hqnn-"

    def _normalize_name(self, name: str) -> str:
        return (self.prefix + name).replace("/", "-").replace("_", "-")

    def store_secret(self, name: str, data: dict) -> None:
        self.client.set_secret(
            self._normalize_name(name),
            json.dumps(data),
        )

    def get_secret(self, name: str) -> Optional[dict]:
        try:
            secret = self.client.get_secret(self._normalize_name(name))
            return json.loads(secret.value)
        except Exception:
            return None

    def delete_secret(self, name: str) -> None:
        poller = self.client.begin_delete_secret(
            self._normalize_name(name))
        poller.result()

    def list_secrets(self) -> list:
        return [
            s.name.replace(self.prefix, "")
            for s in self.client.list_properties_of_secrets()
            if s.name.startswith(self.prefix)
        ]