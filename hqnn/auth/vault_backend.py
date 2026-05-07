import hvac
import os
from typing import Optional


class VaultSecretBackend:
    def __init__(self, vault_addr: Optional[str] = None,
                 vault_token: Optional[str] = None):
        self.client = hvac.Client(
            url=vault_addr or os.environ["VAULT_ADDR"],
            token=vault_token or os.environ["VAULT_TOKEN"],
        )
        if not self.client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

    def store_api_key(self, key_id: str, key_data: dict) -> None:
        self.client.secrets.kv.v2.create_or_update_secret(
            path=f"hqnn/api_keys/{key_id}",
            secret=key_data,
            mount_point="secret",
        )

    def get_api_key(self, key_id: str) -> Optional[dict]:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=f"hqnn/api_keys/{key_id}",
                mount_point="secret",
            )
            return response["data"]["data"]
        except Exception:
            return None

    def store_license(self, org: str, license_data: dict) -> None:
        self.client.secrets.kv.v2.create_or_update_secret(
            path=f"hqnn/licenses/{org}",
            secret=license_data,
            mount_point="secret",
        )

    def get_license(self, org: str) -> Optional[dict]:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=f"hqnn/licenses/{org}",
                mount_point="secret",
            )
            return response["data"]["data"]
        except Exception:
            return None

    def rotate_secret(self, path: str, new_data: dict) -> None:
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=new_data,
            mount_point="secret",
        )

    def delete_secret(self, path: str) -> None:
        self.client.secrets.kv.v2.delete_metadata_and_all_versions(
            path=path,
            mount_point="secret",
        )

    def list_api_keys(self) -> list:
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path="hqnn/api_keys",
                mount_point="secret",
            )
            return response["data"]["keys"]
        except Exception:
            return []