import secrets
import hashlib
import hmac
import time
import json
import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


@dataclass
class APIKey:
    key_id: str
    key_hash: str
    owner: str
    tier: str
    created_at: float
    expires_at: Optional[float]
    permissions: list
    is_active: bool = True
    request_count: int = 0
    last_used: Optional[float] = None


class HQNNKeyManager:
    TIERS = {
        "research": {
            "rate_limit": 1000,
            "permissions": ["grover", "shor", "vqe", "simulation"],
            "max_qubits": 20,
        },
        "enterprise": {
            "rate_limit": 10000,
            "permissions": ["grover", "shor", "vqe", "simulation",
                            "beam_cage", "rl_optimizer", "tensor_network"],
            "max_qubits": 50,
        },
        "internal": {
            "rate_limit": -1,
            "permissions": ["*"],
            "max_qubits": -1,
        },
    }

    def __init__(self, storage_path: str = ".hqnn_keys",
                 master_password: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, mode=0o700)
        self._encryption_key = self._derive_key(
            master_password or os.environ.get("HQNN_MASTER_PASSWORD", "")
        )
        self._fernet = Fernet(self._encryption_key)
        self._keys: Dict[str, APIKey] = self._load_keys()

    def _derive_key(self, password: str) -> bytes:
        salt = b"hqnn_topology_2024_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(
            kdf.derive(password.encode() or secrets.token_bytes(32))
        )

    def _load_keys(self) -> Dict[str, APIKey]:
        keys_file = self.storage_path / "keys.enc"
        if not keys_file.exists():
            return {}
        try:
            encrypted = keys_file.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            raw = json.loads(decrypted)
            return {k: APIKey(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def _save_keys(self) -> None:
        keys_file = self.storage_path / "keys.enc"
        raw = {k: vars(v) for k, v in self._keys.items()}
        encrypted = self._fernet.encrypt(json.dumps(raw).encode())
        keys_file.write_bytes(encrypted)
        keys_file.chmod(0o600)

    def generate_key(self, owner: str, tier: str = "research",
                      expires_days: Optional[int] = 365) -> str:
        if tier not in self.TIERS:
            raise ValueError(f"Invalid tier: {tier}. Choose from {list(self.TIERS)}")

        raw_key = f"hqnn_{tier}_{secrets.token_urlsafe(32)}"
        key_id = f"kid_{secrets.token_hex(8)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        expires_at = None
        if expires_days:
            expires_at = time.time() + expires_days * 86400

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            owner=owner,
            tier=tier,
            created_at=time.time(),
            expires_at=expires_at,
            permissions=self.TIERS[tier]["permissions"],
        )
        self._keys[key_id] = api_key
        self._save_keys()
        return raw_key

    def validate_key(self, raw_key: str,
                      required_permission: Optional[str] = None) -> Dict:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        matched = None
        for api_key in self._keys.values():
            if hmac.compare_digest(api_key.key_hash, key_hash):
                matched = api_key
                break

        if not matched:
            return {"valid": False, "reason": "invalid_key"}
        if not matched.is_active:
            return {"valid": False, "reason": "revoked"}
        if matched.expires_at and time.time() > matched.expires_at:
            return {"valid": False, "reason": "expired"}
        if required_permission:
            perms = matched.permissions
            if "*" not in perms and required_permission not in perms:
                return {"valid": False, "reason": "insufficient_permissions"}

        matched.request_count += 1
        matched.last_used = time.time()
        self._save_keys()

        return {
            "valid": True,
            "key_id": matched.key_id,
            "owner": matched.owner,
            "tier": matched.tier,
            "permissions": matched.permissions,
            "rate_limit": self.TIERS[matched.tier]["rate_limit"],
            "max_qubits": self.TIERS[matched.tier]["max_qubits"],
        }

    def revoke_key(self, key_id: str) -> bool:
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            self._save_keys()
            return True
        return False

    def list_keys(self, owner: Optional[str] = None) -> list:
        keys = list(self._keys.values())
        if owner:
            keys = [k for k in keys if k.owner == owner]
        return [
            {
                "key_id": k.key_id,
                "owner": k.owner,
                "tier": k.tier,
                "is_active": k.is_active,
                "created_at": datetime.fromtimestamp(k.created_at).isoformat(),
                "expires_at": datetime.fromtimestamp(k.expires_at).isoformat()
                              if k.expires_at else None,
                "request_count": k.request_count,
            }
            for k in keys
        ]

    def rotate_key(self, old_raw_key: str, owner: str) -> Optional[str]:
        validation = self.validate_key(old_raw_key)
        if not validation["valid"]:
            return None
        key_id = validation["key_id"]
        old_key = self._keys[key_id]
        new_raw_key = self.generate_key(
            owner=owner,
            tier=old_key.tier,
            expires_days=int((old_key.expires_at - time.time()) / 86400)
            if old_key.expires_at else 365,
        )
        self.revoke_key(key_id)
        return new_raw_key