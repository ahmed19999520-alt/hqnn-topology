import hashlib
import hmac
import json
import time
import base64
from typing import Dict, Optional
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend


@dataclass
class License:
    license_id: str
    organization: str
    tier: str
    issued_at: float
    expires_at: float
    max_nodes: int
    max_qubits: int
    features: list
    signature: str


class HQNNLicenseManager:
    LICENSE_TIERS = {
        "community": {
            "max_nodes": 9,
            "max_qubits": 10,
            "features": ["grover", "vqe", "simulation"],
            "price": "free",
        },
        "research": {
            "max_nodes": 25,
            "max_qubits": 30,
            "features": ["grover", "shor", "vqe", "simulation",
                          "beam_cage", "noise_prediction"],
            "price": "contact",
        },
        "enterprise": {
            "max_nodes": -1,
            "max_qubits": -1,
            "features": ["*"],
            "price": "contact",
        },
    }

    def __init__(self):
        self._private_key, self._public_key = self._generate_key_pair()

    def _generate_key_pair(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def issue_license(self, organization: str, tier: str,
                       valid_days: int = 365) -> Dict:
        if tier not in self.LICENSE_TIERS:
            raise ValueError(f"Unknown tier: {tier}")

        tier_config = self.LICENSE_TIERS[tier]
        license_id = f"HQNN-{tier.upper()}-{hashlib.sha1(organization.encode()).hexdigest()[:8].upper()}"
        now = time.time()

        payload = {
            "license_id": license_id,
            "organization": organization,
            "tier": tier,
            "issued_at": now,
            "expires_at": now + valid_days * 86400,
            "max_nodes": tier_config["max_nodes"],
            "max_qubits": tier_config["max_qubits"],
            "features": tier_config["features"],
        }

        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = self._private_key.sign(
            payload_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        payload["signature"] = base64.b64encode(signature).decode()
        return payload

    def verify_license(self, license_data: Dict) -> Dict:
        try:
            signature = base64.b64decode(license_data["signature"])
            payload = {k: v for k, v in license_data.items()
                       if k != "signature"}
            payload_bytes = json.dumps(payload, sort_keys=True).encode()

            self._public_key.verify(
                signature,
                payload_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            if time.time() > license_data["expires_at"]:
                return {"valid": False, "reason": "expired"}

            return {
                "valid": True,
                "license_id": license_data["license_id"],
                "organization": license_data["organization"],
                "tier": license_data["tier"],
                "max_nodes": license_data["max_nodes"],
                "max_qubits": license_data["max_qubits"],
                "features": license_data["features"],
                "days_remaining": int(
                    (license_data["expires_at"] - time.time()) / 86400),
            }
        except Exception as e:
            return {"valid": False, "reason": str(e)}

    def check_feature_access(self, license_data: Dict,
                              feature: str) -> bool:
        verification = self.verify_license(license_data)
        if not verification["valid"]:
            return False
        features = verification["features"]
        return "*" in features or feature in features

    def export_public_key(self) -> str:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()