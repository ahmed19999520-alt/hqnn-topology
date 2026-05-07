import boto3
import json
import os
from typing import Optional
from botocore.exceptions import ClientError


class AWSSecretBackend:
    def __init__(self, region: Optional[str] = None):
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.client = boto3.client(
            "secretsmanager",
            region_name=self.region,
        )
        self.prefix = "hqnn/"

    def store_secret(self, name: str, data: dict,
                      description: str = "") -> None:
        secret_name = f"{self.prefix}{name}"
        try:
            self.client.create_secret(
                Name=secret_name,
                Description=description,
                SecretString=json.dumps(data),
                Tags=[
                    {"Key": "Project", "Value": "HQNN-Topology"},
                    {"Key": "ManagedBy", "Value": "hqnn-auth"},
                ],
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                self.client.put_secret_value(
                    SecretId=secret_name,
                    SecretString=json.dumps(data),
                )
            else:
                raise

    def get_secret(self, name: str) -> Optional[dict]:
        try:
            response = self.client.get_secret_value(
                SecretId=f"{self.prefix}{name}"
            )
            return json.loads(response["SecretString"])
        except ClientError:
            return None

    def delete_secret(self, name: str,
                       recovery_days: int = 7) -> None:
        self.client.delete_secret(
            SecretId=f"{self.prefix}{name}",
            RecoveryWindowInDays=recovery_days,
        )

    def rotate_secret(self, name: str, new_data: dict) -> None:
        self.client.put_secret_value(
            SecretId=f"{self.prefix}{name}",
            SecretString=json.dumps(new_data),
        )

    def list_secrets(self) -> list:
        paginator = self.client.get_paginator("list_secrets")
        secrets = []
        for page in paginator.paginate(
            Filters=[{"Key": "name", "Values": [self.prefix]}]
        ):
            secrets.extend(page["SecretList"])
        return [s["Name"].replace(self.prefix, "") for s in secrets]