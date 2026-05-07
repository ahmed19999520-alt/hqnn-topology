import argparse
import json
import os
from hqnn.auth.api_key_manager import HQNNKeyManager
from hqnn.auth.license_manager import HQNNLicenseManager


def parse_args():
    parser = argparse.ArgumentParser(description="HQNN Admin Key Management")
    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser("generate", help="Generate new API key")
    gen.add_argument("--owner", required=True)
    gen.add_argument("--tier", default="research",
                     choices=["research", "enterprise", "internal"])
    gen.add_argument("--expires_days", type=int, default=365)

    val = subparsers.add_parser("validate", help="Validate API key")
    val.add_argument("--key", required=True)

    rev = subparsers.add_parser("revoke", help="Revoke API key")
    rev.add_argument("--key_id", required=True)

    lst = subparsers.add_parser("list", help="List all keys")
    lst.add_argument("--owner", default=None)

    lic = subparsers.add_parser("license", help="Issue license")
    lic.add_argument("--org", required=True)
    lic.add_argument("--tier", default="research",
                     choices=["community", "research", "enterprise"])
    lic.add_argument("--days", type=int, default=365)

    return parser.parse_args()


def main():
    args = parse_args()
    master_password = os.environ.get("HQNN_MASTER_PASSWORD", "")
    manager = HQNNKeyManager(master_password=master_password)
    license_manager = HQNNLicenseManager()

    if args.command == "generate":
        key = manager.generate_key(args.owner, args.tier, args.expires_days)
        print(f"\nGenerated API Key:")
        print(f"  Owner : {args.owner}")
        print(f"  Tier  : {args.tier}")
        print(f"  Expires: {args.expires_days} days")
        print(f"\n  KEY   : {key}")
        print(f"\n  Store this key securely. It will not be shown again.")

    elif args.command == "validate":
        result = manager.validate_key(args.key)
        print(json.dumps(result, indent=2))

    elif args.command == "revoke":
        success = manager.revoke_key(args.key_id)
        print(f"Revoked: {success}")

    elif args.command == "list":
        keys = manager.list_keys(args.owner)
        print(json.dumps(keys, indent=2))

    elif args.command == "license":
        license_data = license_manager.issue_license(args.org, args.tier, args.days)
        output_path = f"license_{args.org.lower().replace(' ', '_')}.json"
        with open(output_path, "w") as f:
            json.dump(license_data, f, indent=2)
        print(f"License issued: {output_path}")
        print(f"  ID     : {license_data['license_id']}")
        print(f"  Org    : {license_data['organization']}")
        print(f"  Tier   : {license_data['tier']}")


if __name__ == "__main__":
    main()