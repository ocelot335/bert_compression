import os
import subprocess
import sys

subprocess.run(
    ["git", "clone", "https://github.com/ocelot335/bert_compression.git"],
    check=True,
)

subprocess.run(
    ["pip", "install", "-r", "bert_compression/requirements/base.txt", "-q"],
    check=True,
)

sys.path.insert(0, "/kaggle/working/bert_compression")

try:
    from kaggle_secrets import UserSecretsClient

    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
except Exception:
    print("No HF_TOKEN found in secrets, skipping")
