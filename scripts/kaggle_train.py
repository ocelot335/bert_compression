import os
import subprocess
import sys


def setup_kaggle_environment() -> None:
    subprocess.run(
        ["git", "clone", "https://github.com/ocelot335/bert_compression.git"],
        check=True,
    )

    subprocess.run(
        [
            "pip",
            "install",
            "-r",
            "bert_compression/kaggle.txt",
            "-q",
        ],
        check=True,
    )

    sys.path.insert(0, "/kaggle/working/bert_compression")


def setup_tokens() -> None:
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()

        hf_token = secrets.get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token

        comet_key = secrets.get_secret("COMET_API_KEY")
        os.environ["COMET_API_KEY"] = comet_key

    except Exception:
        print("No secrets found, skipping")


if __name__ == "__main__":
    setup_kaggle_environment()
    setup_tokens()
