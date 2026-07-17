"""Set up all tutorial dependencies and external repositories in one command."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command, dry_run=False):
    printable = " ".join(map(str, command))
    print(f"> {printable}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print setup commands without changing the environment.",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Do not upgrade pip before installing requirements.",
    )
    args = parser.parse_args()

    run(["git", "submodule", "update", "--init", "--recursive"], args.dry_run)
    run([sys.executable, "scripts/apply_external_patches.py"], args.dry_run)

    if not args.skip_pip_upgrade:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)

    run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-all.txt"],
        args.dry_run,
    )

    if not args.dry_run:
        print("\nSetup complete. Start Jupyter from the repository root.")
        print("Note: this installs CPU JAX. Install a CUDA-specific JAX build separately for GPU acceleration.")


if __name__ == "__main__":
    main()
