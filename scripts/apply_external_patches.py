"""Apply the tutorial's compatibility patches to initialized Git submodules."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PATCHES = (
    (
        ROOT / "external" / "computation-thru-dynamics",
        ROOT / "patches" / "computation-thru-dynamics-modern-jax.patch",
    ),
    (
        ROOT / "external" / "gpslds",
        ROOT / "patches" / "gpslds-numpy2-compat.patch",
    ),
)


def git(repository, *arguments):
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *map(str, arguments),
        ],
        capture_output=True,
        text=True,
    )


def apply_patch(repository, patch):
    if not repository.is_dir():
        raise FileNotFoundError(
            f"Missing submodule {repository}. Run "
            "'git submodule update --init --recursive' first."
        )

    if git(repository, "apply", "--check", patch).returncode == 0:
        result = git(repository, "apply", patch)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        print(f"Applied {patch.name}")
        return

    if git(repository, "apply", "--reverse", "--check", patch).returncode == 0:
        print(f"Already applied: {patch.name}")
        return

    raise RuntimeError(
        f"Cannot apply {patch.name}; the submodule has unexpected changes."
    )


def main():
    for repository, patch in PATCHES:
        apply_patch(repository, patch)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
