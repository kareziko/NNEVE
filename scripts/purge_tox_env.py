#!/usr/bin/python3
import shutil
from pathlib import Path

import click

TOX_LOCATIONS = (
    Path(".") / ".tox",
    Path(__file__).parent / ".." / ".tox",
)


@click.command()
@click.argument("env_name", required=False, default=None)
@click.option("--purge-all", is_flag=True, default=False)
def purge_tox_env_cli(*args, **kwargs):
    return purge_tox_env(*args, **kwargs)


def purge_tox_env(env_name: str, purge_all: bool = False):
    if env_name is None and purge_all is False:
        print(
            "Neither environment name nor --all flag was given: no purge done."
        )
        return 0

    if purge_all is False:
        return _purge_env(env_name)
    else:
        return _purge_all()


def _purge_env(env_name: str):
    for possible_path in TOX_LOCATIONS:
        env_path: Path = possible_path / env_name
        if env_path.is_dir():
            shutil.rmtree(str(env_path))
            print(f"Removed {env_path}")
            break
    else:
        print("Tox directory not found.")
        return -999


def _purge_all():
    for possible_path in TOX_LOCATIONS:
        if possible_path.is_dir():
            shutil.rmtree(str(possible_path))
            print(f"Removed {possible_path}")
            break
    else:
        print("Tox directory not found.")
        return -999
    return 0


if __name__ == "__main__":
    exit(purge_tox_env_cli())
