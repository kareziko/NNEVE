"""CLI wrapper package."""


import importlib.util
from pathlib import Path
from typing import Any, List

import click

from nneve import __version__
from nneve.common.logconfig import configure_logger

DIR = Path(__file__).parent


def cli(args: List[str]) -> Any:
    """NNEVE CLI interface API endpoint."""
    return nneve(args)


@click.group()
@click.version_option(
    __version__,
    "--version",
    "-V",
    package_name="NNEVE",
)
@click.option(
    "-d",
    "--debug",
    default=False,
    is_flag=True,
    help="Enable debug mode, implies verbose logging.",
)
@click.option(
    "-v",
    "--verbose",
    default=False,
    is_flag=True,
    help="Enable verbose logging, do not implies debug mode.",
)
def nneve(debug: bool, verbose: bool) -> None:
    """nneve entry point description."""
    configure_logger(debug, verbose)


def auto_load_commands_from_cli_folder() -> None:
    # automatically add all commands defined in CLI dir
    for file in DIR.glob("*.py"):
        if not file.name.startswith("_"):
            module_name = file.name.lstrip().rstrip(".py")
            module_spec = importlib.util.spec_from_file_location(
                module_name, str(file)
            )
            assert module_spec is not None
            module = importlib.util.module_from_spec(module_spec)
            assert module_spec.loader is not None
            module_spec.loader.exec_module(module)

            nneve.add_command(getattr(module, module_name))
