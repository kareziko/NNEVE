"""CLI entry point."""
import sys

from .cli import auto_load_commands_from_cli_folder, cli

auto_load_commands_from_cli_folder()
cli(sys.argv[1:])
