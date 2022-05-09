from nneve.cli import auto_load_commands_from_cli_folder


def test_auto_cli_discovery():
    auto_load_commands_from_cli_folder()
