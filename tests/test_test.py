from pathlib import Path

import pytest

from tests.marker_flag_pair import MarkerFlagPairBase


def test_dir_fixtures(
    test_dir: Path, repo_dir: Path, source_dir: Path, package_dir: Path
):
    assert test_dir == Path(__file__).parent
    assert repo_dir == test_dir.parent
    assert source_dir == repo_dir / "source"
    assert package_dir == source_dir / "nneve"


def test_fail_to_instantiate_pair_base():
    with pytest.raises(RuntimeError):
        MarkerFlagPairBase()

    class Subclass(MarkerFlagPairBase):
        flag_name: str = "xdd"
        flag_doc: str = "xdd"
        mark: str = "xdd"
        mark_doc: str = "xdd"
        mark_reason: str = "xdd"

    with pytest.raises(RuntimeError):
        Subclass()
