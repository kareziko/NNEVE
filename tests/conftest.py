from pathlib import Path
from typing import Any, List

import pytest

from .marker_flag_pair import MarkerFlagPairBase, MarkerFlagPairMeta


class OfflineOnlinePair(MarkerFlagPairBase):
    flag_name: str = "offline"
    flag_doc: str = "Run only offline tests"
    mark: str = "online"
    mark_doc: str = (
        "mark test as requiring network & database connection to run"
    )
    mark_reason: str = "Online tests were explicitly disabled."


def pytest_addoption(parser: Any):
    MarkerFlagPairMeta.addoptions(parser)


def pytest_configure(config: Any):
    MarkerFlagPairMeta.addinivalue_line(config)


def pytest_collection_modifyitems(config: Any, items: List[Any]):
    MarkerFlagPairMeta.collection_modifyitems(config, items)


@pytest.fixture(scope="session")
def test_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="session")
def repo_dir() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def source_dir(repo_dir: Path) -> Path:
    return repo_dir / "source"


@pytest.fixture(scope="session")
def package_dir(source_dir: Path) -> Path:
    return source_dir / "nneve"
