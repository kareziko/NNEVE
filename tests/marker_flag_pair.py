from typing import Any, Dict, List, Tuple, Type

import pytest


class MarkerFlagPairMeta(type):

    subclasses: List[Type["MarkerFlagPairBase"]] = []

    def __new__(
        cls: Any,
        name: str,
        bases: Tuple[type, ...],
        attrs: Dict[str, Any],
    ):
        subclass: Type["MarkerFlagPairBase"] = super(
            MarkerFlagPairMeta, cls
        ).__new__(cls, name, bases, attrs)
        if name != "MarkerFlagPairBase":
            cls.subclasses.append(subclass)
        return subclass

    @classmethod
    def addoptions(cls, parser: Any):
        for subclass in cls.subclasses:
            parser.addoption(
                subclass.get_flag(),
                action="store_true",
                default=False,
                help=subclass.flag_doc,
            )

    @classmethod
    def addinivalue_line(cls, config: Any):
        for subclass in cls.subclasses:
            config.addinivalue_line(
                "markers",
                (f"{subclass.mark}: {subclass.mark_doc}"),
            )

    @classmethod
    def collection_modifyitems(
        cls, config: Any, items: List[Any]
    ):  # pragma: no cover
        for subclass in cls.subclasses:
            if config.getoption(subclass.get_flag()):
                cls._skip_item(items, subclass)

    @classmethod
    def _skip_item(
        cls, items: List[Any], subclass: Type["MarkerFlagPairBase"]
    ):  # pragma: no cover
        skip = pytest.mark.skip(reason=subclass.mark_reason)
        for item in items:
            if subclass.mark in item.keywords:
                item.add_marker(skip)


class MarkerFlagPairBase(metaclass=MarkerFlagPairMeta):
    flag_name: str
    flag_doc: str
    mark: str
    mark_doc: str
    mark_reason: str

    def __new__(cls):
        raise RuntimeError(
            f"{cls.__qualname__} is not meant to be instantiated."
        )

    @classmethod
    def get_flag(cls):
        return f"--{cls.flag_name}"
