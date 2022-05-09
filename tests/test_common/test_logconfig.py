import logging
from typing import Any
from unittest.mock import Mock


def test_logconfig_debug(caplog: Any):
    import nneve.common.logconfig as logconfig

    logconfig.configure_logger(True, False)
    logging.warning("Test warning")
    logging.debug("Test debug")
    logging.info("Test info")
    assert "warning" in caplog.text
    assert "debug" in caplog.text
    assert "info" in caplog.text


def test_logconfig_verbose(caplog: Any):
    import nneve.common.logconfig as logconfig

    logconfig.configure_logger(False, True)
    logging.warning("Test warning")
    logging.debug("Test debug")
    logging.info("Test info")
    assert "warning" in caplog.text
    assert "debug" not in caplog.text
    assert "info" in caplog.text


def test_logconfig_warning(caplog: Any):
    import nneve.common.logconfig as logconfig

    logconfig.configure_logger(False, False)
    logging.warning("Test warning")
    logging.debug("Test debug")
    logging.info("Test info")
    assert "warning" in caplog.text
    assert "debug" not in caplog.text
    assert "info" not in caplog.text


def test_logconfig_disable_warnings():
    urllib3: Any = Mock()
    import sys

    sys.modules["urllib3"] = urllib3
    import nneve.common.logconfig as logconfig

    logconfig.configure_logger(False, False)
    urllib3.disable_warnings.assert_called()
