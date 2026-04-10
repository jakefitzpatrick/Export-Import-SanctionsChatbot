# Logging Test Instructions

This guide explains how to write and run unit tests to validate that structured logs emit correctly in both JSON and plain‑text formats.

## 1. Prerequisites
- Ensure `pytest` is installed:  
  ```bash
  pip install pytest
  ```
- Confirm `python-json-logger` is in your environment (already in `requirements.txt`).

## 2. Create the test file
Add a new test module at `tests/test_logging.py` with example tests:

```python
import os
import json
import pytest

from logger import setup_logger


def test_json_log_output(caplog):
    # Force JSON format and no file output
    os.environ["LOG_FORMAT"] = "json"
    os.environ.pop("LOG_FILE", None)
    caplog.set_level("INFO")
    logger = setup_logger("test_json")
    logger.info("hello", extra={"foo": "bar"})

    output = caplog.text.strip()
    data = json.loads(output)
    assert data["message"] == "hello"
    assert data["levelname"] == "INFO"
    assert data["foo"] == "bar"


def test_text_log_format(caplog):
    os.environ["LOG_FORMAT"] = "text"
    caplog.set_level("DEBUG")
    logger = setup_logger("test_text")
    logger.debug("debug msg")

    output = caplog.text.strip()
    assert "DEBUG: debug msg" in output


def test_file_handler(tmp_path):
    # Verify file handler writes to LOG_FILE
    log_file = tmp_path / "app.log"
    os.environ["LOG_FORMAT"] = "json"
    os.environ["LOG_FILE"] = str(log_file)
    logger = setup_logger("test_file")
    logger.warning("warn")

    content = log_file.read_text().strip()
    # JSON line should parse and contain warning
    record = json.loads(content)
    assert record["levelname"] == "WARNING"
    assert record["message"] == "warn"
```

## 3. Run the tests
Execute only the logging tests or your full suite:

```bash
pytest tests/test_logging.py
# or
pytest
```

Passing tests confirm correct log formatting, levels, and file output.
