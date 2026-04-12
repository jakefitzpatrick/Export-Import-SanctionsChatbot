# Logging Integration Changes

This guide explains how we added structured logging to the chatbot, mixing clear explanations with the technical steps.

## 1. Dependency Update
- Added **python-json-logger>=2.0.0** to `requirements.txt` so our logs can be output in JSON format when needed.

## 2. Central Logger Module (`logger.py`)
- Moved all logger setup into `logger.py`. It reads these environment variables:
  - `LOG_LEVEL` (e.g. INFO, DEBUG)
  - `LOG_FORMAT` (`json` or `text`)
  - `LOG_FILE` (optional file path)
- Automatically creates:
  - A console handler that writes structured logs to stdout.
  - A file handler if `LOG_FILE` is set.

## 3. Adding Logs to Key Modules
- **app.py**: Logs app startup and RAG index initialization.
- **rag.py**: Logs when creating document sources and picking the latest HTS CSV file.
- **risk_model.py**: Swapped summary and error `print()` calls for `info()`, `debug()`, and `exception()` logs.

## 4. Console Output Fix
- Updated the console handler to explicitly use `sys.stdout`. This ensures JSON logs appear correctly in Streamlit’s terminal and in containers.

## 5. Tests for Logging (`logging_tests.md`)
- Provided pytest examples showing how to:
  - Confirm JSON vs text log formats.
  - Check log levels and message fields.
  - Verify file handler output.

---
_Together, these updates ensure the chatbot emits consistent, readable, and machine‑friendly logs for local development, automated testing, and production monitoring._
