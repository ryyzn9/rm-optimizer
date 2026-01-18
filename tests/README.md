# RM-Optimizer Tests

This directory contains tests for the RM-Optimizer framework.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rm_optimizer --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py -v

# Run specific test
pytest tests/test_core.py::TestPreferencePair -v
```

## Test Structure

- `test_core.py`: Tests for base classes and data structures
- `test_hessian.py`: Tests for Hessian analysis (requires GPU)
- `test_training.py`: Tests for training module

## Fixtures

Common fixtures are defined in `conftest.py`.
