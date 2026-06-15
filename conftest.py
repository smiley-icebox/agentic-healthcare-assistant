"""Shared pytest fixtures — every test runs against an isolated, freshly-seeded DB and
trace log in a temp dir, fully offline (USE_LLM off, so the deterministic answer/plan
paths run with no API key). The LLM synthesis path is covered separately in
test_synthesis.py, which flips USE_LLM on and injects a STUB chat model. The
repository/observability read their paths from module globals, so monkeypatching those
globals redirects all file I/O."""

import pytest

import config
import observability
import repository
import seed_data


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    repository.reset_repository_singleton()
    monkeypatch.setattr(config, "USE_LLM", False)
    monkeypatch.setattr(config, "USE_LIVE_SEARCH", False)   # offline corpus only, deterministic
    monkeypatch.setattr(observability, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(observability, "TRACE_PATH", str(tmp_path / "traces.jsonl"))
    seed_data.seed()
    yield
    repository.reset_repository_singleton()
