# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_config_defaults():
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.model_name == "qwen3.5-4b"
    assert s.context_tokens == 16384
    assert s.enable_thinking is False
    assert s.parsed_search_providers == ["ddgs", "searxng", "brave"]


def test_config_rejects_non_loopback():
    with pytest.raises(ValueError, match="Knight is strictly local-first/loopback only"):
        Settings(host="0.0.0.0")


def test_config_rejects_ipv6_wildcard_and_non_loopback():
    """C-25: `::` (and any non-loopback address) must fail startup."""
    for host in ("::", "[::]", "2001:db8::1", "192.168.1.10", "example.com"):
        with pytest.raises(ValueError, match="loopback"):
            Settings(host=host)

    assert Settings(host="::1").host == "::1"
    assert Settings(host="localhost").host == "localhost"


def test_settings_construction_does_not_mutate_tracing_env(monkeypatch):
    """C-23: an operator's explicit setting is never silently overwritten."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    Settings()
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_startup_refuses_enabled_tracing(monkeypatch):
    """C-23: tracing is refused explicitly instead of being force-disabled."""
    from app.main import assert_tracing_disabled

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(RuntimeError, match="LANGSMITH_TRACING"):
        assert_tracing_disabled()

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert_tracing_disabled()


def test_importing_app_main_has_no_side_effects(tmp_path):
    """C-24: importing the module must not construct the FastAPI app."""
    code = "import app.main as m; print('app' in vars(m))"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "KNIGHT_DATA_DIR": str(tmp_path / "data"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_create_app_is_the_only_constructor():
    from app.main import create_app

    application = create_app(Settings(auth_token=""))
    assert application.title == "Knight"


def test_config_paths(tmp_path):
    s = Settings(data_dir=tmp_path / "custom_data")
    assert s.app_db_path == tmp_path / "custom_data" / "knight.sqlite"
    assert s.checkpoint_db_path == tmp_path / "custom_data" / "checkpoints.sqlite"
    assert s.logs_dir == tmp_path / "custom_data" / "logs"

    s.ensure_dirs()
    assert s.logs_dir.is_dir()
    assert s.models_dir.is_dir()
    assert s.cache_dir.is_dir()


def test_log_prompts_setting():
    """L-02: log_prompts defaults to False and can be enabled via configuration."""
    s_default = Settings()
    assert s_default.log_prompts is False

    s_enabled = Settings(log_prompts=True)
    assert s_enabled.log_prompts is True
