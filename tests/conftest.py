# SPDX-License-Identifier: Apache-2.0
import pytest

from app.config import Settings


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        llama_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=8000,
    )
