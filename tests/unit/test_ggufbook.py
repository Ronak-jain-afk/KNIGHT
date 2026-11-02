# SPDX-License-Identifier: Apache-2.0
import pytest

from app.llm.ggufbook import evaluate_hardware_fit, extract_quant_type


def test_extract_quant_type():
    assert extract_quant_type("Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert extract_quant_type("Llama-3.2-3B.Q8_0.gguf") == "Q8_0"
    assert extract_quant_type("DeepSeek-R1-IQ3_M.gguf") == "IQ3_M"
    assert extract_quant_type("model_bf16.gguf") == "BF16"
    assert extract_quant_type("plain_model.gguf") == "GGUF"


def test_evaluate_hardware_fit_optimal():
    # 2 GB model on 6 GB VRAM, 16 GB RAM -> optimal
    gpu = {"available": True, "vram_total_mb": 6141}
    ram = {"available": True, "total_mb": 16000}
    res = evaluate_hardware_fit(2 * 1024 * 1024 * 1024, gpu, ram)
    assert res["status"] == "optimal"
    assert res["can_fit_gpu"] is True
    assert res["warning"] is None


def test_evaluate_hardware_fit_tight():
    # 5.5 GB model on 6.1 GB VRAM -> fits in VRAM, but less than 1.2 GB context headroom
    gpu = {"available": True, "vram_total_mb": 6141}
    ram = {"available": True, "total_mb": 16000}
    res = evaluate_hardware_fit(int(5.5 * 1024 * 1024 * 1024), gpu, ram)
    assert res["status"] == "tight"
    assert res["can_fit_gpu"] is True
    assert "large context" in res["warning"]


def test_evaluate_hardware_fit_partial_cpu():
    # 8.5 GB model on 6.1 GB VRAM, 16 GB RAM -> exceeds VRAM, fits in RAM
    gpu = {"available": True, "vram_total_mb": 6141}
    ram = {"available": True, "total_mb": 16000}
    res = evaluate_hardware_fit(int(8.5 * 1024 * 1024 * 1024), gpu, ram)
    assert res["status"] == "partial_cpu"
    assert res["can_fit_gpu"] is False
    assert "exceeds GPU VRAM" in res["warning"]


def test_evaluate_hardware_fit_oom_danger():
    # 24 GB model on 6.1 GB VRAM, 16 GB RAM -> exceeds both VRAM and RAM
    gpu = {"available": True, "vram_total_mb": 6141}
    ram = {"available": True, "total_mb": 16000}
    res = evaluate_hardware_fit(24 * 1024 * 1024 * 1024, gpu, ram)
    assert res["status"] == "oom_danger"
    assert res["can_fit_gpu"] is False
    assert "Out-Of-Memory" in res["warning"]
    assert res["label"] == res["badge"] == "Insufficient RAM"
    assert res["explanation"] == res["detail"]


def test_download_task_to_dict_aligned_keys():
    """U-01, U-02, U-03, U-04: Task serialization provides both API & UI key sets."""
    from app.llm.ggufbook import DownloadTask

    task = DownloadTask(
        task_id="task-123",
        repo_id="test/repo",
        filename="model.gguf",
        url="https://huggingface.co/test/resolve/main/model.gguf",
        size_bytes=100_000_000,
    )
    task.downloaded_bytes = 25_000_000
    task.speed_mbps = 12.34

    d = task.to_dict()
    # U-01: task_id aligns with id
    assert d["task_id"] == d["id"] == "task-123"
    # U-02: status enum
    assert d["status"] in ("queued", "downloading", "completed", "failed", "cancelled")
    # U-03: size / progress alignment
    assert d["total_bytes"] == d["size_bytes"] == 100_000_000
    assert d["progress_pct"] == d["percentage"] == 25.0
    # U-04: speed alignment
    assert d["speed_mb_s"] == d["speed_mbps"] == 12.34


def test_curated_models_have_no_fabricated_metrics():
    """D-01: Curated models must not serialize fabricated download or like counts."""
    from app.llm.ggufbook import CURATED_TRENDING_MODELS

    assert len(CURATED_TRENDING_MODELS) >= 4
    for m in CURATED_TRENDING_MODELS:
        assert m.get("is_curated") is True
        assert "downloads" not in m
        assert "likes" not in m


def test_evaluate_hardware_fit_unknown_size():
    """D-02: Missing size must report status='unknown' instead of guessing 4 GB."""
    gpu = {"available": True, "vram_total_mb": 6141}
    ram = {"available": True, "total_mb": 16000}
    res = evaluate_hardware_fit(None, gpu, ram)
    assert res["status"] == "unknown"
    assert res["can_fit_gpu"] is None
    assert res["label"] == res["badge"] == "Unknown Size"
    assert "not reported" in res["warning"]


def test_dead_code_functions_removed():
    """D-03, D-04, D-05, D-06, D-07: Verify deprecated/dead functions are deleted."""
    import app.db.memories as db_memories
    import app.db.runs as db_runs
    import app.llm.client as llm_client
    import app.memory.embeddings as mem_embeddings

    assert not hasattr(llm_client, "tokenize_text")  # D-03
    assert not hasattr(mem_embeddings.EmbeddingService, "embed_batch")  # D-04
    assert not hasattr(db_memories, "get_memory_by_hash")  # D-05
    assert not hasattr(db_memories, "record_memory_usage")  # D-06
    assert not hasattr(db_runs, "list_runs")  # D-07


@pytest.mark.asyncio
async def test_download_manager_bounds_task_registry(tmp_path, monkeypatch):
    """KN-11: task registry evicts older terminal tasks when size reaches limit."""
    from app.config import Settings
    from app.llm.ggufbook import DownloadManager, DownloadTask

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()

    # Pre-populate registry with 50 completed tasks
    DownloadManager._tasks.clear()
    for i in range(50):
        t = DownloadTask(f"t_{i}", "repo/model", f"model_{i}.gguf", "https://huggingface.co/m.gguf", 100)
        t.status = "completed"
        t.created_at = f"2026-10-31T00:00:{i:02d}Z"
        DownloadManager._tasks[t.id] = t

    monkeypatch.setattr("app.llm.ggufbook._sync_download_worker", lambda task, s: None)
    monkeypatch.setattr("app.llm.ggufbook.validate_download_url", lambda url, s: url)

    # Starting a 51st download should evict the oldest completed task (t_0)
    new_task = await DownloadManager.start_download(
        "repo/model", "new_model.gguf", "https://huggingface.co/new.gguf", 100, settings
    )
    assert len(DownloadManager._tasks) <= 50
    assert "t_0" not in DownloadManager._tasks
    assert new_task.id in DownloadManager._tasks

