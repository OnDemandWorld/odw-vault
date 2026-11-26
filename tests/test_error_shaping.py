"""回归：422 校验错误瘦身 + /health TTL 缓存（2026-09-12 评审批次 B）。"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestValidationShrinking:
    def test_oversized_query_error_is_truncated(self):
        # 50k 字符的非法输入不应被完整回显进 422 错误体
        r = client.post("/query", json={"query": "字" * 50000, "top_k_chunks": 3})
        assert r.status_code == 422
        assert len(r.text) < 5000
        assert "截断" in r.text

    def test_normal_validation_error_still_actionable(self):
        r = client.post("/query", json={})
        assert r.status_code == 422
        assert "query" in r.text  # 字段定位信息保留


class TestHealthCache:
    def test_health_cached_within_ttl(self, tmp_path):
        """OCR audit #64: mock the probes (no real Ollama/Chroma//tmp writes)
        and assert the cache by CALL COUNTS — r1.json()==r2.json() held even
        with the cache removed, so the old test could not detect its absence."""
        import sqlite3

        import sqlite_utils

        from pipeline.db import migrate

        conn = sqlite3.connect(str(tmp_path / "t.db"), check_same_thread=False)
        db = sqlite_utils.Database(conn)
        migrate(db)

        mock_cfg = MagicMock()
        mock_cfg.ollama.host = "http://localhost:11434"
        mock_cfg.chroma_root_path = str(tmp_path / "chroma")
        mock_cfg.models.embedding.collection_suffix = "test"
        mock_cfg.models.language_id.model_path = str(tmp_path / "no-model")

        with (
            patch("api.main._load_config", return_value=mock_cfg) as mock_load_cfg,
            patch("api.main._get_db", return_value=db),
            patch("api.main.ollama.Client") as mock_ollama,
            patch("api.main.chromadb.PersistentClient") as mock_chroma,
        ):
            mock_chroma.return_value.get_collection.return_value = MagicMock()
            r1 = client.get("/health")
            r2 = client.get("/health")

        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json() == r2.json()
        # Cache hit short-circuits before _load_config and the probes.
        assert mock_load_cfg.call_count == 1
        assert mock_ollama.call_count == 1
        assert mock_chroma.call_count == 1
        conn.close()
