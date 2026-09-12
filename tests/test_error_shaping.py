"""回归：422 校验错误瘦身 + /health TTL 缓存（2026-09-12 评审批次 B）。"""

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
    def test_health_cached_within_ttl(self):
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json() == r2.json()
