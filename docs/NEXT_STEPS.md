# ODW Vault — 下一步开发计划

> 生成：2026-09-13 ｜ 基线：dev @ `821aca2` + 两轮审查修复（未提交）
> 依据：两轮代码审查与端到端实测结论（见 `docs/CODE_REVIEW_FIXES_2026-09-11.md`）、竞品差距分析（`docs/PRODUCT_ROADMAP.md`，已按当前实现修正）、README 承诺的 v0.3.0 目标。
> 优先级约定：P0 = 公开版发布阻塞项；P1 = v0.3.0 核心卖点；P2 = 企业化扩展；P3 = 工程化/运维。

---

## 0. 现状快照

- **功能**：15 阶段管线、混合检索（Dense+BM25+RRF+层级收窄+可选 Reranker）、强制引用、多轮对话、流式输出、增量同步+失败自愈、多工作区隔离、中英文、审计/追踪/指标、Gradio 聊天 UI（状态面板/会话侧栏/文件夹筛选）。
- **质量**：694 项测试全通过，覆盖率 67.5%；两轮共修复 4 个 P0（含一个会**损坏数据库**的 FTS 双删 bug）与 ~35 个 P1/P2。
- **安全**：服务端默认无鉴权 + CORS `*`；UI 代理 `/api/*` 完全无鉴权——**这是公开版前必须收口的头号问题**（见 §1）。

> 注意：`docs/PRODUCT_ROADMAP.md`（2026-07）的差距矩阵有多处已过时——对话历史管理、文件上传、增量实时索引、多轮记忆、Reranker 等已实现或部分实现，本文档以当前代码为准。

---

## 1. P0 — 安全与稳定性收口（目标：可公开部署，约 2–3 周）

### 1.1 鉴权与网络暴露策略（产品决策 + 实现）
- **现状**：`api/main.py` 支持可选 `VAULT_API_KEY`（默认关），CORS 默认 `*`；`ui/gradio_app.py` 的代理 `/api/*`（含云模型 API Key 读写、文件删除、同步触发）**没有任何鉴权**，且 `--host 0.0.0.0` 时直接暴露；`POST /api/indexing/sync` 可被任意网页 CSRF 触发。
- **任务**：
  1. 定义部署档位：`local`（默认，绑 127.0.0.1，无鉴权）/ `lan`（强制 API Key + 收窄 CORS + HTTPS 引导）/ `public`（Key + 速率限制 + HTTPS，拒绝明文）；`vault serve`/`vault ui` 启动时打印当前档位风险提示。
  2. UI 代理 `/api/*` 接入与 API 相同的 Key 校验（`BaseHTTPMiddleware`，该 import 已存在未用），或至少为写操作（POST/DELETE）加同源 token。
  3. CORS 默认值从 `*` 改为 `http://127.0.0.1:7860,http://localhost:7860`，`VAULT_CORS_ORIGINS=*` 变为显式 opt-in。
  4. UI 设置面板存储的云服务商 API Key 加密落盘（`llm_providers.json` 当前明文；至少 0600 + 主密码可选）。
- **验收**：`lan` 档下未带 Key 的所有请求 401 且带 CORS 头；从非白名单 origin 的浏览器 fetch 被拒；`/api/indexing/sync` 无 token 返回 401。

### 1.2 预检查与云端配置一致性
- **现状**：`/query`、`/query/stream`、`/health` 的可达性预检查只打 `cfg.ollama.host`——纯云生成端点（无本地 Ollama）时接口在门口就 503，而实际生成可用。
- **任务**：预检查按 `generation.endpoint` 探活（`_make_client(cfg).list()`），本地/云双模各一条集成测试。

### 1.3 写入路径的事务与并发
- **任务**：
  1. `DELETE /files/{id}`（`api/main.py`）的磁盘+Chroma+SQLite 删除包进单一事务（当前中途失败把脏状态留给复用该线程的下个请求）；
  2. watcher（`rag/watcher.py`）与 UI/CLI `sync_all` 共用与 `api._sync_lock` 等价的进程内互斥（跨进程用 `BEGIN IMMEDIATE` 或锁文件），避免同文件并发双写；
  3. 上传的 DB 写入移出事件循环（`run_in_threadpool`）。
- **验收**：并发 `sync + 删除同一文件 + 上传` 压测脚本（新增 `tests/test_concurrency_stress.py`）integrity_check 始终 ok。

### 1.4 备份与恢复工具
- **现状**：corpus.db + chroma/ + 语料目录三者需同步快照，无官方手段。
- **任务**：`vault backup create|restore`（SQLite `VACUUM INTO` + chroma 目录打包 + 清单哈希校验）；恢复前自动校验 schema_version。
- **验收**：备份→删库→恢复→全量查询一致的自动化测试。

### 1.5 遗留小项打包清理（1–2 天）
- thread-local DB 连接在 shutdown 时关闭；`_bind_trace_id` 头部日志脱敏已由校验覆盖（补单测）；`/eval/run` 复用 `_sync_lock` 模式加并发闸；`config.py` 对 `corpus_root` 为空/不存在给启动级错误提示；`pipeline_run.phase` CHECK 与 schema 演进一致性（加 migration 允许 `indexer`/`embed` 或改列）。

---

## 2. P1 — 产品能力补齐（v0.3.0 卖点，约 4–6 周，可并行）

### 2.1 文档管理 UI（竞品全有，当前最大体验缺口）
- 文件列表页（搜索/筛选/排序/分页，数据源已有：`GET /files`）、单文件详情（元数据、摘要、分块预览、来源索引状态）、重新索引/删除按钮（API 已有 `DELETE /files/{id}`）、**拖拽上传入口**（API 已有 `/files/upload`，UI 缺——非技术用户目前只能请管理员拷文件）。
- 原文预览：`GET /files/{id}/text` 已存在，做侧栏阅读视图 + 命中片段高亮（引用定位体验升级）。
- **验收**：非技术用户按《USER_GUIDE》可独立完成"上传→同步→检索→预览"闭环。

### 2.2 实时索引体验升级
- **现状**：`CorpusWatcher` 与 `vault watch` 已实现，但默认 `watcher.enabled=false`，且与 UI 状态面板不联动。
- **任务**：`vault ui` 可拉起内嵌 watcher（开关进设置面板）；状态面板显示 watcher 运行态与"最近 5 条索引事件"流水；失败队列页（哪些文件失败、错误分类、一键重试——修复轮已提供 `force` 能力）。
- **验收**：拖入新文件 ≤30s 内可被检索到，无需手动 Sync。

### 2.3 检索质量
- 查询改写/扩展（LLM 生成 2–3 个等价查询合并召回；配置开关，默认关以控延迟）；
- 语义缓存（相同/相近 query 命中缓存答案，SQLite 新表 + 向量阈值；`/query` 与 UI 均受益）；
- Reranker 效果 A/B：现有 eval 框架跑 50 题问题集出报告，给出"默认是否开启"的数据结论（当前默认关）。

### 2.4 多模态输入
- 聊天输入框支持粘贴/拖拽图片：走已有 OCR 提取链（`ocr_extractor`）做"以图搜文"；音频文件上传→whisper 转录（extractor 已有，缺上传接线）。

### 2.5 评估与可观测 UI
- eval 结果仪表板（跑批→按题展示命中率/引用正确率/延迟分布，数据源 `eval/runner.py` + `eval_run` 表）；
- `/metrics` 之外给管理页展示近 7 天查询量、失败率、索引进度趋势。

---

## 3. P2 — 企业化扩展（发布后迭代）

| 项 | 说明 | 依赖 |
|---|---|---|
| 用户与权限 | 当前 `user` 仅日志标签。需要：用户表、API Key per-user、workspace 级 ACL（V1.1 的 workspace 隔离正好是基础）、审计按人归因 | 1.1 |
| 向量库可插拔 | 抽象 `VectorStore` 接口（Chroma 为默认实现），提供 Qdrant/pgvector 实现 | — |
| Webhook/集成 | 索引完成、同步失败、评估完成等事件出站通知（Slack/飞书）；`api/spans.py` 的事件基建可复用 | 2.2 |
| 嵌入式组件 | iframe/widget 形态的只读问答卡片（Loop 集成的自然延伸） | 1.1 |
| SSO/LDAP | 企业目录对接 | 用户与权限 |
| 知识图谱（可选） | 文件夹→文档→实体抽取，Mem.ai 式关联；先用现有 folder/summary 推理链做轻量版 | 2.3 |

---

## 4. P3 — 发布工程（与 README v0.3.0 承诺对齐）

1. **安装包**：PyInstaller/Nuitka 打包 macOS `.dmg`、Linux `.deb/.AppImage`、Windows `.exe/.msi`（README 已承诺，当前用户仍需装 Python+uv+ollama）。捆绑 Ollama 或提供"仅本地模型/仅云"两种发行版。
2. **CI 门禁补强**：`ruff check`（当前源码可操作项已清零，设零容忍）+ `pytest --cov-fail-under=70` 阶梯 + 新增的并发/备份/E2E 冒烟测试进 CI；PR 模板附审查清单。
3. **覆盖率洼地**（本轮实测）：`rag/indexer.py` 修复轮路径、`rag/watcher.py`(73%)、`rag/phase8b_transcribe.py`(31%)、`rag/extractors/*`(6–33%)、`rag/phase11_embed.py`(56%) —— 用假服务/二进制替身补齐关键分支。
4. **可观测性验证**：OTLP 导出链路端到端测试（本轮已修 traceId 格式；仍缺带 collector 的契约测试——可用本地 fake-collector 容器）。
5. **文档**：API 参考由 `openapi.json` 自动生成交互式文档站；部署手册（local/lan/public 三档）；《USER_GUIDE_ZH》进发行包。

---

## 5. 建议时间线（单人力参考）

| 阶段 | 周期 | 内容 |
|---|---|---|
| Sprint 1 | 2 周 | §1 全部（安全收口 + 事务/并发 + 备份工具 + 预检查云模 + 小项清理）→ 打 **v0.3.0-rc1** |
| Sprint 2–3 | 3–4 周 | §2.1 文档管理 UI + §2.2 实时索引体验（两条线并行） |
| Sprint 4 | 2 周 | §2.3 检索质量（改写/缓存/rerank 数据结论）+ §2.5 评估仪表板 |
| Sprint 5+ | 持续 | §4 安装包工程（可与 Sprint 3–4 并行）；§2.4 多模态、§3 企业化按需求拉动 |

**发布判据（v0.3.0 GA）**：§1 全部完成且 CI 绿；非技术用户用指南 + 安装包可在无终端环境下完成"装→放文档→问→看到引用"；并发压测 24h integrity ok；三平台安装包冒烟测试通过。

---

*本计划由 2026-09 两轮审查的数据驱动生成；执行前建议对 §1.1 的部署档位策略做一次产品评审。*
