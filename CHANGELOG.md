# Changelog

All notable changes to ODW.ai Vault will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> 版本对照：项目 semver（本文件）与套件里程碑（V1.1-V1.6，见仓库根 `roadmap/`）并存。
> 套件 V1.4-V1.6 的 Vault 功能（审计导出、分布式追踪、流式引用修复、多 provider 设置）
> 落在项目 0.2.x 内，见下方对应条目。

## [Unreleased] - 2026-10-22（第六轮深度审查）

### Fixed
- **[P1] `/files?status=failed` 永远匹配不到**：过滤用的失败相位标签是 `'extraction'`，
  而管线写入的是 `'extract'`（phase8/indexer/embed）——真实失败的文件永远显示为 pending。
  连同把旧标签锁进测试的 fixture 一并修正（tests/test_review_round6_fixes.py）。
- **[P1] UI 点赞/点踩静默失败**：`UPDATE … ORDER BY … LIMIT` 不是合法 SQLite 语法；
  改为 UI 轮次记录 query_log 行后按主键精确定位更新。
- **[P1] 引用页码几乎恒 null**：两处 embed 均未把 `start_page` 写入 Chroma metadata，
  RRF 融合也不回填 BM25 命中的页码。三路修复：embed 写入 + 融合回填 + chunk 表兜底。
- **[P1] `use_augmentation` 死参数**：schema/CLI 接受并透传但 retrieve() 从不读取；
  现按配置默认（contextual_retrieval.enabled）或请求覆盖，启用时把 phase-10.5 情境
  前缀进生成上下文。
- **[P2] `/query/stream` 不拒答空上下文**：与非流式路径分叉；现按
  refuse_on_empty_context 直接返回拒答（refused=true）。
- **[P2] 生成端点故障裸 500**：预检只 ping embedding host；现同时预检
  models.generation.endpoint，重试耗尽映射为 503。
- **[P2] `/queries` end_date 排除当天**：裸日期字符串比较边界错误；归一化为当天
  23:59:59（含 ISO `T` 分隔兼容）。
- **[P2] 上传 mtime 格式不一致**：isoformat（微秒+时区）vs indexer 的
  `%Y-%m-%dT%H:%M:%SZ`——sync 的 mtime+size 快速跳过对上传文件永不命中，每次重算
  SHA-256。统一为 indexer 格式。
- **[P2] `vault models check` 误报**：子串匹配（qwen3 命中 qwen3-embedding）且每模型
  跑一次 ollama list；改为解析一次精确匹配。
- **[P2] UI 多轮对话名存实亡**：_stream_tokens 只发当前轮；现注入与 API /query 相同的
  历史窗口。UI 查询现也落 query_log（/queries 历史与指标不再对 UI 流量全盲）。
- **[P2] `/eval/run` 同步阻塞**：分钟级评估在请求线程执行；改后台线程 + run_id +
  `GET /eval/run/{id}` 轮询；501 文案同步修正。
- **[P2] 文档**：README/CLAUDE 中 `vault serve` 的 Datasette 描述纠正（serve 启动
  FastAPI；Datasette 是 serve-datasette）。

### Deferred
- require_citations 生成期强制（V-14）列 V1.7（当前仅提示词约束 + 设置页展示）。

## [Unreleased] - 2026-10-08（上线前深度审查轮）

### Fixed
- **[P0] 文件监听（watcher）跨线程使用 SQLite 连接**：API/CLI 把事件循环线程的连接传给
  监听 worker 线程，每个文件变更事件都抛 `sqlite3.ProgrammingError` 被吞成 failed——
  watcher 开启后一个文件也索不进去且完全静默。改为 worker 线程经 `db_factory` 自建连接；
  `startup_sync` 全量同步也移出事件循环线程（此前直接阻塞服务启动分钟级）。
- **[P1] summary 向量删除错位/泄漏**：`_remove_chroma_vectors` 对 `summaries__` 集合用
  `s_{chunk_id}` 构造 ID，而真实主键是 `s_{summary_id}`（两条自增序列）——通常删不到
  任何东西（孤儿向量永久累积），序列撞号时误删其他文件的 summary 向量（数据损坏）。
  改为按 `summary_embedding_ref` 记录的真实 external_id 删除；`DELETE /files/{id}`
  同步补上 summary 向量、`summary_embedding_ref` 行与中文 BM25 索引清理。
- **[P1] 检索候选集 `$in` 过滤未分批**：Chroma 对单次 `$in` 有规模上限（嵌入管线自身按
  1000 分批），检索侧一次性塞入全部候选时大目录下 dense 召回静默归零。dense 与英文
  BM25 路径均改为分批查询后按分数/rank 合并取 top-n（BM25 同时规避 SQLite 绑定参数
  上限）；中文 BM25 失败日志 debug→warning。
- **[P1] fastText 语言模型每查询从磁盘重载**（~130MB）：改用 `rag.tokenization` 的按
  路径缓存加载器；`/health` 探测同改。
- **[P1] 上传阻塞事件循环**：分块写盘移入线程池（`run_in_threadpool`），大文件上传
  期间不再停摆所有并发请求/流式 token 推送。
- **[P2] CLI `query` 默认强行启用 reranker**：`use_reranker=not no_rerank` 恒为 True，
  覆盖配置且在未配置真实 reranker 时退化为逐文档 LLM 打分（一次查询多出 50 次调用）。
  改为 `None` 时回落到 `cfg.models.reranker.enabled`。
- **[P2] 上传同名文件 TOCTOU**：`exists()` 检查后写入可被并发请求双双通过并互相覆盖。
  改为独占创建 `open("xb")` + `FileExistsError` 原子换名重试。
- **[P2] 上传路径未 resolve 与索引器不一致**：符号链接语料根下产生重复根 folder、
  `sync_all` 把全部上传文件误判为已删除。上传侧统一存 resolved 路径。

### Changed
- 测试规模 694 → 703（新增 `tests/test_review_round4_fixes.py` 9 条回归：
  watcher 线程亲和 ×2、summary 向量删除 ×2、dense/BM25 分批 ×4、CLI reranker 默认 ×1）。

## [Unreleased] - 2026-09-12（全流程测试与验收轮）

### Changed
- `/health` 探测 TTL 5s → 15s：并发 LLM 负载下冷重探测实测 1.0–2.0s（100 画像 UAT 7 次超 SLO），
  15s 覆盖推理突发（/health 是状态探针而非实时仪表）。
## [Unreleased] - 2026-09-12

### Changed
- **422 校验错误瘦身**（评审批次 B）：`RequestValidationError` 的 `input` 回显截断
  （超长字符串保留前 200 字符 + 原长标注），此前 50KB 的 query 会把错误体放大到 >50KB。
- **`/health` 组件探测 5s TTL 缓存**：Ollama list / Chroma 打开集合 / fasttext 加载每次
  探测代价高，LLM 并发负载下 /health p95 曾超 1s。
- **`VAULT_OLLAMA_HOST` / `VAULT_GENERATION_HOST` 环境变量覆盖**：容器/套件部署可将
  Ollama 指向套件内服务，无需改烤入镜像的 config.toml（generation 端点默认云端的
  `ollama.com` 会被同源覆盖）。
- UI 微字号可读性：8.5px→10px、9px→10.5px（等宽大写标签）。

### Fixed（2026-09-11 深度代码审查，详见 docs/CODE_REVIEW_FIXES_2026-09-11.md）
- 增量索引器提取器调用约定不匹配（同步全量失败）、`/files/upload` 路径穿越、
  流式查询跨线程 SQLite 连接、流式生成阻塞事件循环、文件夹/工作区过滤静默失效、
  知识库状态面板多处缺陷、XSS 加固系列、去重幂等性等 24 项。

## [0.2.0] - 2026-08-02

### Added（对应套件 V1.1-V1.6 的 Vault 部分）
- 多工作区知识库隔离（V1.1）、多分块策略 + 中文检索（V1.2）、合规审计日志（V1.3）、
  审计报告导出 CSV/JSON（V1.4）、分布式追踪 `X-Trace-Id`（V1.5）、追踪 span + 采样 +
  Console/OTLP 导出（V1.6）。
- 测试规模：676 pytest（覆盖率 66.6%）。
