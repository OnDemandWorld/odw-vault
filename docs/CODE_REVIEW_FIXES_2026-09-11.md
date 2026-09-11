# ODW Vault — 代码审查与修复报告（2026-09-11）

> 范围：`api/`、`rag/`、`pipeline/`、`ui/gradio_app.py`、`cli.py`
> 方法：4 路并行深度审查（含对照安装库源码与真实 `corpus.db` 的实证验证）+ 667 项既有测试回归 + 新增 9 项回归测试 + 本地端到端实测（UI/API/同步链路）。
> 结果：**676 / 676 测试通过**（覆盖率 65%→66.6%）。本文记录已修复项与建议后续处理的遗留项。

---

## 一、已修复（P0 — 严重）

### 1. 增量索引器的提取器调用约定不匹配 → 同步必然全量失败
- **位置**：`rag/indexer.py` `_run_extraction()`
- **问题**：所有提取器返回 `(text, meta, succeeded, err)` 元组（与 `rag/phase8_extract.py` 的用法一致），而索引器把返回值当 dict 调用 `.get("text")`，每次都会 `AttributeError` 并把**每一个文件**记录为提取失败。受影响入口：UI 的 "Sync Now"、`cli.py sync`、`watcher`。
- **修复**：按元组解包，`succeeded=False` 时转入失败路径；tika 提取器补传 `tika_url` / `brute_force` 配置。
- **为何既有测试没发现**：`tests/test_indexer.py` 将 `_run_extraction` 整体 mock 掉了。新增 `test_review_fixes.py::TestRunExtractionContract` 用**真实提取器**钉住该契约。

### 2. `POST /files/upload` 路径穿越（未认证默认部署下可任意写文件）
- **位置**：`api/main.py` `_upload_files_impl()`；`DELETE /files/{id}` 同类隐患
- **问题**：`UploadFile.filename` 未清洗即拼接 `corpus_root / filename`，`..` 段或绝对路径可越出语料根目录；写盘发生在越界校验之前，失败还会被吞进 `failed`。
- **修复**：取 `Path(filename).name` 作为 basename（拒绝空名/`.`/`..`），写盘前强制 `resolve().is_relative_to(corpus_root)` 包含性校验；删除接口在 `unlink()` 前做同样的包含性校验（纵深防御）。
- **回归测试**：`test_review_fixes.py::TestUploadPathTraversal`（父目录段、绝对路径用原始 multipart 构造绕过 httpx 的自动清洗、正常文件不受影响）。

### 3. `/query/stream` 在事件循环线程使用跨线程 SQLite 连接 → 持久化每次必失败
- **位置**：`api/main.py` `query_stream()`
- **问题**：处理函数是同步 `def`（运行于 worker 线程，线程本地 `_get_db()` 连接绑在该线程），而 SSE 生成器在事件循环线程迭代。sqlite3 默认 `check_same_thread=True` → 每次流式查询的 `query_log` 插入/审计都抛 `ProgrammingError`，被宽泛 except 吞成一条 `error` SSE 事件。
- **修复**：持久化（含真实 `retrieved_chunks`、审计记录）封装为 `_persist_stream_query()`，经 `run_in_threadpool` 回 worker 线程执行；同时补上真实检索块（此前恒写 `[]`，`/queries` 的 source_count 恒为 0）。

### 4. 流式生成使用同步 Ollama 客户端阻塞整个事件循环
- **位置**：`api/main.py` `event_generator()`
- **问题**：`ollama.Client(...).chat(stream=True)` 的逐 token 读取是阻塞网络 IO，放在 `async def` 生成器里会把**所有**并发请求（含健康检查、其他 SSE 流）卡住数秒到数分钟。
- **修复**：改用 `ollama.AsyncClient` + `async for`；客户端统一携带 `cfg.ollama.timeout_seconds`（此前为死配置），`/health` 与 `/query` 的可达性探测客户端加 10s 超时，防止 Ollama 挂起时健康检查无限阻塞。
- **测试同步更新**：`tests/test_api_query.py` 流式用例改为 mock `AsyncClient`（async 迭代契约）。

---

## 二、已修复（P1 — 重要）

### 5. 文件夹/工作区过滤在两条路径下会静默失效（检索越界）
- **位置**：`rag/retrieval.py` `retrieve()` 与 `_hierarchical_narrowing()`
- **问题**：
  1. `hierarchical=false` 时 `allowed_file_ids` 在检索主流程中**完全未被使用**——工作区/文件夹过滤被整体绕过（V1.1 的隔离特性失效）；
  2. 层级收窄结果与允许集交集为空时 `return None`，被调用方理解为"无限制"，退化为**全库检索**；
  3. 路径匹配增广未与 `allowed_file_ids` 求交，可把范围外文件并入候选。
- **修复**：过滤存在时 `candidate_file_ids` 永远是"允许集的子集且非空"（收窄失败回退到整个允许集；`hierarchical=false` 时直接取允许集）；路径匹配候选与允许集求交；收窄是召回启发式而**不是边界**这一语义写入注释与测试。
- **回归测试**：`test_review_fixes.py::TestHierarchicalNarrowingScope`。

### 6. 知识库状态面板（未提交的新功能）多处缺陷
- **位置**：`ui/gradio_app.py`
- **修复清单**：
  - **失败口径漏项**：`failure.phase` 只统计 `('extract','indexer')`，漏掉嵌入失败（`embed`）与摘要失败（`summarize`），导致"0 失败 0 待处理但进度卡在中间"。重构为按 **embedded > failed > extracted > pending** 优先级分类，四个数与总数恒一致；
  - **文件夹明细与总数口径不一致**：文件夹级 embedded 未过滤 `embedding_ref.is_current=1`（显示"5/5 embedded"而实际 0 个有效嵌入），已统一口径；
  - **`last_sync` 永远为空**：查询 `pipeline_run WHERE phase='indexer'` 违反该表 CHECK 约束（phase 只允许 7 个 preflight 值），永远 NULL；且回退逻辑把一次性结果永久显示为 "just now"。改为记录 UI 同步完成时间 `finished_at`（`/api/indexing/status` 中的诚实来源，CLI 同步暂不体现——见遗留项 #8）；
  - **同步错误不可见**：后台线程异常只写进内部 dict，UI 显示"同步完成"。现在 status 响应携带 `last_error`，面板新增红色错误横幅（`#kb-sync-error` + CSS），轮询结束时按 `last_error` 弹出失败 toast；
  - **部分载荷渲染错误**：同步启动时以 `{sync_running:true}` 调用渲染器导致文件夹列表被清成"未找到文件夹"、进度条写出 `NaN%`。渲染器改为区分"完整载荷/仅同步状态"两条路径；
  - **轮询失控**：补 30 分钟轮询上限；
  - **同步线程读盘上的 config.toml**：忽略运行时配置，可能索引到与检索不同的语料/向量库。改为复用启动时闭包中的 `cfg` 与 `_chroma_path`，并在结束时关闭线程内 DB 连接。

### 7. XSS（跨站脚本）系列加固
- **位置**：`ui/gradio_app.py`
- **问题与修复**：
  - JS 转义助手 `_esc`/`_escHtml` 只转义 `& < >`，用于 `title="..."`、`value="..."` 属性时可用 `"` 突破属性边界（文件夹名、远端模型 ID 均可触达）。补 `"` → `&quot;`、`'` → `&#39;`；
  - 服务端 `_citations_html()` 把**语料原文**（snippet）与文件路径直接拼进 HTML——文档内容含 `<img src=x onerror=...>` 即形成存储型 XSS（本应用是"读公司文档"的场景，风险最高）。已对 `rel_path`（含引号）与 `snippet` 做 `html.escape`；
  - 首屏服务端渲染的 `folder_options` 对 `rel_path` 未转义，含 `<`/`"` 的目录名可注入页面。已转义。

### 8. 设置面板打开时的自动加载从未生效
- **位置**：`ui/gradio_app.py`（监听器注册 vs 包装函数重赋值的顺序）
- **问题**：`settingsBtn.addEventListener('click', _openSettings)` 在 KB 状态逻辑对 `_openSettings` 做**重赋值包装**（增加自动拉取 + 10s 自动刷新）之前就把原函数值固定进监听器，包装永不生效。实测：打开设置后面板一直显示 `--`，必须手动点 Refresh。
- **修复**：监听器改为 `function(){ _openSettings(); }` 调用时解析（settings/model-badge/close 三处）。

### 9. 其余 API/Pipeline 修复
- **`POST /feedback`**：`feedback: str` 直通 DB 的 CHECK 约束，非法值（如 `"upvote"`）→ 未捕获 `IntegrityError` 500。schema 改为 `Literal["up","down"]`（非法输入 422）；
- **`GET /health`**：fasttext 导入失败时异常处理器引用未绑定变量 `model_path` → 健康检查自身 500。变量提升至 try 之前（实测：本机 fasttext 缺失时现在返回 `{"fasttext": false}` 而非 500）；
- **`GET /queries` / `GET /conversations` 分页**：`size=-1` 会被 SQLite 渲染成无 LIMIT（整表含答案全文载入内存）。补 `Query(ge=1, le=...)` 校验；
- **SSE `error` 事件**：不再把原始异常文本（SQL/路径/库内部信息）发给客户端，改为服务端 `logger.exception`（带 trace_id）+ 通用消息；
- **会话消息排序**：`ORDER BY created_at` 秒级时间戳频繁并列，历史窗口可能截错消息。补 `id` 作 tiebreaker（`rag/conversation.py` 两处）；
- **`pipeline/db.py`**：
  - 新增 `PRAGMA busy_timeout = 5000`——此前 CLI 阶段与 API/UI 并发写时任何不经重试包装的写操作直接抛 `database is locked`；
  - 修复 `v_extraction_status` 视图引用**不存在的列** `file.extract_status`（每个安装都带一个查询即报错的视图；SQLite 延迟校验让建表时不报错）。重写为从 `extraction`/`failure` 派生状态，修入 migration 2（新库）并新增 **migration 8**（存量库自动修复，已在真实库副本上验证）；
- **`pipeline/phase4_dedup.py`（去重幂等性）**：
  - 规范副本（canonical）选择不含 `excluded=0` 过滤——被排除的副本可能抢走 canonical 位，**整组文件从所有视图与下游阶段消失**；
  - 重跑不重置 `is_dup_primary`/`dup_group_id`，sha256 变更后旧标记永久残留。两处均已修复（先全量重置再按 eligible 集合重新分组）；
- **`pipeline/phase3_triage.py`（单文件故障隔离）**：fitz 惰性解析使 `is_encrypted`/`page_count`/`load_page` 均可能抛异常，但只包了 `fitz.open` 一层——一个畸形 PDF 会终止整个 triage 阶段。将函数体包入 `_triage_pdf_body()`（try/except → `is_corrupt=1`，`finally` 关闭文档），并给 `future.result()` 循环加逐文件隔离；
- **静态检查**：ruff 29 项 → 7 项（修复 20 项自动项 + 2 项手工项；其余为 FastAPI 惯用法或微风格，见遗留项）。

---

## 三、测试与端到端验证

| 验证 | 结果 |
|---|---|
| 全量测试 | **676 通过 / 0 失败**（原 667 + 新增回归 9），15s，覆盖率 66.6% |
| 新增回归测试 | `tests/test_review_fixes.py`：上传穿越×3、反馈校验×1、索引器提取契约×2、检索范围×3 |
| 真实库副本 | migration 8 修复视图后 `v_extraction_status` 可查询且数字正确 |
| API 实测（`cli.py serve`） | `/health` 正确降级（fasttext 缺失不再 500）；`/feedback` 非法值 422；`/queries?size=-1` 422；上传 `../evil.txt` 落在语料库根目录内、库外无文件 |
| UI 实测（`cli.py ui` + 浏览器自动化） | 页面加载、会话历史、文件夹树、"🔴 Ollama down"徽标准确；Sync Now 全链路（Ollama 关闭时 5 个文件在 embed 阶段失败并被**如实计数与展示**——修复前显示"0 失败 0 待处理"）；面板数值与 `/api/indexing/status` 完全一致（37.5% / 8 总 / 3 已索引 / 5 失败 / 文件夹明细一致 / 上次同步时间） |
| 工具限制（如实记录） | 自动化点击在部分悬浮元素上不稳定（IAB 后端点击/截图间歇失败），"打开设置面板→自动加载"一环以 served-HTML 断言 + 代码路径审阅佐证；此改动（#8）逻辑简单且已上线验证 served 内容 |

---

## 四、遗留项（建议后续处理，本次未改动）

按影响排序：

1. **P1 · CORS 默认 `*` + 默认无 API Key**：`api/main.py` 注释明示是刻意选择，但任何网页可对 `127.0.0.1:8765` 发起跨源读取（答案、文档全文、审计导出）。建议默认收窄为 UI 来源 + 文档强制推荐 `VAULT_API_KEY`；同时 401 响应未过 CORS 中间件（中间件注册顺序），跨源客户端看到的是不透明错误。
2. **P1 · UI 代理无鉴权/CSRF**：`/api/*`（设置含云 API Key 的写入、文件删除、`/api/indexing/sync`）无任何鉴权；默认绑定 127.0.0.1 时风险可控，`--host 0.0.0.0` 时完全暴露。未使用的 `BaseHTTPMiddleware` 导入提示鉴权是"计划中"。
3. **P1 · `phase1_walk.py` 符号链接**：`fp.resolve()` 使两个指向同目标的链接触发 UNIQUE 冲突并中止整个 walk；指向语料外的链接会把外部路径写进 `file.path`（下游会读取/哈希语料外文件）。建议跳过 symlink 并记 failure。
4. **P1 · 索引器超时**：`rag/generation.py`、`retrieval.py` 的 Ollama 调用未传 timeout（`timeout=None` = 禁用），挂起的 Ollama 会无限阻塞。建议统一走带 timeout 的客户端工厂（本次修复了 `api/main.py` 内的三处）。
5. **P2 · 一批健壮性项**：`/query` 与 `/query/stream` 的生成端点/提示词不一致（stream 侧未走 `generation.endpoint`，云端配置下会鉴权失败）；`/eval/run` 无并发闸；OTLP 导出的 traceId 非 32-hex（合规采集器会拒收）且导出在事件循环上阻塞；`/audit/export` CSV 公式注入（`=+-@` 前缀单元格）；上传一次性读入内存无大小上限；`DELETE /files/{id}` 无事务包裹；`_persist_stream_query` 不写会话消息（流式多轮对话仍由前端自理）；`migrate()` 的 `executescript` 隐式提交破坏逐迁移原子性（当前迁移皆幂等，风险后置）；thread-local DB 连接从不关闭、每线程首次访问重复 `migrate()`。
6. **P2 · pipeline 其余项**：phase 0 dry-run 计数 ×depth、部分失败后非幂等（`folder.insert` UNIQUE 冲突、`archive_file_id=0` 违 FK）；phase 1 全量重哈希 + `mtime` 列存的是 `now_iso()`（破坏 phase 4 最老 mtime 决胜）+ `--rehash` 无操作 + 超限尺寸检查被跳过；phase 2 `sf` 路径与 DB resolved 路径失配（相对/符号链接语料根时全部静默落 unknown）；phase 0/1 文件夹父子关系不回填（`.extracted` 目录永久脱树）；phase 6 报告默认写进语料根（自摄取）；phase 7 批量 CSV 无逐行容错/事务；config 无路径存在性校验、未知 section 静默忽略；只读 CLI 命令未挡 `is_db_initialized`（预初始化状态会创建空库后崩）。
7. **P2 · 观测**：索引器不写 `model_run`/`pipeline_run` 行，导致 CLI 同步在 UI"上次同步"中不可见（本次用 UI 侧时间戳兜底）；`pipeline_run.phase` CHECK 与 schema 演进脱节。

## 五、风格遗留（不阻塞）

- `api/main.py`：B008（FastAPI `File(...)` 惯用法）、RUF005、RUF015；
- `pipeline/phase1_walk.py`：SIM103、F841（`cache_root` 未用）；
- `ui/gradio_app.py`：SIM117（嵌套 with）。

---

*生成：2026-09-11 · 分支 dev · 676 tests passing*
