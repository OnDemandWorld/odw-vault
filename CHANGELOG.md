# Changelog

All notable changes to ODW.ai Vault will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> 版本对照：项目 semver（本文件）与套件里程碑（V1.1-V1.6，见仓库根 `roadmap/`）并存。
> 套件 V1.4-V1.6 的 Vault 功能（审计导出、分布式追踪、流式引用修复、多 provider 设置）
> 落在项目 0.2.x 内，见下方对应条目。

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
