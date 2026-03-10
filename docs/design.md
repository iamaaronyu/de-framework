# 数据蒸馏流水线系统 — 设计文档

> 版本：v1.0
> 更新日期：2026-03-10
> 仓库：`de-framework`

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [系统总览](#2-系统总览)
3. [模块设计](#3-模块设计)
   - 3.1 [执行框架 (framework/)](#31-执行框架-framework)
   - 3.2 [NPU 推理服务 (npu_service/)](#32-npu-推理服务-npu_service)
   - 3.3 [代码下载服务 (code_download/)](#33-代码下载服务-code_download)
   - 3.4 [业务流水线 (pipelines/)](#34-业务流水线-pipelines)
4. [接口规范](#4-接口规范)
5. [数据流](#5-数据流)
   - 5.1 [开发场景](#51-开发场景)
   - 5.2 [生产场景端到端时序](#52-生产场景端到端时序)
6. [配置参考](#6-配置参考)
7. [关键设计决策](#7-关键设计决策)
8. [监控与可观测性](#8-监控与可观测性)
9. [验证方案](#9-验证方案)

---

## 1. 背景与目标

数据工程团队需要批量运行**数据蒸馏**任务：从原始语料读取数据，调用大模型（教师模型）在 NPU 推理集群上生成高质量标注，最终写出蒸馏数据集供下游训练使用。

系统需同时满足：

| 维度 | 需求 |
|------|------|
| **开发态** | 本地快速迭代，可在验证服务器用小数据集端到端跑通 |
| **生产态** | 通过云道平台提交、调度、监控，结果可复现 |
| **稳定性** | NPU 调用失败自动重试，单 Stage 故障不级联整个任务 |
| **可追溯** | 每次生产任务的代码版本、数据血缘、质量指标全链路记录 |
| **扩展性** | 新增流水线类型只需新增 `pipelines/` 子目录，不改框架代码 |

---

## 2. 系统总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据蒸馏流水线系统                                  │
│                                                                             │
│  ┌──────────────────────────┐     ┌────────────────────────────────────┐   │
│  │        开发态             │     │            生产态                   │   │
│  │                          │     │                                    │   │
│  │  本地IDE ──► 验证服务器   │     │  云道平台(UI) ──► 任务队列          │   │
│  │      │                   │     │       │                            │   │
│  │      ▼                   │     │       ▼                            │   │
│  │   CodeHub ──► CI/CD      │     │  代码下载服务 ──► 执行框架           │   │
│  └──────────────────────────┘     └────────────────────────────────────┘   │
│                                              │              │               │
│                                              ▼              ▼               │
│                              ┌──────────────────────────────────────┐      │
│                              │           CPU 业务集群                 │      │
│                              │  Pipeline Runner  ·  Data Loader      │      │
│                              │  NPU Client       ·  Stage Runner     │      │
│                              └──────────────┬───────────────────────┘      │
│                                             │  HTTP POST /v1/inference/batch│
│                                             ▼                               │
│                              ┌──────────────────────────────────────┐      │
│                              │          NPU 推理集群                  │      │
│                              │  FastAPI · Prometheus · 模型后端       │      │
│                              └──────────────────────────────────────┘      │
│                                             │  DataHub SDK                  │
│                                             ▼                               │
│                              ┌──────────────────────────────────────┐      │
│                              │     元数据管理平台 (DataHub)            │      │
│                              │  数据血缘 · 任务元数据 · 质量指标        │      │
│                              └──────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 组件职责速查

| 组件 | 所在仓库路径 | 职责 |
|------|------------|------|
| 执行框架 | `framework/` | 任务初始化、DAG 执行、进度上报、元数据采集 |
| NPU 推理服务 | `npu_service/` | 对外暴露标准化推理 REST API，采集 Prometheus 指标 |
| 代码下载服务 | `code_download/` | 从 OBS 拉取指定版本流水线代码并本地缓存 |
| 业务流水线 | `pipelines/` | 数据读取、推理调用、后处理的具体业务逻辑 |
| CI/CD | `.github/workflows/` | Lint + 测试 + 打包发布到 OBS |

---

## 3. 模块设计

### 3.1 执行框架 (`framework/`)

框架是系统的调度核心，负责将云道下发的任务参数转化为一次完整的 Pipeline 执行。框架与业务逻辑通过 `pipeline.yaml` 和 `BaseStage` 接口解耦——框架不感知具体数据处理逻辑。

#### 文件结构

```
framework/
├── models.py             # 共享数据模型
├── bootstrap.py          # 任务入口 & 执行编排
├── dag_executor.py       # pipeline.yaml 解析 + DAG 拓扑排序
├── stage_runner.py       # BaseStage 抽象 + 单 Stage 执行与重试
├── npu_invoker.py        # NPU REST 客户端（含 mini-batch 拆分 + 重试）
├── progress_reporter.py  # 向云道实时上报任务/Stage 状态
└── metadata_collector.py # DataHub 元数据写入 + 数据血缘
```

#### 核心数据模型 (`models.py`)

```
JobContext          任务上下文（job_id / pipeline_name / version / input_path / output_path）
PipelineConfig      解析自 pipeline.yaml 的流水线配置
StageConfig         单个 Stage 的配置（id / type / module / class / depends_on / config）
StageResult         Stage 执行结果（status / records_in / records_out / filtered / duration）
JobResult           整体任务结果（status / stage_results / total_records_in / total_records_out）
```

状态机：

```
Stage:  PENDING → RUNNING → SUCCESS
                          └→ FAILED
Job:    QUEUED  → RUNNING → SUCCESS
                          └→ FAILED
```

#### 任务初始化与编排 (`bootstrap.py`)

`run_job()` 是生产执行的入口，`main()` 提供 CLI 接口（`make run-dev` 使用）。

执行步骤：

```
run_job(job_id, pipeline_name, version, input_path, output_path)
  │
  ├── 1. ProgressReporter.report_job_start()       → 通知云道任务启动
  ├── 2. MetadataCollector.add_lineage()           → 预登记血缘关系
  ├── 3. _download_pipeline_code() 或 dev_mode     → 拉取/定位代码
  ├── 4. load_pipeline_config(pipeline.yaml)       → 解析配置
  ├── 5. build_execution_order(stages)             → 拓扑排序
  ├── 6. for stage in ordered_stages:
  │       reporter.report_stage_start(stage_id)
  │       StageRunner.run(stage, data, ctx)
  │       reporter.report_stage_result(result)
  │       if FAILED: break
  ├── 7. reporter.report_job_complete(status)
  └── 8. MetadataCollector.flush(job_id)           → 写入 DataHub
```

关键细节：
- `dev_mode=True` 时跳过代码下载，直接使用本地源码树，便于开发态验证
- 单 Stage 失败后立即中断（不继续执行后续 Stage），但进度/元数据仍正常上报
- 未处理异常被顶层 `try/except` 捕获，确保云道始终收到最终状态回调

#### DAG 执行引擎 (`dag_executor.py`)

**`load_pipeline_config(path)`** — 解析 YAML 为 `PipelineConfig`，`stages[].class` 字段对应到 Python 类名。

**`build_execution_order(stages)`** — 使用 **Kahn 算法**做拓扑排序，复杂度 O(V+E)：

```python
# 伪代码
in_degree = {s.id: len(s.depends_on) for s in stages}
queue = [s for s in stages if in_degree[s.id] == 0]
while queue:
    s = queue.pop()
    yield s
    for dependent in dependents[s.id]:
        in_degree[dependent] -= 1
        if in_degree[dependent] == 0:
            queue.append(dependent)
if len(ordered) != len(stages):
    raise DAGValidationError("Cycle detected")
```

校验：
- 依赖的 Stage ID 不存在 → `DAGValidationError`
- 存在环 → `DAGValidationError`

**`load_stage_instance(stage_cfg, extra_kwargs)`** — 通过 `importlib.import_module` 动态加载 Stage 类并实例化，`extra_kwargs` 用于注入 `NPUInvoker`（仅 `npu_inference` 类型 Stage 需要）。

#### Stage 执行与重试 (`stage_runner.py`)

**`BaseStage`** — 所有业务 Stage 的抽象基类：

```python
class BaseStage(ABC):
    def __init__(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]: ...
```

设计选择：`process()` 使用 `Iterator` 而非 `list`，Stage 间数据以惰性方式传递，支持大批量数据而不全量加载到内存。当前 `StageRunner` 实现为 `list(stage.process(...))` 以便统计 `records_out`，后续可优化为流式处理。

**`StageRunner.run()`** 重试逻辑：

```
attempt 0 → 执行 → 成功 → 返回 (StageResult.SUCCESS, outputs)
          └→ 失败 → sleep(2^attempt) → attempt 1 → ...
                                     └→ 超过 max_retries → 返回 (StageResult.FAILED, [])
```

`StageResult` 包含：`records_in`、`records_out`、`records_filtered`（= in - out）、`duration_seconds`、`error`。

#### NPU 调用客户端 (`npu_invoker.py`)

```
NPUInvoker
  ├── infer(model, inputs)              单次同步调用（透传到 batch 接口）
  ├── infer_batch(model, inputs, batch_size=32)  自动拆 mini-batch，聚合结果
  ├── list_models()                     查询可用模型列表
  └── health()                          健康探测（5s 超时，不抛异常）
```

重试策略（`tenacity`）：
- 触发条件：`httpx.HTTPError` 或 `NPUInvokerError`（HTTP 非 2xx / 响应格式异常）
- 退避：指数退避，`wait_exponential(min=2, max=30)`
- 最大尝试次数：`max_retries`（默认 3）

#### 进度上报 (`progress_reporter.py`)

向云道平台的所有 HTTP 调用均在 `_post()` 中捕获 `httpx.HTTPError`，**失败只记录 warning 日志，不抛出异常**——进度上报失败绝不能中断数据生产流水线。

上报时机：

| 事件 | 接口路径 |
|------|---------|
| 任务启动 | `POST /api/jobs/{job_id}/status` |
| Stage 启动 | `POST /api/jobs/{job_id}/stages/{stage_id}` |
| Stage 完成 | `POST /api/jobs/{job_id}/stages/{stage_id}`（含指标） |
| 任务完成 | `POST /api/jobs/{job_id}/status` |

#### 元数据采集 (`metadata_collector.py`)

- `add_lineage(upstream_path, downstream_path)` — 将 OBS 路径转换为 DataHub URN 格式 `urn:li:dataset:(urn:li:dataPlatform:obs,{path},PROD)`，记录血缘边
- `record_metric(key, value)` — 累积 KV 指标（如 `total_duration_seconds`、`total_records_out`）
- `flush(job_id)` — 批量写入 DataHub（run event + lineage），写入失败只 warning，不影响任务

依赖 `datahub-ingestion` 包，若未安装则优雅降级（不中断任务）。

---

### 3.2 NPU 推理服务 (`npu_service/`)

独立部署的 FastAPI 应用，对外暴露标准化推理 REST API，向内调用 NPU 硬件后端（Triton Inference Server 或厂商 SDK）。

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/health` | 健康检查，返回 `{"status": "ok"}` |
| `GET` | `/v1/models` | 查询可用模型列表及元信息 |
| `POST` | `/v1/inference` | 同步推理（单次，不推荐大批量） |
| `POST` | `/v1/inference/batch` | 批量推理（推荐，含 batch size 上限校验） |
| `GET` | `/metrics` | Prometheus 指标（供 Grafana 抓取） |

#### 请求 / 响应 Schema

```
POST /v1/inference/batch
Request:
  {
    "model": "llm-v2",
    "inputs": [
      {"text": "原始文本 1", "id": "record-001"},
      ...
    ]
  }

Response:
  {
    "model": "llm-v2",
    "outputs": [
      {"input_id": 0, "output": "...", "quality_score": 0.92},
      ...
    ],
    "latency_ms": 342.5
  }
```

错误码：

| HTTP 状态 | 触发条件 |
|-----------|---------|
| 404 | 请求的 model 不在 `AVAILABLE_MODELS` 中 |
| 400 | `len(inputs) > model.max_batch_size` |
| 500 | NPU 后端调用异常 |

#### Prometheus 指标

| 指标名 | 类型 | 标签 | 说明 |
|--------|------|------|------|
| `npu_inference_requests_total` | Counter | `model`, `status` | 总请求数（success/error） |
| `npu_inference_latency_seconds` | Histogram | `model` | 推理耗时分布 |
| `npu_inference_batch_size` | Histogram | — | batch 大小分布 |

#### 模型注册

生产环境中 `AVAILABLE_MODELS` 字典由模型注册表动态加载；当前实现为静态配置，便于开发阶段 mock：

```python
AVAILABLE_MODELS = {
    "llm-v2": {"description": "...", "max_batch_size": 64},
    "llm-v1": {"description": "...", "max_batch_size": 32},
}
```

---

### 3.3 代码下载服务 (`code_download/`)

在任务执行前，将对应版本的流水线代码从 OBS 拉取到执行节点本地，提供缓存避免重复下载。

#### API 端点

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| `GET` | `/v1/health` | — | 健康检查 |
| `GET` | `/code/check` | `pipeline`, `version` | 查询本地缓存状态 |
| `GET` | `/code/download` | `pipeline`, `version` | 下载 `code.tar.gz` |

#### 下载流程（`_fetch_from_obs`）

```
1. GET {OBS_BASE_URL}/{pipeline}/{version}/manifest.json
       └─ 解析 expected_sha256

2. GET {OBS_BASE_URL}/{pipeline}/{version}/code.tar.gz  (流式下载)
       └─ 写入临时文件，同步计算 sha256

3. 校验：actual_sha256 == expected_sha256
       └─ 不一致 → 删除临时文件，返回 502

4. shutil.move(tmp → dest)   # 原子写，避免部分文件被读取
```

#### 缓存结构

```
CODE_CACHE_DIR/          （默认 /tmp/pipeline-cache）
└── {pipeline_name}/
    └── {version}/
        └── code.tar.gz
```

缓存 key = `(pipeline_name, version)`，相同 key 直接返回本地文件，不再访问 OBS。缓存不设 TTL，版本不可变，永久有效。

#### OBS 对象路径规范

```
obs://pipeline-release-bucket/
└── {pipeline_name}/
    └── {version}/
        ├── code.tar.gz     # 流水线代码打包
        └── manifest.json   # 版本元信息（含 sha256、commit_hash、changelog）
```

`manifest.json` 结构：

```json
{
  "pipeline": "llm-distill",
  "version": "v2.3.1",
  "commit_hash": "a1b2c3d...",
  "build_time": "2026-03-10T08:00:00Z",
  "sha256": "abc123...",
  "dependencies": {"python": ">=3.10", "framework": ">=0.1.0"},
  "changelog": "..."
}
```

---

### 3.4 业务流水线 (`pipelines/`)

业务代码与执行框架通过两个接口约定解耦：

1. **`pipeline.yaml`** — 声明 Stage 列表、依赖关系、各 Stage 的 Python module 路径和 config
2. **`BaseStage.process(inputs, ctx)`** — Stage 实现的唯一入口

#### `llm-distill` 流水线

```
data_load ──► inference ──► postprocess
```

| Stage | 类型 | 职责 |
|-------|------|------|
| `DataLoadStage` | `data_loader` | 读取 JSONL 文件，逐行 yield dict，跳过格式错误行 |
| `InferenceStage` | `npu_inference` | 按 `batch_size` 拆批调用 `NPUInvoker.infer_batch()`，NPU 故障时标记 `inference_ok=False` 而非丢弃 |
| `PostprocessStage` | `data_writer` | 按 `quality_threshold` 和 `min_output_tokens` 过滤，写 JSONL 到 `output_path` |

`InferenceStage` 的故障隔离策略：NPU 调用失败时，对应 batch 的所有 record 标记 `inference_ok=False`、`npu_error=<原因>`，在下一 Stage（postprocess）中被过滤掉。这样**一个 batch 的 NPU 异常不会让整个 Stage 失败**，而是转化为 `records_filtered` 指标。

#### `data-clean` 流水线

```
data_load ──► clean
```

`CleanStage` 执行去重（`hash(text)` 内存 set）、长度过滤、语种白名单过滤，CPU-only，不调用 NPU。

#### 新增流水线步骤

```
1. 在 pipelines/ 下新建子目录
2. 编写 pipeline.yaml（声明 name / version / stages）
3. 实现各 Stage 继承 BaseStage，实现 process() 方法
4. 本地用 make run-dev 验证
5. PR → merge → CI 自动发布新版本到 OBS
```

---

## 4. 接口规范

### 4.1 服务间接口汇总

| 调用方 | 被调用方 | 协议 | 关键接口 |
|--------|---------|------|---------|
| 执行框架 (bootstrap) | 代码下载服务 | HTTP GET | `/code/download?pipeline=&version=` |
| 执行框架 (InferenceStage) | NPU 推理服务 | HTTP POST | `/v1/inference/batch` |
| 执行框架 (ProgressReporter) | 云道平台 | HTTP POST | `/api/jobs/{id}/status` |
| 执行框架 (MetadataCollector) | DataHub | DataHub SDK | `DatahubRestEmitter.emit()` |
| CI/CD | OBS 对象存储 | S3 API | `PUT /{pipeline}/{version}/code.tar.gz` |
| 代码下载服务 | OBS 对象存储 | HTTP GET | `/{pipeline}/{version}/manifest.json` |
| Grafana | NPU 推理服务 | HTTP GET | `/metrics` |

### 4.2 环境变量注入（由云道/K8s 注入）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `YUNDAO_URL` | `http://yundao-platform/` | 云道平台地址 |
| `NPU_SERVICE_URL` | `http://npu-service:8080/` | NPU 推理服务地址 |
| `CODE_DOWNLOAD_URL` | `http://code-download-service:8081/` | 代码下载服务地址 |
| `DATAHUB_GMS_URL` | `http://datahub-gms:8080/` | DataHub GMS 地址 |
| `PIPELINE_WORK_DIR` | `/tmp/de-pipelines` | 代码解压缓存目录（执行节点） |
| `OBS_BASE_URL` | — | OBS bucket 基础 URL（代码下载服务读取） |
| `CODE_CACHE_DIR` | `/tmp/pipeline-cache` | 代码 tar.gz 本地缓存目录（代码下载服务） |

---

## 5. 数据流

### 5.1 开发场景

```
┌─────────────────────────────────────────────────────────┐
│                       开发者本地                          │
│                                                         │
│  1. 编写 / 修改 Stage 代码                               │
│  2. 编写单元测试                                         │
│  3. make test        ← pytest tests/                    │
└──────────────────────────┬──────────────────────────────┘
                           │ git push / PR
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     验证服务器                            │
│                                                         │
│  4. make run-npu-service   ← 启动本地 NPU mock 服务       │
│  5. make run-dev           ← 用小数据集端到端验证          │
│     python -m framework.bootstrap                       │
│       --pipeline llm-distill --version v2.3.1           │
│       --input-path data/sample/ --dev-mode              │
└──────────────────────────┬──────────────────────────────┘
                           │ 验证通过，提交 PR
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      CodeHub                             │
│                                                         │
│  6. Code Review → Merge to main                         │
└──────────────────────────┬──────────────────────────────┘
                           │ push 到 main 触发 CI
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   CI/CD (.github/workflows/ci.yaml)      │
│                                                         │
│  7. ruff check . + pytest tests/                        │
│  8. tar -czf code.tar.gz .                              │
│  9. sha256sum → manifest.json                           │
│  10. 上传到 obs://pipeline-release-bucket/              │
│      └── llm-distill/v2.3.1/code.tar.gz                │
│      └── llm-distill/v2.3.1/manifest.json               │
└─────────────────────────────────────────────────────────┘
```

### 5.2 生产场景端到端时序

```
用户 (云道UI)          云道平台           执行节点(bootstrap)    代码下载服务      NPU推理服务      DataHub
     │                   │                      │                    │               │               │
     │ 提交任务配置        │                      │                    │               │               │
     │──────────────────►│                      │                    │               │               │
     │                   │ 入队，分配节点         │                    │               │               │
     │                   │─────────────────────►│                    │               │               │
     │                   │                      │                    │               │               │
     │                   │◄─────────────────────│ report_job_start   │               │               │
     │                   │  (status: running)   │                    │               │               │
     │                   │                      │ GET /code/check    │               │               │
     │                   │                      │───────────────────►│               │               │
     │                   │                      │◄───────────────────│ cached: false  │               │
     │                   │                      │ GET /code/download │               │               │
     │                   │                      │───────────────────►│               │               │
     │                   │                      │                    │ GET OBS manifest               │
     │                   │                      │                    │ GET OBS code.tar.gz            │
     │                   │                      │                    │ SHA256 校验    │               │
     │                   │                      │◄───────────────────│ 200 code.tar.gz│               │
     │                   │                      │ 解析 pipeline.yaml │               │               │
     │                   │                      │ 拓扑排序           │               │               │
     │                   │                      │                    │               │               │
     │                   │                      │─── Stage: data_load ───────────────────────────────│
     │                   │◄─────────────────────│ report_stage_start(data_load)      │               │
     │                   │                      │ [读取 JSONL, yield records]        │               │
     │                   │◄─────────────────────│ report_stage_result(records_in=N)  │               │
     │                   │                      │                    │               │               │
     │                   │─── Stage: inference ───────────────────────────────────────              │
     │                   │◄─────────────────────│ report_stage_start(inference)      │               │
     │                   │                      │ [mini-batch 拆分]  │               │               │
     │                   │                      │ POST /v1/inference/batch ──────────►               │
     │                   │                      │◄────────────────────────────────── outputs         │
     │                   │                      │ [重试 if 失败, 指数退避]            │               │
     │                   │◄─────────────────────│ report_stage_result(records_out=M) │               │
     │                   │                      │                    │               │               │
     │                   │─── Stage: postprocess ──────────────────────────────────────             │
     │                   │◄─────────────────────│ report_stage_start(postprocess)    │               │
     │                   │                      │ [质量过滤, 写出 JSONL]             │               │
     │                   │◄─────────────────────│ report_stage_result(records_out=K) │               │
     │                   │                      │                    │               │               │
     │                   │◄─────────────────────│ report_job_complete(SUCCESS)        │               │
     │                   │                      │ metadata.flush()   │               │               │────►│
     │ 任务完成通知        │                      │                    │               │  lineage+run  │
     │◄──────────────────│                      │                    │               │               │
```

---

## 6. 配置参考

### `pipeline.yaml` 完整字段说明

```yaml
name: "llm-distill"          # 流水线名称（与发布仓目录名一致）
version: "v2.3.1"            # 语义版本号
description: "..."           # 可选描述

stages:
  - id: data_load            # Stage 唯一标识（同一 pipeline 内不重复）
    type: data_loader        # Stage 类型：data_loader / npu_inference / cpu_transform / data_writer
    module: "pipelines.llm_distill.stages.data_load"  # Python 模块路径（import 用）
    class: "DataLoadStage"   # 模块内的类名（继承 BaseStage）
    depends_on: []           # 依赖的 stage id 列表，空表示起点
    config:                  # 透传给 BaseStage.__init__(config=...) 的 KV
      batch_size: 1000
      file_format: "jsonl"

  - id: inference
    type: npu_inference      # 此类型会额外注入 npu_invoker= 参数
    module: "pipelines.llm_distill.stages.inference"
    class: "InferenceStage"
    depends_on: [data_load]
    config:
      model: "llm-v2"
      batch_size: 32
      timeout_seconds: 120
      max_retries: 3
```

### 云道任务配置（提交时填写）

```yaml
job:
  name: "data-distill-job-001"
  pipeline:
    name: "llm-distill"
    version: "v2.3.1"
  data:
    input_path: "obs://bucket/raw/"
    output_path: "obs://bucket/distilled/"
  resources:
    cpu_nodes: 4
    npu_nodes: 2
  priority: "high"
```

---

## 7. 关键设计决策

### 7.1 CPU 集群与 NPU 集群解耦

**决策**：CPU 业务集群通过同步 REST API 调用 NPU 推理集群，两者独立部署。

**理由**：
- NPU 资源昂贵，独立集群可被多条流水线共享，避免 NPU 资源随 CPU 任务数线性扩展
- 接口标准化（兼容 OpenAI schema 风格）后，可按需替换 NPU 后端实现（Triton、vLLM、厂商 SDK）而不修改业务代码
- 故障域隔离：NPU 集群重启不影响 CPU 侧的数据处理逻辑

**批量优化**：`InferenceStage` 攒够 `batch_size` 条记录后一次调用 `POST /v1/inference/batch`，平衡单次调用延迟与吞吐量：

```
N 条记录 / batch_size = ceil(N/32) 次 HTTP 请求
vs.
N 次 HTTP 请求（逐条调用）
```

### 7.2 版本化不可变发布（Immutable Release）

**决策**：每次 CI 发布产生固定版本的 `code.tar.gz`，存入 OBS 后不可修改，通过 `manifest.json` 中的 `sha256` 校验完整性。

**理由**：生产任务必须可复现——同一 `(pipeline_name, version, input_path)` 组合，任何时间执行结果应一致。若允许就地修改版本代码，会破坏历史任务的可回溯性。

**实现**：
- CI 用 `git rev-parse HEAD` 将 commit hash 写入 `manifest.json`
- 代码下载服务下载后进行 SHA256 校验，不一致直接拒绝（502）
- 缓存以 `(pipeline, version)` 为 key 永久保存，不设 TTL

### 7.3 执行框架与业务代码分离

**决策**：框架负责调度、重试、上报等基础能力；业务 Stage 只关注数据处理逻辑，通过 `pipeline.yaml` 声明式描述 DAG。

**理由**：
- 新增流水线无需修改框架代码
- 框架升级（如改进重试策略）不需要业务 Stage 感知
- `BaseStage.process()` 的 `Iterator` 接口天然支持流式处理，后续可扩展为流式管道

**约定**：
- `stage.type == "npu_inference"` → 框架自动注入 `NPUInvoker` 实例
- 其他类型 Stage 不感知 NPU，保持纯 CPU 逻辑

### 7.4 进度上报与元数据采集的容错策略

**决策**：进度上报和元数据写入的失败均不中断流水线主流程。

**理由**：这两类操作属于可观测性基础设施，其故障不应影响核心数据生产能力。宁可丢失监控数据，也不因监控系统不可用而停止生产任务。

**实现**：
- `ProgressReporter._post()` 捕获所有 `httpx.HTTPError`，只记录 `WARNING` 日志
- `MetadataCollector.flush()` 的 DataHub 写入异常同样只 `WARNING`，不 `raise`

### 7.5 `InferenceStage` 的局部降级策略

**决策**：NPU 调用失败时，将受影响的 record 标记为 `inference_ok=False`，而非让整个 Stage 失败。

**理由**：大批量任务（如百万级数据）中，NPU 偶发错误不应导致全量数据重跑。失败 record 在 `PostprocessStage` 中被过滤，最终体现为 `records_filtered` 指标。运营侧可通过监控 `records_filtered` 率判断是否需要补跑。

---

## 8. 监控与可观测性

### 8.1 NPU 推理服务指标（Prometheus）

Grafana Dashboard 建议面板：

| 面板 | PromQL 示例 |
|------|------------|
| QPS | `rate(npu_inference_requests_total[1m])` |
| 错误率 | `rate(npu_inference_requests_total{status="error"}[5m]) / rate(npu_inference_requests_total[5m])` |
| P99 延迟 | `histogram_quantile(0.99, rate(npu_inference_latency_seconds_bucket[5m]))` |
| Batch 大小均值 | `rate(npu_inference_batch_size_sum[5m]) / rate(npu_inference_batch_size_count[5m])` |

### 8.2 任务级别指标（通过云道 + DataHub）

每次任务完成后写入 DataHub 的指标：

| 指标 | 来源 |
|------|------|
| `total_duration_seconds` | bootstrap.py 计时 |
| `total_records_out` | 所有 Stage `records_out` 汇总 |
| `records_filtered` | postprocess Stage 结果 |

### 8.3 告警建议

| 告警规则 | 阈值 |
|---------|------|
| NPU 错误率过高 | 5min 错误率 > 5% |
| NPU P99 延迟过长 | P99 > 30s 且持续 10min |
| 任务连续失败 | 同一 pipeline 连续 3 次失败 |
| `records_filtered` 率过高 | filtered / in > 50% |

---

## 9. 验证方案

### 9.1 开发态验证

```bash
# 1. 单元测试（包含 DAG、Stage、NPU 客户端、FastAPI 端点）
make test

# 2. 启动 NPU mock 服务
make run-npu-service   # 监听 :8080

# 3. 用本地样本数据端到端跑通（dev-mode，不下载代码，不实际写 DataHub）
make run-dev
# 等价于：
python -m framework.bootstrap \
  --pipeline llm-distill --version v2.3.1 \
  --input-path data/sample/ --output-path data/output/ \
  --dev-mode

# 4. 检查 data/output/ 下生成的 distilled_dev-local-001.jsonl
```

### 9.2 生产态全链路验证

1. **代码下载验证**：调用 `GET /code/check` 确认缓存命中，调用 `GET /code/download` 检查 SHA256 校验通过
2. **NPU 调用验证**：在云道提交小规模任务，观察 Grafana 中 `npu_inference_requests_total{status="success"}` 上涨
3. **进度上报验证**：云道 UI 中任务状态实时变化，各 Stage 完成后显示 `records_in/out`
4. **元数据验证**：任务完成后在 DataHub 中查询对应 dataset URN，确认血缘关系图已建立
5. **可复现性验证**：相同 `(pipeline_name, version, input_path)` 提交两次，对比 `distilled_*.jsonl` 文件的 MD5

### 9.3 测试覆盖范围

| 测试文件 | 覆盖点 |
|---------|--------|
| `test_dag_executor.py` | 线性/菱形 DAG 排序、环检测、未知依赖检测 |
| `test_stage_runner.py` | 透传/过滤/失败/重试后成功四种场景 |
| `test_npu_invoker.py` | 健康检查、mini-batch 拆分计数、响应格式异常 |
| `test_npu_service.py` | 全部 API 端点（health/models/inference/batch/metrics）含边界错误 |
| `test_pipeline_config.py` | `pipeline.yaml` 解析字段完整性 |
