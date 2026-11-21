# FrontierHarness Eval Kit (`Terra` + `medium`)

面向云端 Linux / Docker 环境评测 **Mizar** 对照 **Pi (`1.4.0`)** 与 **Codex CLI (`0.164.0`)** 的独立套件，对齐 [FrontierHarness Eval v1.1](https://github.com/frontier-harness-eval/eval) 的 62 题数据集与判分规则。

## 目录结构

- `Dockerfile`: Harbor 控制器镜像（`frontier-eval:v1.1.0`）
- `run_eval.py`: Docker Harbor 启动器与 `--resume` 助手（默认 `isok/gpt-6.1-terra` + `effort=medium` + `max_tokens=131072`）
- `eval_proxy.py`: 统一本地中转代理（按 Harness / Trial 路径隔离流量，强制一致模型与 `reasoning.effort`，实时落盘 `proxy_usage.jsonl` 采集 Token、Prompt Cache 命中与断点、TTFT、耗时、HTTP 状态）
- `score_report.py`: 生成单 Job 报告或**多 Job / 多 Harness 横向对比报告**（`SCORE_REPORT.md`、`live_report.html`、`score_summary.json`）
- `frontier-v1.1-images.json`: 62 题固定清单（`tb4` 17 题独立判分容器 + `tb21` 45 题）
- `vendor/checksums.txt`: Mizar release 校验清单（二进制 `vendor/bin/` 需按运行平台提前准备好）

## 云端初始化

1. 构建 Harbor 控制器镜像：

```bash
docker build -t frontier-eval:v1.1.0 - < Dockerfile
```

2. 准备 `vendor/bin/linux-amd64/{mizar,rg}`（若用 `--task-platform linux/arm64` 则准备 `linux-arm64/`）：

```bash
mkdir -p vendor/bin/linux-amd64
# 将编译好或下载的 mizar 与 rg Linux amd64 可执行文件放入 vendor/bin/linux-amd64/ 并 chmod +x
chmod +x vendor/bin/linux-amd64/mizar vendor/bin/linux-amd64/rg
```

3. 设置云端直连 API 凭据（任意 OpenAI 兼容 Responses / Chat Completions 接口）：

```bash
export OPENAI_BASE_URL="https://your-api-endpoint.example/v1"
export OPENAI_API_KEY="sk-..."
```

## 常用评测命令

### 1. Top 5 快速对照冒烟测试（Mizar vs Pi vs Codex，Terra + medium）

```bash
python3 run_eval.py \
  --agents mizar pi codex \
  --tasks "*/mvcc-lsm-compaction" "*/bun-sourcemap-leak" "*/wal-recovery-ordering" "*/session-window-debug" "*/ontology-kg-querying" \
  --concurrency 3 \
  --attempts 1 \
  --job-name terra-medium-top5
```

生成报告（包含主榜、Mizar 缓存诊断表与逐题横评矩阵）：

```bash
python3 score_report.py jobs/terra-medium-top5
```

### 2. 使用自定义编译版 Mizar 做迭代 A/B 对比

```bash
# 只跑新版 Mizar
python3 run_eval.py \
  --agents mizar \
  --mizar-bin /path/to/mizar-linux-amd64 \
  --mizar-label mizar-cachefix \
  --mizar-version 0.1.6-dev \
  --mizar-args "" \
  --tasks "*/mvcc-lsm-compaction" "*/ontology-kg-querying" \
  --job-name mizar-cachefix-top2

# 把多个独立 Job 横向合并对比（支持相同 harness 不同版本自动按 Job 名区分）
python3 score_report.py jobs/mizar-top5 jobs/mizar-cachefix-top2 jobs/pi-top5 jobs/codex-top5
```

### 3. 全集 62 题标准 FrontierHarness 评测（4 attempts / 题）

```bash
python3 run_eval.py \
  --agents mizar pi codex \
  --attempts 4 \
  --concurrency 8 \
  --detach \
  --job-name terra-medium-full
```

实时查看进度：

```bash
python3 score_report.py jobs/terra-medium-full --watch 30
```

断线或基础设施异常后清理并续跑（会自动检查并拉起已停止的 study proxy）：

```bash
python3 run_eval.py --resume terra-medium-full --filter-error-type AgentEnvironmentError
```

## 计价说明

报告默认使用 Terra 参考费率估算 USD 成本（`uncached_input=$1.00/1M`、`cached_input=$0.05/1M`、`output=$5.00/1M`）。若上游响应 `usage.cost` 已包含费用则直接采用；也可通过 `--price-in`、`--price-cached`、`--price-out` 覆盖。

## 单元测试

```bash
python3 -m unittest discover -s tests -v
```
