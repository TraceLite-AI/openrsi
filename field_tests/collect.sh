#!/usr/bin/env bash
# 把所有测试结果打包成一个文件，发回来即可（不含模型权重和数据）。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
tar czf openrsi-field-report.tgz --exclude='out/local/venv' --exclude='*.parquet' --exclude='*.safetensors' \
  --exclude='*.bin' --exclude='*.pt' --exclude='out/verl/v4/weights/checkpoints/*/*' out
echo "$ROOT/openrsi-field-report.tgz（$(du -h openrsi-field-report.tgz | cut -f1)）"
