#!/usr/bin/env bash
# L0 本机基线：干净 venv 安装全部包，跑测试、lint、四个示例、跨进程一致性测试。
# 用法：bash field_tests/run_local.sh [python 可执行文件，默认 python3]
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${1:-python3}
OUT=$ROOT/field_tests/out/local
V=$OUT/venv
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$ROOT"
pass=0; fail=0
step() {  # step <编号> <说明> <命令...>
  local id=$1 desc=$2; shift 2
  if "$@" >"$OUT/$id.log" 2>&1; then echo "PASS $id $desc"; pass=$((pass+1))
  else echo "FAIL $id $desc（日志 field_tests/out/local/$id.log）"; fail=$((fail+1)); fi
}
"$PY" -m venv "$V" && . "$V/bin/activate"
"$PY" --version | tee "$OUT/python.txt"
step L0-1 "安装全部包" pip install -q -e sdk/python -e runtime -e components/judge -e components/memory \
  -e components/attribution -e components/proposer -e components/algorithm -e components/gate -e components/lineage \
  -e components/evolvables -e components/guard -e components/harness -e adapters/dspy -e adapters/langgraph \
  -e adapters/verl -e adapters/rsihub pytest ruff pandas pyarrow
step L0-2 "单元测试" pytest -q -p no:cacheprovider
step L0-3 "lint" ruff check .
step L0-4 "spec 是最新的" bash -c 'python -m openrsi.export_spec spec && git diff --exit-code spec'
for ex in secret_ops skill_ops judge_coevolution harness_ops; do
  rm -rf "examples/$ex/.openrsi"
  step "L0-5-$ex" "示例 $ex" bash -c "cd examples/$ex && openrsi run && openrsi best"
  rm -rf "examples/$ex/.openrsi"
done
grep -q "护栏拦截" "$OUT/L0-5-skill_ops.log" || { echo "FAIL L0-5-skill_ops 没看到护栏拦截"; fail=$((fail+1)); }
grep -q '"progress": 3.0' "$OUT/L0-5-judge_coevolution.log" || { echo "FAIL L0-5-judge_coevolution progress 不是 3.0"; fail=$((fail+1)); }
grep -q '"holdout": 1.0' "$OUT/L0-5-harness_ops.log" || { echo "FAIL L0-5-harness_ops holdout 没到 1.0"; fail=$((fail+1)); }
# 每个组件作为独立 MCP 子进程，用一致性测试检查（和其他语言的客户端看到的一样）
TMP=$(mktemp -d)
for cmd in "openrsi-gate" "openrsi-guard" "openrsi-lineage" "openrsi-algorithm --kind shinka" \
           "openrsi-algorithm --kind pareto" "openrsi-memory --path $TMP/mem" \
           "openrsi-evolvables --kind skills --path $TMP/skills" "openrsi-harness --repo $TMP/harness" \
           "openrsi-verl weights --base-model m --path $TMP/w --dry-run"; do
  n=$((${n:-0}+1))
  step "L0-6-$n" "一致性测试：$cmd" openrsi-conformance -- $cmd
done
rm -rf "$TMP"
echo "合计：$pass 通过，$fail 失败"
