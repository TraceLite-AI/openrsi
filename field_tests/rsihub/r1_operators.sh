#!/usr/bin/env bash
# R1（不花钱，不需要 Docker）：把 OpenRSI 算子装进 RSIHub，确认 RSIHub 能发现、描述、校验配置，
# 并且在 RSIHub 的 Python 里能 import 到 OpenRSI。
#   bash field_tests/rsihub/r1_operators.sh /path/to/RSIHub
# 在装了 openrsi 的环境里运行（openrsi-rsihub 命令要能找到）。
set -uo pipefail
RSIHUB=$(cd "${1:?用法：r1_operators.sh /path/to/RSIHub}" && pwd)
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/out/rsihub
mkdir -p "$OUT"; LOG=$OUT/r1.log; : > "$LOG"
pass=0; fail=0
check() {  # check <编号> <说明> <命令...>
  local id=$1 desc=$2; shift 2
  echo "### $id $desc: $*" >> "$LOG"
  if "$@" >> "$LOG" 2>&1; then echo "PASS $id $desc"; pass=$((pass+1)); else echo "FAIL $id $desc"; fail=$((fail+1)); fi
}
expect_fail() { ! "$@"; }

ours=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
theirs=$(cd "$RSIHUB" && uv run --frozen python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
echo "openrsi 的 Python $ours，RSIHub 的 Python $theirs" | tee -a "$LOG"
[[ $ours == "$theirs" ]] || echo "注意：版本不同时，算子脚本里写入的包路径（pydantic_core 等编译模块）可能加载失败，见 R1-5"

check R1-1 "安装算子" openrsi-rsihub install "$RSIHUB" --overwrite
cd "$RSIHUB"
E=(uv run --frozen evolve)
check R1-2 "RSIHub 能发现三个算子" bash -c "${E[*]} operator list | tee /dev/stderr | grep -c openrsi_ | grep -qx 3"
for op in validate/openrsi_guard gate/openrsi_no_regression select/openrsi_shinka; do
  check "R1-3-${op%%/*}" "describe $op" "${E[@]}" operator describe "$op" --json
done
check R1-4a "合法配置通过" "${E[@]}" operator check gate/openrsi_no_regression --config '{"max_task_regressions": 1, "allow_equal": true}'
check R1-4b "非法配置被拒" expect_fail "${E[@]}" operator check gate/openrsi_no_regression --config '{"max_task_regressions": -1}'
check R1-4c "未知字段被拒" expect_fail "${E[@]}" operator check validate/openrsi_guard --config '{"no_such_field": 1}'
check R1-4d "列表配置通过" "${E[@]}" operator check validate/openrsi_guard --config '{"protected_paths": ["evaluator/**"], "check_code": true}'
check R1-5 "RSIHub 的 Python 能 import OpenRSI（直接装的或者脚本里写入的路径）" \
  uv run --frozen python -c "import runpy; runpy.run_path('library/gate/openrsi_no_regression.py', run_name='probe'); \
import openrsi_rsihub.core, openrsi_gate, openrsi_guard; print('import ok')"
echo "合计：$pass 通过，$fail 失败。日志 $LOG"
