#!/usr/bin/env bash
# 跑一组对照实验。先跑过 make_recipes.py 和 openrsi-rsihub install。
#   bash run_arm.sh /path/to/RSIHub baseline [代数，默认按配方]
set -euo pipefail
RSIHUB=$(cd "${1:?用法：run_arm.sh /path/to/RSIHub <baseline|openrsi|shinka> [代数]}" && pwd)
ARM=${2:?缺少组名}
GENS=${3:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$HERE/out; mkdir -p "$OUT"
RECIPE=$RSIHUB/recipes/openrsi_parity_$ARM/evolve.yaml
WS=${WORKSPACE:-$RSIHUB/runs/openrsi-parity-$ARM}
ASSET_ROOT=${EVOLVE_ASSET_DIR:-$RSIHUB/.evolve-assets/terminal-bench-2.0}
DATASET=$ASSET_ROOT/raw/terminal-bench
ENV_FILE=${ENV_FILE:-$RSIHUB/.env}
[[ -f $RECIPE ]] || { echo "没有 $RECIPE，先跑 make_recipes.py" >&2; exit 2; }
[[ -d $DATASET ]] || { echo "没有 $DATASET，先跑 RSIHub 的 setup_terminal_bench.sh（见 README）" >&2; exit 2; }
[[ -e $WS ]] && { echo "$WS 已存在；要续跑直接用 RSIHub 的 run 命令，要重跑先删掉它" >&2; exit 2; }
GENS=${GENS:-$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment']['max_generations'])" "$RECIPE")}
U=(uv run --frozen); [[ -f $ENV_FILE ]] && U+=(--env-file "$ENV_FILE")
LOG=$OUT/$ARM.log
cd "$RSIHUB"
{
  echo "### $(date -u +%FT%TZ) arm=$ARM gens=$GENS rsihub=$(git rev-parse HEAD)"
  "${U[@]}" evolve init "$WS" --recipe-path "$RECIPE" --dataset "$DATASET"
  "${U[@]}" "$WS/evolve" preflight "$WS"
  "${U[@]}" "$WS/evolve" run "$WS" --max-generations "$GENS" --children-per-gen 1
  "${U[@]}" "$WS/evolve" status "$WS"
  "${U[@]}" "$WS/evolve" verify "$WS"
  "${U[@]}" "$WS/evolve" report "$WS" | tee "$OUT/$ARM.report.txt"
  echo "### done $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$LOG"
echo "workspace: $WS"
