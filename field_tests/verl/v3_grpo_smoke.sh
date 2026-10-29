#!/usr/bin/env bash
# V3（GPU）：小模型 GRPO，奖励用 OpenRSI 判别器；CONTROL=1 时再用 verl 自带 gsm8k 奖励跑一遍对照。
# 先跑 V2 生成数据。参数名按 verl 0.4 以后的 main_ppo 写；你的 verl 版本报未知参数就告诉我。
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct N_GPUS=1 STEPS=30 CONTROL=1 bash field_tests/verl/v3_grpo_smoke.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$HERE/../out/verl
DATA=${DATA:-$OUT/data}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
N_GPUS=${N_GPUS:-1}
STEPS=${STEPS:-30}
mkdir -p "$OUT"
export PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH}

run() {  # run <名字> [额外参数...]
  local name=$1; shift
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA/train.parquet" data.val_files="$DATA/test.parquet" \
    data.train_batch_size=${BATCH:-64} data.max_prompt_length=512 data.max_response_length=${RESP_LEN:-512} \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=${ROLLOUT:-vllm} actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 trainer.logger='["console"]' \
    trainer.total_training_steps="$STEPS" trainer.test_freq=10 trainer.save_freq=-1 trainer.val_before_train=True \
    trainer.project_name=openrsi-field trainer.experiment_name="$name" \
    "$@" 2>&1 | tee "$OUT/v3-$name.log"
}

run openrsi custom_reward_function.path="$HERE/gsm8k_reward.py" custom_reward_function.name=compute_score
[[ ${CONTROL:-0} == 1 ]] && run verl_builtin

echo "--- 验证集分数（训练前 / 训练后）"
for f in "$OUT"/v3-*.log; do
  echo "$(basename "$f"):"; grep -oE "val-core/[^ ]*reward[^ ]*: ?[0-9.]+" "$f" | sed -n '1p;$p' || true
done
echo "通过标准：两次运行都跑完；训练前的验证分数两者一致（同一模型、同一数据、贪心解码）；训练后分数上升。"
echo "把 field_tests/out/verl/v3-*.log 发回来（太大就发最后 200 行和所有 val-core 行）。"
