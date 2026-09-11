#!/usr/bin/env bash
# Continue from the selected stable PPO actor, focusing command tracking.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
student="${ROLLING_STUDENT:-results/rolling_command_distill_medium_fast/student_params}"
restore="${ROLLING_TRACKING_RESTORE:-results/rolling_ppo_actor_probe_20260911_123158/params_final}"
out="${ROLLING_TRACKING_OUT:-results/rolling_ppo_tracking_$(date +%Y%m%d_%H%M%S)}"
for source in "$student" "$restore"; do
  if [[ ! -f "$source" ]]; then
    echo "Missing model: $source" >&2
    exit 2
  fi
done
for artifact in "$out" "${out}.log" "${out}_diagnostics.zip"; do
  if [[ -e "$artifact" ]]; then
    echo "Already exists: $artifact; set a new ROLLING_TRACKING_OUT." >&2
    exit 2
  fi
done
mkdir -p "$(dirname "$out")"
set +e
python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "$student" \
  --restore-ppo "$restore" \
  --geometry rollingquad_2_abd10_no_self_collision \
  --command-conditioned --rolling-snapshots \
  --forward-command-min-m-s 0.4111 --forward-command-max-m-s 0.8110 \
  --turn-command-min-rad-s 0.02 --turn-command-max-rad-s 0.08 \
  --turn-command-straight-fraction 0.4 --command-interval-s 10 \
  --snapshot-pool-size 2048 --eval-snapshot-pool-size 1024 \
  --snapshot-sampling tracking_focus \
  --snapshot-warmup-min-steps 100 --snapshot-warmup-max-steps 300 \
  --forward-tracking-weight 6 --yaw-tracking-weight 3 \
  --episode-length 500 --minimum-success-turns 5 \
  --preset h200 --max-devices 4 --envs 2048 --eval-envs 256 \
  --batch-size 256 --num-minibatches 8 --unroll-length 20 \
  --steps 1966080 --num-evals 17 --updates-per-batch 1 \
  --dr-strength 0 --observation-noise-scale 1 --student-anchor-weight 0 \
  --initial-policy-std 0.02 --entropy-cost 0 --discounting 0.99 \
  --learning-rate 0.000003 --learning-rate-schedule adaptive_kl \
  --desired-kl 0.01 --min-learning-rate 0.0000001 --max-learning-rate 0.00001 \
  --clipping-epsilon 0.05 --max-grad-norm 0.5 \
  --fixed-eval-envs 256 --fixed-eval-observation-noise-scale 1 \
  --seed 20260914 --fixed-eval-seed 20260915 \
  --stop-success-drop 0.10 --out "$out" 2>&1 | tee "${out}.log"
run_status=$?
set -e
if [[ -d "$out" ]]; then
  python -m scripts.collect_rolling_ppo_diagnostics "$out" \
    --out "${out}_diagnostics.zip" --log "${out}.log"
fi
exit "$run_status"
