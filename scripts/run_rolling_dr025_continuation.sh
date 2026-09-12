#!/usr/bin/env bash
# Continue the deployed step-81920 rolling PPO with mild deploy DR.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

student="${ROLLING_DR_STUDENT:-results/rolling_low_speed_20260912_065438/actor/checkpoints/000000081920/student_params}"
restore="${ROLLING_DR_RESTORE:-results/rolling_low_speed_20260912_065438/actor/checkpoints/000000081920/params}"
steering="${ROLLING_DR_STEERING_CALIBRATION:-assets/controllers/rollingquad_abd10_high_speed_steering_v1.json}"
out="${ROLLING_DR_OUT:-results/rolling_dr025_lateral050_$(date +%Y%m%d_%H%M%S)}"

for source in "$student" "$restore" "$steering"; do
  if [[ ! -f "$source" ]]; then
    echo "Missing checkpoint: $source" >&2
    exit 2
  fi
done
for artifact in "$out" "${out}.log" "${out}_diagnostics.zip"; do
  if [[ -e "$artifact" ]]; then
    echo "Already exists: $artifact; set a new ROLLING_DR_OUT." >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$out")"
set +e
python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "$student" \
  --restore-ppo "$restore" \
  --geometry rollingquad_2_abd10_no_self_collision \
  --command-conditioned --rolling-snapshots \
  --steering-calibration "$steering" \
  --forward-command-min-m-s 0.4111 --forward-command-max-m-s 0.8110 \
  --turn-command-min-rad-s 0.02 --turn-command-max-rad-s 0.08 \
  --turn-command-straight-fraction 0.4 --command-interval-s 10 \
  --snapshot-pool-size 2048 --eval-snapshot-pool-size 1024 \
  --snapshot-sampling uniform \
  --snapshot-warmup-min-steps 100 --snapshot-warmup-max-steps 300 \
  --episode-length 500 --minimum-success-turns 5 \
  --terminate-lateral-drift-m 0.50 \
  --preset h200 --max-devices 4 --envs 2048 --eval-envs 256 \
  --batch-size 256 --num-minibatches 8 --unroll-length 20 \
  --steps 819200 --num-evals 21 --updates-per-batch 1 \
  --dr-strength 0.25 --observation-noise-scale 1 \
  --student-anchor-weight 0.02 \
  --initial-policy-std 0.02 --entropy-cost 0 --discounting 0.99 \
  --learning-rate 0.000003 --learning-rate-schedule adaptive_kl \
  --desired-kl 0.01 --min-learning-rate 0.0000001 \
  --max-learning-rate 0.000003 \
  --clipping-epsilon 0.05 --max-grad-norm 0.5 \
  --fixed-eval-envs 0 \
  --seed 20260916 --memory-fraction 0.80 --mujoco-gl disable \
  --out "$out" 2>&1 | tee "${out}.log"
run_status=$?
set -e

if [[ -d "$out" ]]; then
  python -m scripts.collect_rolling_ppo_diagnostics "$out" \
    --out "${out}_diagnostics.zip" --log "${out}.log"
fi
exit "$run_status"
