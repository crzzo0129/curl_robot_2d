#!/usr/bin/env bash
# Evaluate three policies on shared, new rolling states; no optimization.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
actor_run="${ROLLING_ACTOR_RUN:-results/rolling_ppo_actor_probe_20260911_123158}"
eval_out="${ROLLING_CANDIDATE_EVAL_OUT:-results/rolling_ppo_candidates_$(date +%Y%m%d_%H%M%S)}"
student="${ROLLING_STUDENT:-results/rolling_command_distill_medium_fast/student_params}"
best="$actor_run/checkpoints/000000286720/params"
final="$actor_run/params_final"
for source in "$student" "$best" "$final"; do
  if [[ ! -f "$source" ]]; then
    echo "Missing policy file: $source" >&2
    exit 2
  fi
done
if [[ -e "$eval_out" || -e "${eval_out}_diagnostics.zip" ]]; then
  echo "Output already exists: $eval_out; set a new ROLLING_CANDIDATE_EVAL_OUT." >&2
  exit 2
fi
mkdir -p "$eval_out"
for noise in 0 1; do
  panel="$eval_out/noise_$noise"
  python -u -m scripts.train_mjx_3d_roll_student_dr_ppo "$student" \
    --geometry rollingquad_2_abd10_no_self_collision \
    --command-conditioned --rolling-snapshots --eval-only \
    --compare-ppo "$best" "$final" \
    --forward-command-min-m-s 0.4111 --forward-command-max-m-s 0.8110 \
    --turn-command-min-rad-s 0.02 --turn-command-max-rad-s 0.08 \
    --turn-command-straight-fraction 0.4 --command-interval-s 10 \
    --episode-length 500 --minimum-success-turns 5 \
    --preset h200 --max-devices 4 --dr-strength 0 \
    --student-anchor-weight 0 --initial-policy-std 0.02 \
    --eval-snapshot-pool-size 1024 --fixed-eval-envs 256 \
    --snapshot-warmup-min-steps 100 --snapshot-warmup-max-steps 300 \
    --seed 20260912 --fixed-eval-seed 20260913 \
    --fixed-eval-observation-noise-scale "$noise" \
    --out "$panel" 2>&1 | tee "$panel.log"
  python -m scripts.collect_rolling_ppo_diagnostics "$panel" \
    --out "${panel}_diagnostics.zip" --log "$panel.log"
done
python -m zipfile -c "${eval_out}_diagnostics.zip" "$eval_out"
echo "Send back: ${eval_out}_diagnostics.zip"
