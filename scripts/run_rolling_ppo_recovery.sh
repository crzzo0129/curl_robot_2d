#!/usr/bin/env bash
# Run from the Linux cloud project's curl_robot_2d directory.
set -euo pipefail
stage="${1:-critic}"
case "$stage" in
  critic|actor) ;;
  *) echo 'Usage: bash scripts/run_rolling_ppo_recovery.sh critic|actor' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
student="${ROLLING_STUDENT:-results/rolling_command_distill_medium_fast/student_params}"
critic_out="${ROLLING_CRITIC_OUT:-results/rolling_ppo_critic_warmup_v2}"
actor_out="${ROLLING_ACTOR_OUT:-results/rolling_ppo_actor_probe_v2}"
common=(
  "$student"
  --geometry rollingquad_2_abd10_no_self_collision
  --command-conditioned --rolling-snapshots
  --snapshot-pool-size 512 --eval-snapshot-pool-size 256
  --snapshot-warmup-min-steps 100 --snapshot-warmup-max-steps 300
  --forward-command-min-m-s 0.4111 --forward-command-max-m-s 0.8110
  --turn-command-min-rad-s 0.02 --turn-command-max-rad-s 0.08
  --turn-command-straight-fraction 0.4 --command-interval-s 10
  --episode-length 500 --minimum-success-turns 5
  --preset h200 --max-devices 4 --envs 2048 --eval-envs 64
  --batch-size 256 --num-minibatches 8 --unroll-length 20
  --dr-strength 0 --observation-noise-scale 1
  --student-anchor-weight 0 --initial-policy-std 0.02
  --entropy-cost 0 --discounting 0.99 --reward-scaling 1
  --fixed-eval-envs 64 --fixed-eval-seed 123456 --seed 0
  --clipping-epsilon 0.05 --max-grad-norm 0.5
  --training-metrics-steps 40960 --stop-success-drop 0.10
)
if [[ "$stage" == critic ]]; then
  out="$critic_out"
  options=(--critic-only --steps 409600 --num-evals 11
           --learning-rate 0.0001 --updates-per-batch 2)
else
  out="$actor_out"
  if [[ ! -f "$critic_out/params_final" ]]; then
    echo "Missing completed critic warmup: $critic_out/params_final" >&2
    exit 2
  fi
  options=(--restore-ppo "$critic_out/params_final"
           --steps 819200 --num-evals 21 --updates-per-batch 1
           --learning-rate 0.000003 --learning-rate-schedule adaptive_kl
           --desired-kl 0.01 --min-learning-rate 0.0000001
           --max-learning-rate 0.00001)
fi
if [[ -e "${out}.log" || -e "${out}_diagnostics.zip" || -d "$out" ]]; then
  echo "Existing output for $out; choose a new ROLLING_CRITIC_OUT or ROLLING_ACTOR_OUT." >&2
  exit 2
fi
mkdir -p "$(dirname "$out")"
set +e
python -u -m scripts.train_mjx_3d_roll_student_dr_ppo \
  "${common[@]}" "${options[@]}" --out "$out" 2>&1 | tee "${out}.log"
run_status=$?
set -e
# A deliberate performance stop still produces a diagnostic bundle.
if [[ -d "$out" ]]; then
  python -m scripts.collect_rolling_ppo_diagnostics "$out" \
    --out "${out}_diagnostics.zip" --log "${out}.log"
fi
exit "$run_status"
