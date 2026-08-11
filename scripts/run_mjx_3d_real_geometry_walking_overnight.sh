#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results

echo "Running real-geometry MJX preflight..."
python -m scripts.mjx_3d_walking_smoke \
  --geometry real \
  --physics-profile cg12 \
  --batch-size 8 \
  --steps 4 \
  --mujoco-gl disable

run_id="$(date +%Y%m%d_%H%M%S)"
output="results/mjx_3d_real_geometry_walking_h200_seed0_${run_id}"
log="${output}.log"

nohup python -m scripts.train_mjx_3d_real_geometry_walking \
  --preset h200 \
  --seed 0 \
  --out "${output}" \
  >"${log}" 2>&1 &

pid=$!
echo "PID=${pid}"
echo "log=${log}"
echo "output=${output}"
