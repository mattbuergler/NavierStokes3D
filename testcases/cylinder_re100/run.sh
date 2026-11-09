#!/bin/bash
# Runs the Re = 100 cylinder case on one GPU.
#
#   testcases/cylinder_re100/run.sh [n16|n32] [outdir]
#
# Output goes to `rerun/<case>` in this folder (gitignored), so the stored results are not overwritten.
# Select the GPU with CUDA_VISIBLE_DEVICES and set JULIA if julia is not on the PATH.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
case="${1:-n16}"
outdir="${2:-$here/rerun/$case}"

case "$case" in
    #            Re  cells/D  t_end (D/V∞)  output interval (steps)
    n16) args=(100 16 200 50) ;;    # 6400 steps
    n32) args=(100 32 150 100) ;;   # 9600 steps
    *) echo "unknown case: $case (use n16 or n32)" >&2; exit 1 ;;
esac

cd "$here/../.."
mkdir -p "$outdir"
NS3D_USE_GPU=true "${JULIA:-julia}" --project -O3 --check-bounds=no scripts/benchmark.jl \
    "${args[0]}" "${args[1]}" "${args[2]}" "$outdir" "${args[3]}"
