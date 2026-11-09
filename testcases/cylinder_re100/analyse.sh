#!/bin/bash
# Prints all force statistics of one run, using data after t = 100 D/V∞.
#
#   testcases/cylinder_re100/analyse.sh <rundir>
#
# Set JULIA if julia is not on the PATH.
set -euo pipefail

dir="$(cd "$1" && pwd)"
julia="${JULIA:-julia}"
cd "$(dirname "$0")/../.."

echo "--- solver force"
"$julia" scripts/forces.jl "$dir" 100
for wd in 0.5 1.0; do
    echo "--- surface force, wall distance $wd cells"
    "$julia" --project=scripts scripts/force_surface.jl "$dir" 100 "$wd"
done
echo "--- control volume"
"$julia" --project=scripts scripts/force_cv.jl "$dir" 100
