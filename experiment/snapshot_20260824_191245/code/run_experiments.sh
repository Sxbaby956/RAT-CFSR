#!/usr/bin/env bash
# RAT-CFSR formal experiment launcher.
#
# Runs the 3-unknown x 3-seed matrix (9 trainings) with two concurrent jobs,
# one per GPU (CUDA_VISIBLE_DEVICES 0 / 1). Each job is started under nohup
# with its own persistent log under logs/; a master log records start/finish.
#
# Usage:  ./run_experiments.sh            # launch the full matrix (2 at a time)
#         ./run_experiments.sh --dry      # print the commands without running
set -uo pipefail

cd "$(dirname "$0")"

# CUDA-capable interpreter for this project.
PY=${PY:-/home/zjut/miniconda3/envs/RAT-CFSR/bin/python}
UNKNOWNS=(5G 4G WiFi)
SEEDS=(42 123 2026)
GPUS=(0 1)
WORKERS=${WORKERS:-8}
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1

mkdir -p logs outputs
MASTER_LOG="logs/experiments_master.log"

log() {
    # Write to the master log and stderr only (not stdout), so that callers
    # using $(launch_one ...) capture just the child PID.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG" >&2
}

launch_one() {
    local unknown="$1" seed="$2" gpu="$3"
    local out="outputs/unknown_${unknown}_seed${seed}"
    local logfile="logs/train_${unknown}_seed${seed}.log"
    local cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m rat_cfsr.train
        --data-root GlobecomPOWDER
        --output-dir "$out"
        --unknown "$unknown"
        --window-ms 1.0
        --num-iq-samples 8192
        --max-windows-per-recording 256
        --batch-size 64
        --stage1-epochs 10
        --stage2-epochs 20
        --modality-dropout 0.3
        --open-set-score energy
        --energy-temperature 1.0
        --log-interval 20
        --seed "$seed"
        --workers "$WORKERS"
        --device cuda
    )
    if [[ "$DRY" -eq 1 ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "$gpu" "${cmd[*]}" >&2
        return 0
    fi
    log "START unknown=$unknown seed=$seed gpu=$gpu log=$logfile"
    nohup "${cmd[@]}" > "$logfile" 2>&1 &
    echo "$!"
}

pids=()
i=0
for unknown in "${UNKNOWNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
        i=$((i + 1))
        pid=$(launch_one "$unknown" "$seed" "$gpu")
        [[ "$DRY" -eq 1 ]] && continue
        pids+=("$pid")
        # Keep at most one job per GPU: wait when we reach a full wave.
        if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
            for p in "${pids[@]}"; do
                wait "$p" || log "WARN: job pid=$p exited non-zero"
            done
            pids=()
        fi
    done
done

# Wait for any trailing jobs.
for p in "${pids[@]}"; do
    wait "$p" || log "WARN: job pid=$p exited non-zero"
done

# Render the 3-unknown x 3-seed confusion-matrix grid once all runs finish.
log "Rendering confusion matrix grid"
"$PY" -m rat_cfsr.plot_grid --output-root outputs

log "ALL_EXPERIMENTS_DONE"
