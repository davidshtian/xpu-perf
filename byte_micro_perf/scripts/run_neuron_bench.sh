#!/usr/bin/env bash
# Run all NEURON benchmark workloads under workloads/basic/*/neuron/
# Usage: ./scripts/run_neuron_bench.sh [--device 0] [--timeout 300] [--output results.log]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"

DEVICE=0
TIMEOUT=300   # 5 minutes per workload
OUTPUT="${WORKDIR}/results/neuron_bench_$(date +%Y%m%d_%H%M%S).log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2";  shift 2 ;;
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --output)   OUTPUT="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

VENV="/opt/aws_neuronx_venv_pytorch_2_9"
source "${VENV}/bin/activate"

mkdir -p "$(dirname "$OUTPUT")"

PASS=0
FAIL=0
SKIP=0
declare -a FAILED_LIST=()

log() { echo "$*" | tee -a "$OUTPUT"; }

log "============================================================"
log "NEURON Benchmark Run: $(date)"
log "Device: ${DEVICE}  Timeout: ${TIMEOUT}s"
log "============================================================"
log ""

mapfile -t WORKLOADS < <(find "${WORKDIR}/workloads/basic" -path "*/neuron/*.json" | sort)
TOTAL="${#WORKLOADS[@]}"
log "Found ${TOTAL} workload files"
log ""

# Remove stale neuron compile lock files (lock exists but no active compilation)
cleanup_stale_locks() {
    for lockfile in $(find /var/tmp/neuron-compile-cache/ -name "*.lock" 2>/dev/null); do
        module_hash=$(basename "$(dirname "$lockfile")")
        if ! ps aux 2>/dev/null | grep -qF "$module_hash"; then
            rm -f "$lockfile"
        fi
    done
}

IDX=0
for WFILE in "${WORKLOADS[@]}"; do
    IDX=$((IDX + 1))
    REL="${WFILE#${WORKDIR}/}"
    log "------------------------------------------------------------"
    log "[${IDX}/${TOTAL}] ${REL}"
    log "Start: $(date +%H:%M:%S)"
    cleanup_stale_locks

    START_TS=$(date +%s)

    set +e
    timeout "${TIMEOUT}" python "${WORKDIR}/launch.py" \
        --backend NEURON \
        --device "${DEVICE}" \
        --workload "${WFILE}" \
        2>&1 | tee -a "$OUTPUT"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    if [[ $EXIT_CODE -eq 0 ]]; then
        log "PASS  (${ELAPSED}s)"
        PASS=$((PASS + 1))
    elif [[ $EXIT_CODE -eq 124 ]]; then
        log "TIMEOUT after ${TIMEOUT}s"
        SKIP=$((SKIP + 1))
        FAILED_LIST+=("TIMEOUT: ${REL}")
    else
        log "FAIL  exit=${EXIT_CODE}  (${ELAPSED}s)"
        FAIL=$((FAIL + 1))
        FAILED_LIST+=("FAIL(${EXIT_CODE}): ${REL}")
    fi
    log ""
done

log "============================================================"
log "Summary: PASS=${PASS}  FAIL=${FAIL}  TIMEOUT=${SKIP}  TOTAL=${TOTAL}"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    log "Failed/Timed-out:"
    for ITEM in "${FAILED_LIST[@]}"; do
        log "  ${ITEM}"
    done
fi
log "Full log: ${OUTPUT}"
log "============================================================"
