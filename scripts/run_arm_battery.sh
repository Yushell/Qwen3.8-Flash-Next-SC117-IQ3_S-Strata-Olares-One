#!/bin/bash
# run_arm_battery.sh <tag> <served-model-id>
# One identical battery per recipe arm, run on the Olares box while its lane serves :8001.
#   probes (functional + decode/prefill) -> needle 32k/128k/250k -> strata bench (+32K prompt)
# Artifacts: ~/Tools/arm-<tag>/
set -u
TAG="$1"; MODEL="$2"
D="$HOME/Tools/arm-$TAG"
mkdir -p "$D"
LOG="$D/battery.log"

# The battery must not start against a lane that is still loading: a 68 GB expert tier takes
# minutes to fill, and every request would otherwise fail against a closed port.
echo "=== waiting for :8001 to report healthy (up to 100 min) ===" | tee "$LOG"
ready=0
for _ in $(seq 1 600); do
    if curl -sf --max-time 5 http://127.0.0.1:8001/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 10
done
if [[ "$ready" != "1" ]]; then
    echo "FAIL: lane never became healthy; battery not run" | tee -a "$LOG"
    exit 1
fi
echo "lane healthy after the wait" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== $TAG ($MODEL): functional + speed probes $(date -Is) ===" | tee -a "$LOG"
LMB_MODEL="$MODEL" python3 -u "$HOME/Tools/probe_lfm.py" 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== $TAG: needle 32k/128k/250k x depths 10/50/90 $(date -Is) ===" | tee -a "$LOG"
cd "$HOME/Strata" || exit 1
python3 tools/needle_bench.py --url http://127.0.0.1:8001 \
  --lengths 32k,128k,250k --depths 10,50,90 --out "$D/needles.json" 2>&1 | tail -24 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== $TAG: engine suite (chat / 4K / 16K / agent / 32K) $(date -Is) ===" | tee -a "$LOG"
python3 tools/strata_bench.py --url http://127.0.0.1:8001 --long \
  --out "bench/results/$(date +%Y-%m-%d)-arm-$TAG" 2>&1 | tail -28 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== $TAG: state ===" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$LOG"
free -g | sed -n "1,3p" | tee -a "$LOG"
echo BATTERY-DONE-$TAG | tee -a "$LOG"
