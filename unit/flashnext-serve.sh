#!/bin/bash
# flashnext-serve.sh — supervised launcher for the Qwen3.8-Flash-Next (Strata) production lane.
#
# A supervisored launcher for the Qwen3.8-Flash-Next (Strata) production lane. Responsibilities
# beyond a bare ExecStart:
#   1. Fail the unit (exit non-zero) if the model did not load the way the lane needs it: the
#      engine's own VRAM below a floor, or its process tree's RSS below the resident-expert
#      floor (the engine then reads experts from disk instead, which is far slower).
#   2. Enforce a hard whole-GPU memory ceiling: the engine sizes its own expert cache and
#      exposes no VRAM budget flag, so the supervisor kills the server if TOTAL GPU usage (this
#      card also drives the display) exceeds VRAM_CAP_MIB twice in a row, instead of letting it
#      OOM-thrash the desktop.
#   3. Keep the system temp dir: agent shells export TMPDIR into an auto-pruned scratch dir and
#      a long-lived server must not inherit it (a pruned temp dir 503s every request).
#
# The engine is a child of serve/server.py, so every check walks the process tree: measuring
# the Python wrapper alone would report ~0.2 GiB RSS and ~0 VRAM and pass garbage.
#
# NOTE on systemd's MemoryMax: the engine keeps the routed experts in host RAM by design
# (~44-46 GiB) plus ~3 GiB of pinned KV and prints its own memory guard. The unit sets
# MemoryMax=75000M, above the lane's own footprint and below a runaway; the GPU ceiling is
# enforced here, not by systemd (the engine exposes no VRAM budget flag).
#
# Config lives in ~/.config/flashnext-serve.env (see that file for the fields).
set -u

export TMPDIR=/tmp TMP=/tmp TEMP=/tmp

ENV_FILE="${FLASHNEXT_ENV:-$HOME/.config/flashnext-serve.env}"
# shellcheck source=/dev/null
source "$ENV_FILE"
# A sourced file does not export: the server child reads STRATA_API_KEY from its own
# environment (0.2.0 requires a key for a non-loopback bind), so export it explicitly.
[[ -n "${STRATA_API_KEY:-}" ]] && export STRATA_API_KEY

STRATA_DIR="${STRATA_DIR:-$HOME/Strata}"
CONFIG="${CONFIG:?CONFIG (the strata-*.json) must be set in $ENV_FILE}"
PORT="${PORT:-8001}"
VRAM_CAP_MIB="${VRAM_CAP_MIB:-23600}"
VRAM_FLOOR_MIB="${VRAM_FLOOR_MIB:-14000}"
RSS_FLOOR_GIB="${RSS_FLOOR_GIB:-30}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"

STATE_DIR="$HOME/.local/state/flashnext-serve"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/server.log"
GUARD_LOG="$STATE_DIR/supervisor.log"

PY="$STRATA_DIR/.venv/bin/python"
[[ -x "$PY" ]] || { echo "$(date -Is) FAIL: no engine venv at $PY" >> "$GUARD_LOG"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "$(date -Is) FAIL: config not found: $CONFIG" >> "$GUARD_LOG"; exit 1; }

# --- the engine runs as a child of serve/server.py, so the checks walk the tree ---
engine_pids() {
    local p
    for p in $(pgrep -P "$SERVER_PID" 2>/dev/null); do
        echo "$p"
        pgrep -P "$p" 2>/dev/null
    done
}

tree_rss_gib() {
    local pid total=0 rss
    for pid in "$SERVER_PID" $(engine_pids); do
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -dc '0-9')
        total=$(( total + ${rss:-0} ))
    done
    echo $(( total / 1024 / 1024 ))
}

engine_vram_mib() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -v pids=" $(engine_pids | tr '\n' ' ') " '
            { gsub(/,/,""); if (index(pids, " " $1 " ") > 0) s += $2 }
            END { print s + 0 }'
}

echo "$(date -Is) supervisor start: $PY serve/server.py --engine strata --config $CONFIG --port $PORT" >> "$GUARD_LOG"

cd "$STRATA_DIR" || exit 1
# 0.2.0 (engine 0.1.45.3) refuses a non-loopback bind without an inference API key
# ("a non-loopback bind requires an inference API key", exit 2). This lane binds 0.0.0.0 and
# runs STRATA_API_KEY=none by default (the server warns once; see the README's security note).
# When a key is set, the supervisor exports it (a sourced file does not) and every client sends
# "Authorization: Bearer <key>" or "x-api-key: <key>" - the server does not exempt loopback.
# Keep :8001 on a trusted LAN/tailnet either way.
"$PY" -u serve/server.py --engine strata --config "$CONFIG" --host 0.0.0.0 --port "$PORT" >> "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    for p in $(engine_pids); do kill "$p" 2>/dev/null; done
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup TERM INT EXIT

# --- readiness: wait for /health to report the model loaded, and for OUR engine to exist ---
# A 2xx is not enough: the server answers /health while the engine is still filling its expert
# tier (max_context 0, loaded false), and a load check run then would see almost no VRAM.
# Nor is "loaded": true enough on its own: a stale server from a previous lane can hold the
# port and answer for us, so this also requires our own server's engine child to be alive.
ready=0
for _ in $(seq 1 $(( READY_TIMEOUT_S / 2 )) ); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "$(date -Is) FAIL: server exited during load" >> "$GUARD_LOG"
        exit 1
    fi
    if [[ -n "$(engine_pids)" ]] && \
       curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"loaded": *true'; then
        ready=1
        break
    fi
    sleep 2
done
if [[ "$ready" != "1" ]]; then
    echo "$(date -Is) FAIL: server never reported loaded in ${READY_TIMEOUT_S}s" >> "$GUARD_LOG"
    exit 1
fi

# --- reasoning default for requests that send no reasoning_effort ---
# Strata has no server-side reasoning flag: thinking is per request, or the lane's own
# /settings default. `llm start … --reasoning <level>` persists REASONING/REASONING_EFFORT
# into this unit's env (the source of truth), so apply it here as the engine default.
# Empty REASONING_EFFORT + REASONING=on leaves the engine's own template default alone.
# The ENGINE's request vocabulary (frontend.EFFORT, 0.1.45.3) is none|off|minimal|disabled|
# false (thinking off), low, medium, high|xhigh|max|maximum (=xhigh) — but the /settings
# endpoint (clean_shared_defaults) takes only none|low|medium|high, so heavy levels go out
# as "high", which the template resolves to xhigh.
re_env=$(printf '%s' "${REASONING_EFFORT:-}" | tr '[:upper:]' '[:lower:]')
case "$re_env" in
    xhigh|high|max|maximum)             re_level=high ;;
    low|medium)                         re_level="$re_env" ;;
    none|off|minimal|disabled|false)    re_level=none ;;
    "")                                 re_level="" ;;
    *)  re_level=""
        echo "$(date -Is) reasoning default: unknown REASONING_EFFORT '$REASONING_EFFORT' - left at the engine default" >> "$GUARD_LOG" ;;
esac
if [[ -z "$re_level" && "${REASONING:-on}" == "off" ]]; then
    re_level=none
fi
if [[ -n "$re_level" ]]; then
    auth_hdr=()
    # STRATA_API_KEY may be a comma list (the server accepts several); a client sends ONE.
    [[ -n "${STRATA_API_KEY:-}" ]] && auth_hdr=(-H "Authorization: Bearer ${STRATA_API_KEY%%,*}")
    applied=$(curl -sf -m 15 -X POST "http://127.0.0.1:$PORT/settings" \
        -H 'Content-Type: application/json' "${auth_hdr[@]}" \
        -d "{\"defaults\": {\"reasoning_effort\": \"$re_level\"}}" 2>/dev/null)
    echo "$(date -Is) reasoning default: ${re_level} (from REASONING_EFFORT='${REASONING_EFFORT:-}', REASONING='${REASONING:-on}') -> ${applied:-FAILED (engine did not accept it)}" >> "$GUARD_LOG"
fi

# --- guard 1: the model must be loaded the way this lane needs it ---
# The driver's compute-apps table can lag the engine's own context creation, so read it a few
# times before judging; and if it still cannot attribute VRAM to the engine's PIDs, fall back
# to whole-GPU use (a lane that failed to load leaves the card near idle and still fails the
# floor, while a healthy lane is never killed over an unreadable table).
sleep 10
evram=0
for _ in $(seq 1 12); do
    evram=$(engine_vram_mib)
    [[ "$evram" -gt 0 ]] && break
    sleep 5
done
vram_src="engine"
if [[ "$evram" -le 0 ]]; then
    evram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
    vram_src="total(fallback)"
fi
rss=$(tree_rss_gib)
echo "$(date -Is) load check: vram_used=${evram}MiB (${vram_src}) floor=${VRAM_FLOOR_MIB}MiB tree_rss=${rss}GiB floor=${RSS_FLOOR_GIB}GiB pids=$(engine_pids | tr '\n' ' ')" >> "$GUARD_LOG"
if [[ "$evram" -lt "$VRAM_FLOOR_MIB" ]]; then
    echo "$(date -Is) FAIL: VRAM ${evram} MiB (${vram_src}) below the ${VRAM_FLOOR_MIB} MiB floor (expert tier did not fill)" >> "$GUARD_LOG"
    exit 1
fi
if [[ "$rss" -lt "$RSS_FLOOR_GIB" ]]; then
    echo "$(date -Is) FAIL: process tree RSS ${rss} GiB below the ${RSS_FLOOR_GIB} GiB floor (experts are not resident; the engine fell back to disk reads)" >> "$GUARD_LOG"
    exit 1
fi

# --- guard 2: whole-GPU ceiling ---
strikes=0
while kill -0 "$SERVER_PID" 2>/dev/null; do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
    if [[ -n "$used_mib" && "$used_mib" -gt "$VRAM_CAP_MIB" ]]; then
        strikes=$((strikes + 1))
        echo "$(date -Is) WARN: GPU used ${used_mib} MiB > cap ${VRAM_CAP_MIB} MiB (strike $strikes)" >> "$GUARD_LOG"
        if [[ "$strikes" -ge 2 ]]; then
            echo "$(date -Is) FAIL: GPU memory cap exceeded, killing server" >> "$GUARD_LOG"
            exit 1
        fi
    else
        strikes=0
    fi
    sleep 5
done

echo "$(date -Is) server exited, supervisor exiting" >> "$GUARD_LOG"
exit 1
