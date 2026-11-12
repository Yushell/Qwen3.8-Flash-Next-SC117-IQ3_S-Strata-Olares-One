#!/usr/bin/env python3
"""Shared helpers for the measurement scripts in this directory. Run them on the serving machine, against 127.0.0.1:8001.

Every figure is the engine's own: the per-request record in /metrics (prompt_read, reused, prompt_ms,
decode_ms, decode_tok_s, hit_rate, drafts_offered/accepted), never wall clock.
"""
import json
import os
import pathlib
import threading
import time
import urllib.request

BASE = os.environ.get("LANE_URL", "http://127.0.0.1:8001")
STRATA = pathlib.Path(os.environ.get("STRATA_DIR", "/opt/strata"))     # the Strata checkout (its docs and sources are the filler text)
HOST_CPU = int(os.environ.get("STRATA_HOST_CPU", "5"))                 # the logical CPU the engine's start-up line names for its host thread


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def model_id():
    return get("/v1/models")["data"][0]["id"]


def chat(model, messages, max_tokens, effort=None, temperature=None, tools=None, timeout=1800, extra=None):
    """One chat completion. Returns the engine's record of it plus the reply text."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if effort is not None:
        body["reasoning_effort"] = effort
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = tools
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.time() - t0
    msg = d["choices"][0]["message"]
    usage = d.get("usage") or {}
    # the engine files its record of the request as the reply goes out: wait for the one that is ours
    for _ in range(40):
        m = get("/metrics")
        last = m["requests"][0]
        if last.get("time", 0) >= t0 - 0.5 and (not usage.get("prompt_tokens")
                                                 or usage["prompt_tokens"] == last.get("prompt_tokens")):
            break
        time.sleep(0.05)
    rec = {
        "wall_s": round(wall, 2),
        "prompt_tokens": last.get("prompt_tokens"),
        "reused": last.get("reused"),
        "read": last.get("prompt_read"),
        "prompt_ms": last.get("prompt_ms"),
        "first_token_s": last.get("first_engine_token_s"),
        "out": last.get("output_tokens"),
        "decode_ms": last.get("decode_ms"),
        "tps": last.get("decode_tok_s"),
        "hit": last.get("hit_rate"),
        "drafts": last.get("drafts_offered"),
        "accepted": last.get("drafts_accepted"),
        "finish": d["choices"][0].get("finish_reason"),
        "matched": usage.get("prompt_tokens") == last.get("prompt_tokens"),
        "cache": {k: m["conversation_cache"].get(k) for k in ("slots", "parked", "evictions", "bytes", "restores")},
    }
    return rec, (msg.get("content") or ""), msg


class Sampler(threading.Thread):
    """CPU frequency by core class, package temperature and GPU power while a battery runs."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.rows, self._stop_evt = period, [], threading.Event()
        self.pcores = self._cpus("/sys/devices/cpu_core/cpus")
        self.ecores = self._cpus("/sys/devices/cpu_atom/cpus")
        self.zone = None
        for z in pathlib.Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                if (z / "type").read_text().strip() == "x86_pkg_temp":
                    self.zone = z / "temp"
            except OSError:
                pass

    @staticmethod
    def _cpus(path):
        try:
            out = []
            for part in pathlib.Path(path).read_text().strip().split(","):
                a, _, b = part.partition("-")
                out += list(range(int(a), int(b or a) + 1))
            return out
        except OSError:
            return []

    @staticmethod
    def _freq(c):
        try:
            return int(pathlib.Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()) / 1000
        except OSError:
            return 0

    @staticmethod
    def _energy_uj():
        """Package energy counter (root-only file): two reads give the average CPU package watts."""
        import subprocess
        try:
            out = subprocess.run(["sudo", "-n", "cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            return int(out), time.time()
        except Exception:
            return None, time.time()

    def run(self):
        self.e0, self.t0 = self._energy_uj()
        while not self._stop_evt.wait(self.period):
            try:
                hw = get("/metrics", timeout=5)["hardware"]
            except Exception:
                hw = {}
            row = {
                "p_mhz": sum(self._freq(c) for c in self.pcores) / max(len(self.pcores), 1),
                "e_mhz": sum(self._freq(c) for c in self.ecores) / max(len(self.ecores), 1),
                "host_mhz": self._freq(HOST_CPU),
                "pkg_c": int(self.zone.read_text()) / 1000 if self.zone else None,
                "gpu_w": hw.get("gpu_power"), "gpu_util": hw.get("gpu_util"), "gpu_c": hw.get("gpu_temp"),
                "pcie_gen": hw.get("gpu_pcie_gen"), "cpu": hw.get("cpu"),
            }
            self.rows.append(row)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)
        e1, t1 = self._energy_uj()
        pkg_w = None
        if getattr(self, "e0", None) is not None and e1 is not None and e1 > self.e0:
            pkg_w = round((e1 - self.e0) / 1e6 / (t1 - self.t0), 1)   # whole window, reads included
        busy = [r for r in self.rows if (r.get("gpu_util") or 0) > 20] or self.rows
        if not busy:
            return {}

        def mean(k):
            v = [r[k] for r in busy if r.get(k) is not None]
            return round(sum(v) / len(v), 1) if v else None
        return {"samples": len(busy), "cpu_pkg_w": pkg_w, "p_mhz": mean("p_mhz"), "e_mhz": mean("e_mhz"), "host_mhz": mean("host_mhz"),
                "pkg_c_mean": mean("pkg_c"), "pkg_c_max": max((r["pkg_c"] or 0) for r in busy),
                "gpu_w": mean("gpu_w"), "gpu_util": mean("gpu_util"), "gpu_c": mean("gpu_c"),
                "pcie_gen_max": max((r.get("pcie_gen") or 0) for r in busy), "cpu_pct": mean("cpu")}


def throttle_count():
    try:
        return int(pathlib.Path("/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count").read_text())
    except OSError:
        return None


def power_profile():
    import subprocess
    try:
        return subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None
