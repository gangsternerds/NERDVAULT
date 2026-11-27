"""Server benchmark (2026-11-11 launch benchmark): one prompt set x several llama-server arms, paired, on one box.
@H@ in arms = $HOME; per-prompt gen / temperature / top_p / seed; --max-gen cap; output token ids (llama-server
/tokenize) per row; reasoning_content / <think> capture; server RssAnon peak and min MemAvailable per request;
request errors recorded instead of fatal.

Only one 27B server fits in 8 GB, so arms run as sequential servers. Prompts are cut into blocks;
for every block the arm order is reshuffled, each arm's server is started, warmed up, given every
prompt of the block, and stopped. Box drift therefore hits all arms alike, and every prompt has a same-box paired measurement in every arm.

Runs ON the box. Resumable: rows already in --out are skipped.
    python3 bench.py --arms arms.json --prompts prompts.jsonl --shard 0/6 --rep 0 --gen 256 \
        --out ~/res/results.jsonl
arms.json: {"arm": {"env": {"VAR": "val"}, "cmd": "llama-server args without --host/--port"}}
Per row: timings straight from llama-server (prompt/predicted n and ms, draft counts), wall time,
sha256 of the output text; the full output goes to <out dir>/outputs/<arm>/<prompt id>.r<rep>.txt.
"""
import argparse
import hashlib
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from result_rows import read_rows


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start(arm: str, spec: dict, logdir: str, tag: str) -> tuple[int, subprocess.Popen, str]:
    H = os.path.expanduser("~")
    env = dict(os.environ, **{k: v.replace("@H@", H) for k, v in spec.get("env", {}).items()})
    port = free_port()
    log = os.path.join(logdir, f"{arm}.{tag}.log")
    with open(log, "w") as output:
        proc = subprocess.Popen(shlex.split(spec["cmd"].replace("@H@", H)) + ["--host", "127.0.0.1", "--port", str(port)],
                                stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 1200:
            if proc.poll() is not None:
                with open(log) as output:
                    raise SystemExit(f"server {arm} died; tail of {log}:\n" + output.read()[-3000:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        return port, proc, log
            except Exception:
                pass
            time.sleep(1)
        raise SystemExit(f"server {arm} not healthy after 1200 s")
    except BaseException:
        stop(proc)
        raise


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A wrapper may exit before its server children; they belong to this arm too.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def majfaults() -> int:
    for line in open("/proc/vmstat"):
        if line.startswith("pgmajfault "):
            return int(line.split()[1])
    return 0


def request(port: int, p: dict, gen: int) -> tuple[dict, str, float]:
    t0 = time.time()
    body = {"messages": p["messages"], "max_tokens": min(p.get("gen", gen), MAXGEN), "temperature": p.get("temperature", DEFTEMP),
            "seed": p.get("seed", 0), "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    if "top_p" in p:
        body["top_p"] = p["top_p"]
    r = post(port, body)
    return r, r["choices"][0]["message"].get("content") or "", time.time() - t0


MAXGEN = 1 << 30
DEFTEMP = 0   # --temperature: the default for prompts that do not set their own (0 = greedy)


class Sampler(threading.Thread):
    """peak RssAnon (kB) of the server process, min MemAvailable (kB) of the box, while a request runs"""
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.halt, self.anon, self.avail = pid, False, 0, 1 << 62

    def run(self):
        while not self.halt:
            try:
                for l in open(f"/proc/{self.pid}/status"):
                    if l.startswith("RssAnon"):
                        self.anon = max(self.anon, int(l.split()[1]))
                for l in open("/proc/meminfo"):
                    if l.startswith("MemAvailable"):
                        self.avail = min(self.avail, int(l.split()[1]))
            except OSError:
                pass
            time.sleep(0.25)


def tokenize(port: int, text: str) -> list:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="k/n: this box takes every n-th prompt of a fixed shuffle")
    ap.add_argument("--only-arms", default="", help="comma list: run a subset of arms.json")
    ap.add_argument("--only-tasks", default="")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gen", type=int, default=1 << 30, help="cap on each prompt's own gen")
    ap.add_argument("--temperature", type=float, default=0, help="for prompts without their own; the server's other sampler defaults apply")
    a = ap.parse_args()
    global MAXGEN, DEFTEMP
    MAXGEN = a.max_gen
    DEFTEMP = a.temperature

    arms = json.load(open(a.arms))
    if a.only_arms:
        arms = {k: v for k, v in arms.items() if k in a.only_arms.split(",")}
    prompts = [json.loads(l) for l in open(a.prompts)]
    if a.only_tasks:
        prompts = [p for p in prompts if p["task"] in a.only_tasks.split(",")]
    random.Random(2026).shuffle(prompts)                     # same order on every box -> shards mix tasks
    k, n = map(int, a.shard.split("/"))
    prompts = prompts[k::n]

    outdir = os.path.dirname(os.path.abspath(a.out))
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    for arm in arms:
        os.makedirs(os.path.join(outdir, "outputs", arm), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in read_rows(a.out):
            if "error" not in r:
                done.add((r["prompt_id"], r["arm"], r["rep"]))
    host = socket.gethostname()
    rng = random.Random(f"{a.seed}-{a.rep}-{a.shard}")
    blocks = [prompts[i:i + a.block] for i in range(0, len(prompts), a.block)]
    print(f"# {len(prompts)} prompts, {len(blocks)} blocks, arms {list(arms)}, {len(done)} rows already done",
          file=sys.stderr, flush=True)

    with open(a.out, "a") as out:
        for bi, block in enumerate(blocks):
            order = list(arms)
            rng.shuffle(order)
            for pos, arm in enumerate(order):
                todo = [p for p in block if (p["id"], arm, a.rep) not in done]
                if not todo:
                    continue
                port, proc, log = start(arm, arms[arm], logdir, f"r{a.rep}b{bi}")
                try:
                    post(port, {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
                                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                    for p in todo:
                        f0 = majfaults()
                        smp = Sampler(proc.pid)
                        smp.start()
                        try:
                            r, text, wall = request(port, p, a.gen)
                        except Exception as e:  # e.g. prompt over the context: record it, keep going
                            err = getattr(e, "read", lambda: b"")()[:500].decode(errors="replace")
                            out.write(json.dumps({"prompt_id": p["id"], "task": p["task"], "arm": arm, "rep": a.rep,
                                                  "error": f"{e} {err}"}) + "\n")
                            out.flush()
                            print(f"b{bi} {arm:>12} {p['id']:<32} ERROR {e}", file=sys.stderr, flush=True)
                            if proc.poll() is not None:
                                stop(proc)  # the exited launcher may still have live server children
                                port, proc, log = start(arm, arms[arm], logdir, f"r{a.rep}b{bi}x")
                            continue
                        finally:
                            smp.halt = True
                            smp.join()
                        msg = r["choices"][0]["message"]
                        reasoning = msg.get("reasoning_content") or ""
                        try:
                            toks = tokenize(port, text)
                        except Exception:
                            toks = None
                        faults = majfaults() - f0
                        t = r.get("timings", {})
                        with open(os.path.join(outdir, "outputs", arm, f"{p['id']}.r{a.rep}.txt"), "w") as f:
                            f.write(text)
                        row = {
                            "prompt_id": p["id"], "task": p["task"], "arm": arm, "rep": a.rep, "gen": a.gen, "block": bi,
                            "arm_pos": pos, "host": host, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                            "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                            "decode_tok_s": t.get("predicted_per_second"), "prefill_tok_s": t.get("prompt_per_second"),
                            "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
                            "finish": r["choices"][0].get("finish_reason"), "wall_s": round(wall, 3), "majfaults": faults,
                            "sha256": hashlib.sha256(text.encode()).hexdigest(), "timings": t, "out_ids": toks,
                            "rss_anon_peak_kb": smp.anon, "mem_avail_min_kb": smp.avail,
                            "reasoning_chars": len(reasoning), "think_in_content": "<think>" in text or "</think>" in text,
                            "lang": p.get("lang"), "temperature": p.get("temperature", DEFTEMP), "seed": p.get("seed", 0),
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(f"b{bi} {arm:>12} {p['id']:<32} dec {row['decode_tok_s'] or 0:6.2f} tok/s "
                              f"wall {wall:6.1f}s draft {row['draft_accepted']}/{row['draft_n']}",
                              file=sys.stderr, flush=True)
                finally:
                    stop(proc)
    print("# BENCH_DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
