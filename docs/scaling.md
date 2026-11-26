# Scaling `openset-llm` Across Multi-Socket / High-Core CPU Servers

CPU LLM decoding is primarily memory-bandwidth bound. On machines with **16–64+ cores and 2+ NUMA nodes** (for example dual-socket Intel Xeon Ice Lake / Sapphire Rapids or multi-CCD AMD EPYC Zen 4/5 servers), running **one `./serve` process per NUMA node** avoids cross-socket memory traffic and scales aggregate throughput cleanly.

## 1. Check your NUMA topology

```sh
lscpu | grep -E "Model name|Socket|NUMA|Core"
numactl --hardware
```

Look for how many NUMA nodes exist (`node 0`, `node 1`, ...) and ensure each replica has at least **8–10 GB of local RAM** and **8 physical cores** on its assigned node.

## 2. Launch one pinned worker per NUMA node

All workers can read the same prepared weights from `$OPENSET_LLM_HOME/models/model.gguf`:

```sh
# Worker 0 bound strictly to NUMA node 0
numactl --cpunodebind=0 --membind=0 \
  ./serve --no-webui --host 127.0.0.1 --port 8081 --threads 8 &

# Worker 1 bound strictly to NUMA node 1
numactl --cpunodebind=1 --membind=1 \
  ./serve --no-webui --host 127.0.0.1 --port 8082 --threads 8 &
```

For concurrent batch extraction workloads, add `--multi 4` to each pinned worker (streams share each pass over the weights and reuse shared system prompts from disk):

```sh
numactl --cpunodebind=0 --membind=0 \
  ./serve --no-webui --host 127.0.0.1 --port 8081 --threads 8 --multi 4 &
```

## 3. Balance requests across workers (Caddy example)

```caddyfile
:8080 {
	reverse_proxy 127.0.0.1:8081 127.0.0.1:8082 {
		lb_policy least_conn
		health_uri /health
		flush_interval -1
	}
}
```

Use `least_conn` rather than round-robin so long generations never pile up behind one worker while another sits idle.
