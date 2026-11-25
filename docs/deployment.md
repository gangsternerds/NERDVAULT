# Running openset-llm in WSL2 and with systemd

## Windows (WSL2): make sure WSL2 has at least 8 GB of RAM

By default, Windows allocates half of host RAM to WSL2. On a 16 GB laptop, setting an explicit `10GB` limit gives `openset-llm` full 8 GB headroom right from the first launch.

Create or edit `C:\Users\<YourUser>\.wslconfig` on Windows:

```ini
[wsl2]
memory=10GB
swap=4GB
```

Then restart WSL2 from PowerShell before running `./start`:

```powershell
wsl --shutdown
```

Inside WSL2, verify available memory and AVX-512 support:

```sh
free -h
./install.sh --check
```

---

## Linux: run `./serve` in the background with a `systemd` user service

After running `./install.sh` once, you can keep the OpenAI-compatible API running on `http://127.0.0.1:8080/v1` without leaving a terminal open.

Create `~/.config/systemd/user/openset-llm.service` (replace `/path/to/openset-llm` with your clone path):

```ini
[Unit]
Description=openset-llm local OpenAI-compatible server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/openset-llm
ExecStart=/path/to/openset-llm/serve --no-webui --host 127.0.0.1 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable and start it:

```sh
systemctl --user daemon-reload
systemctl --user enable --now openset-llm
systemctl --user status openset-llm
```

Check health and logs:

```sh
curl -fsS http://127.0.0.1:8080/health
journalctl --user -u openset-llm -f
```
