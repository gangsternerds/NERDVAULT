# Putting an Authenticating Reverse Proxy in Front of `openset-llm`

As noted in `SECURITY.md`, `./serve` listens on `127.0.0.1:8080` with no built-in authentication or rate limiting. When sharing a CPU inference box across a team or internal network, keep `./serve` bound to loopback and place **Caddy** or **Nginx** in front.

Start the backend on loopback only:

```sh
./serve --no-webui --host 127.0.0.1 --port 8080
```

---

## Option 1: Caddy (Automatic HTTPS + Bearer Token + Streaming)

Replace `llm.example.com` and set `OPENSET_API_KEY` in Caddy's environment:

```caddyfile
llm.example.com {
	@unauthorized not header Authorization "Bearer {$OPENSET_API_KEY}"
	respond @unauthorized "Unauthorized" 401

	reverse_proxy 127.0.0.1:8080 {
		flush_interval -1
	}
}
```

Clients then connect normally with the standard OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="https://llm.example.com/v1", api_key="your-secret-key")
```

---

## Option 2: Nginx (Bearer Token + Rate Limiting + Unbuffered SSE Streaming)

```nginx
limit_req_zone $binary_remote_addr zone=openset_limit:10m rate=10r/m;

server {
    listen 443 ssl http2;
    server_name llm.example.com;

    # Add your TLS certificate directives here

    location /v1/ {
        if ($http_authorization != "Bearer REPLACE_WITH_LONG_RANDOM_TOKEN") {
            return 401;
        }

        limit_req zone=openset_limit burst=5 nodelay;

        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;

        # Required for smooth token-by-token OpenAI SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    location = /health {
        proxy_pass http://127.0.0.1:8080/health;
    }
}
