# Using `openset-llm` with Open WebUI and LiteLLM Proxy

Start `openset-llm` in API-only mode (add `--multi 4` when serving multiple team members concurrently):

```sh
./serve --no-webui --host 127.0.0.1 --port 8080
# or for up to 4 concurrent streams sharing weight passes & system prompt:
./serve --no-webui --host 127.0.0.1 --port 8080 --multi 4
```

---

## 1. Open WebUI (Multi-User Team Chat)

Point Open WebUI's OpenAI backend straight at your local `openset-llm` instance:

```sh
docker run -d --name open-webui \
  --network=host \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
  -e OPENAI_API_KEY=unused \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

In the Open WebUI model selector, pick **`openset-llm`** — streaming replies work immediately with zero external API keys.

---

## 2. LiteLLM Proxy (Team API Keys, Routing & Spend Controls)

Create `litellm_config.yaml`:

```yaml
model_list:
  - model_name: openset-27b
    litellm_params:
      model: openai/openset-llm
      api_base: http://127.0.0.1:8080/v1
      api_key: unused
      timeout: 300

litellm_settings:
  drop_params: true
```

Run LiteLLM Proxy:

```sh
litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000
```

Any OpenAI-compatible client can now call `http://127.0.0.1:4000/v1` with `model="openset-27b"`, inheriting LiteLLM virtual keys, rate limits, and audit logs on top of your CPU-only `openset-llm` backend.
