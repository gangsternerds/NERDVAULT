# Connecting Local Coding Agents to openset-llm

`openset-llm` serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` with agent-loop prefix resume enabled by default (`$OPENSET_LLM_HOME/state`), keeping multi-step agent turns fast with zero per-token API fees.

## Recommended server command for coding agents

For 8 GB machines running 12k–16k agent contexts with speculative decoding still active:

```sh
./serve --no-webui --kv6 --ctx 12288
# or up to 16k context on 8 GB:
./serve --no-webui --kv5 --ctx 16384
```

For machines with 16+ GB RAM:

```sh
./serve --no-webui --ctx 16384
```

---

## 1. Aider

```sh
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=unused
aider --model openai/openset-llm
```

---

## 2. Continue (VS Code / JetBrains)

Add to `~/.continue/config.yaml`:

```yaml
models:
  - name: openset-llm (local 27B CPU)
    provider: openai
    model: openset-llm
    apiBase: http://127.0.0.1:8080/v1
    apiKey: unused
    roles:
      - chat
      - edit
      - apply
```

---

## 3. Cline (VS Code)

In **Cline Settings → API Provider**:

- **API Provider:** `OpenAI Compatible`
- **Base URL:** `http://127.0.0.1:8080/v1`
- **API Key:** `unused`
- **Model ID:** `openset-llm`

---

## 4. OpenCode

Add a custom local provider in `opencode.json`:

```json
{
  "provider": {
    "openset-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "openset-llm (local CPU)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "unused"
      },
      "models": {
        "openset-llm": {
          "name": "openset-llm (Bonsai-2-27B)"
        }
      }
    }
  }
}
