# Flatline Proxy

Flatline presents one OpenAI Responses API endpoint to Codex and routes each logical model to the cheapest eligible upstream. It streams upstream responses without buffering and keeps API keys out of its configuration file.

## Run

```sh
cp flatline.example.json flatline.json
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export ZAI_API_KEY=...
export OPENROUTER_API_KEY=...
cargo run --release
```

Open <http://127.0.0.1:8080> to edit routing configuration and inspect recent requests. Point Codex's OpenAI-compatible base URL at `http://127.0.0.1:8080/v1` and use a logical model name from `models` (for example `deepseek`). API-provider credentials come from the environment variables named in the provider configuration.

For ChatGPT-subscription routing, merge `codex.flatline.toml` into `~/.codex/config.toml` and select `model_provider = "flatline"`. Its `requires_openai_auth = true` setting makes Codex supply its current, automatically refreshed ChatGPT bearer token and account ID to the local proxy. Flatline forwards those credentials only to providers configured with `auth = "incoming_open_ai"`; it never stores them. The starter policy tries the ChatGPT subscription first and falls back to `OPENAI_API_KEY` on pre-stream quota, rate-limit, authentication, timeout, or server failures.

## Routing behavior

For every route whose provider is enabled and whose key environment variable exists, Flatline estimates request cost from approximate input tokens, `max_output_tokens`, and the configured rates. A recently successful request with the same stable prompt prefix is treated as cache-resident for that provider, causing `cached_input_cost_per_million` to apply. Lowest estimated price wins; `priority` breaks price ties. Retryable pre-stream failures fall through to the next route.

OpenAI, OpenRouter, and xAI use their native Responses endpoints. DeepSeek and Z.AI are translated through Chat Completions; Anthropic is translated through Messages. The starter configuration constrains OpenRouter to `deepseek/*` and `z-ai/*` upstream model names.

Cache residency is currently a local routing hint, not a guarantee from the upstream. Usage history and cache hints are in memory and reset on restart. Translated adapters currently target Codex's streaming Responses usage; general non-streaming compatibility will come later.

## API

- `POST /v1/responses` — OpenAI Responses-compatible transparent forwarding
- `GET /v1/models` — Codex-compatible model-catalog probe (configured aliases use local metadata)
- `GET/PUT /api/config` — configuration management
- `GET /api/usage` — last 1,000 routing decisions
- `GET /health` — liveness

## Backlog

- Configurable system/developer-prompt rewrites. Support arbitrary ordered rules with matching by logical model, resolved provider, and upstream model, plus replace, prepend, and append operations. Rewrites should be visible in request diagnostics and must preserve the original prompt for debugging.
