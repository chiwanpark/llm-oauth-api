# llm-oauth-api

OpenAI-compatible HTTP API backed by `@earendil-works/pi-ai`.

## Disclaimer

> [!CAUTION]
> This is an unofficial community project and is not affiliated with, endorsed by, or sponsored by any LLM provider.
>
> Exposing provider-backed LLM access through a separate OpenAI-compatible API, proxy, or hosted service may be restricted or prohibited by provider terms, acceptable-use policies, or account agreements.
>
> Use this project only for personal, local experimentation. Do not offer it as a hosted service, share access with others, pool accounts, resell access, or redistribute provider access in any form.
>
> You are solely responsible for understanding and complying with each provider's rules and all applicable laws. Misuse may result in rate limits, billing charges, account suspension, termination, or other enforcement actions.
>
> This software is provided as-is, without warranties. You assume all legal, operational, financial, and account-related risks.

## Supported providers

- `anthropic`
- `cerebras` (API-key based)
- `github-copilot`
- `google` (API-key based)
- `nvidia` (API-key based)
- `openai-codex`
- `opencode-go` (API-key based)
- `openrouter` (OAuth or API-key based)

## Features

- Svelte API playground at `GET /`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- streaming SSE responses
- image input
- tool calls
- encrypted reasoning content for multi-turn reasoning continuity
- automatic OAuth credential refresh
- model groups with automatic fallback and cooldown for failing models
- shared API key protection via `LLM_OAUTH_API_KEY`

## Install

```bash
pnpm install
pnpm build
```

## Authenticate a provider

Credentials are stored in a JSON file you choose.

```bash
pnpm loa login anthropic --auth-file ./auth.json
pnpm loa login cerebras --auth-file ./auth.json
pnpm loa login github-copilot --auth-file ./auth.json
pnpm loa login google --auth-file ./auth.json
pnpm loa login nvidia --auth-file ./auth.json
pnpm loa login openai-codex --auth-file ./auth.json
pnpm loa login opencode-go --auth-file ./auth.json
pnpm loa login openrouter --auth-file ./auth.json
```

For Google/Gemini, this stores a Gemini API key. You can also provide it with the `GEMINI_API_KEY` environment variable.
For NVIDIA NIM, this stores an NVIDIA API key. You can also provide it with the `NVIDIA_API_KEY` environment variable.
For Cerebras, this stores a Cerebras API key. You can also provide it with the `CEREBRAS_API_KEY` environment variable.
For OpenRouter, the default login runs an OAuth flow that mints a durable key on your account. You
can also provide an existing key with the `OPENROUTER_API_KEY` environment variable.

Providers that support both flows default to OAuth. Use `--api-key` to store a key instead:

```bash
pnpm loa login openrouter --auth-file ./auth.json --api-key
```

List supported providers:

```bash
pnpm loa providers
```

Remove stored credentials:

```bash
pnpm loa logout anthropic --auth-file ./auth.json
```

## Run the server

```bash
export LLM_OAUTH_API_KEY=your-shared-api-key
pnpm loa serve --auth-file ./auth.json --port 3000
```

Optional provider filtering:

```bash
pnpm loa serve \
  --auth-file ./auth.json \
  --providers anthropic,cerebras,github-copilot,google,nvidia,openai-codex,openrouter
```

While the server is running, it checks stored OAuth credentials for enabled providers every 60
seconds and refreshes credentials that expire within 300 seconds. Refreshed credentials are written
back to the auth file. Change the timing with seconds-based options:

```bash
pnpm loa serve \
  --auth-file ./auth.json \
  --oauth-refresh-interval 30 \
  --oauth-refresh-before-expiry 120
```

Set `--oauth-refresh-before-expiry 0` to refresh only credentials that are already expired. Use
`--no-oauth-auto-refresh` to disable the background scheduler. API-key credentials are never
refreshed. A failed provider refresh is logged without token values and retried on a later check;
request-time refresh remains available as a fallback.

## Model groups

A group is a virtual model backed by an ordered list of real models. Because providers give the
same model different names, every member names its own model explicitly. Declare a group with a
`LLM_OAUTH_GROUP_<NAME>` environment variable:

```bash
export LLM_OAUTH_GROUP_FREE=github-copilot:gpt-5.4-mini,openai-codex:gpt-5-mini
pnpm loa serve --auth-file ./auth.json
```

The group name is itself the model id. Requesting `free` tries `github-copilot:gpt-5.4-mini` first
and falls back to `openai-codex:gpt-5-mini` if that attempt fails:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"free","messages":[{"role":"user","content":"hi"}]}'
```

Define one variable per virtual model to expose several of them:

```bash
export LLM_OAUTH_GROUP_FAST=github-copilot:gpt-5.4-mini,openai-codex:gpt-5-mini
export LLM_OAUTH_GROUP_SMART=anthropic:claude-sonnet-4-5,opencode-go:claude-sonnet-4-5
```

A member written without a `:` names another group, which is spliced into its parent at that
position:

```bash
export LLM_OAUTH_GROUP_FAST=github-copilot:gpt-5-mini,openai-codex:gpt-5.4-mini
export LLM_OAUTH_GROUP_ALL=fast,google:gemini-2.5-pro
```

Here `all` tries `github-copilot:gpt-5-mini`, then `openai-codex:gpt-5.4-mini`, then
`google:gemini-2.5-pro`, and `fast` remains requestable on its own.

### Cooldown for failing models

Rate limits, exhausted quotas, and provider outages last longer than one request, so a member that
fails is remembered: for the next 5 minutes the group passes over it and starts at the following
member instead of paying for the same failure again. Responding successfully clears the record, and
so does the window expiring. Change or disable the window with seconds:

```bash
pnpm loa serve --auth-file ./auth.json --model-cooldown 60
pnpm loa serve --auth-file ./auth.json --model-cooldown 0   # always try every member
```

## Calling the API

Use the shared API key as a bearer token:

```bash
curl http://localhost:3000/v1/models \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY"
```

Models are exposed as `provider:model`, for example:

- `anthropic:claude-sonnet-4-5`
- `cerebras:gpt-oss-120b`
- `github-copilot:gpt-5`
- `google:gemini-2.5-pro`
- `nvidia:meta/llama-3.3-70b-instruct`
- `openai-codex:gpt-5.4`
- `opencode-go:claude-sonnet-4-5`
- `openrouter:anthropic/claude-sonnet-4.5`

OpenRouter model ids contain their own `/` and sometimes a `:` variant suffix. Only the first `:`
separates the provider, so `openrouter:deepseek/deepseek-r1:free` resolves as expected.

Configured groups add a bare virtual model id, such as `free`. See [Model groups](#model-groups).

Example chat completion:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-codex:gpt-5.4",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "reasoning_effort": "medium"
  }'
```

`reasoning_effort` accepts `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`.
Support for individual levels depends on the selected model and provider.

Example streamed responses API call:

```bash
curl http://localhost:3000/v1/responses \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-codex:gpt-5.4",
    "input": "Write a haiku about OAuth",
    "reasoning": {"effort": "high"},
    "stream": true
  }'
```

For `/v1/responses`, use the Responses API form `reasoning.effort`. It accepts the
same effort levels listed above.

## Notes

- `/v1/responses` is implemented as a practical compatibility layer, not a byte-for-byte clone of OpenAI.
- `/v1/models` only lists providers that appear configured from the auth file/environment.
- Remote image URLs are fetched server-side and converted to base64 before forwarding to `pi-ai`.
