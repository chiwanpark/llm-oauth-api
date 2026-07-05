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

- `anthropic` (`claude-code`, `claude` aliases)
- `github-copilot` (`copilot` alias)
- `google` (`gemini`, `google-ai` aliases, API-key based)
- `nvidia` (`nim`, `nvidia-nim` aliases, API-key based)
- `openai-codex` (`codex` alias)
- `opencode-go` (`opencode` alias, API-key based)

## Features

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- streaming SSE responses
- image input
- tool calls
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
pnpm loa login github-copilot --auth-file ./auth.json
pnpm loa login google --auth-file ./auth.json
pnpm loa login nvidia --auth-file ./auth.json
pnpm loa login openai-codex --auth-file ./auth.json
pnpm loa login opencode-go --auth-file ./auth.json
```

For Google/Gemini, this stores a Gemini API key. You can also provide it with the `GEMINI_API_KEY` environment variable.
For NVIDIA NIM, this stores an NVIDIA API key. You can also provide it with the `NVIDIA_API_KEY` environment variable.

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
  --providers anthropic,github-copilot,google,nvidia,openai-codex
```

## Calling the API

Use the shared API key as a bearer token:

```bash
curl http://localhost:3000/v1/models \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY"
```

Models are exposed as `provider:model`, for example:

- `anthropic:claude-sonnet-4-5`
- `github-copilot:gpt-5`
- `google:gemini-2.5-pro`
- `nvidia:meta/llama-3.3-70b-instruct`
- `openai-codex:gpt-5.4`
- `opencode-go:claude-sonnet-4-5`

Example chat completion:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic:claude-sonnet-4-5",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Example streamed responses API call:

```bash
curl http://localhost:3000/v1/responses \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-codex:gpt-5.4",
    "input": "Write a haiku about OAuth",
    "stream": true
  }'
```

## Notes

- `/v1/responses` is implemented as a practical compatibility layer, not a byte-for-byte clone of OpenAI.
- `/v1/models` only lists providers that appear configured from the auth file/environment.
- Remote image URLs are fetched server-side and converted to base64 before forwarding to `pi-ai`.
