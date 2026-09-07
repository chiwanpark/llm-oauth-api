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
- pattern-based credential redaction on the request path, scoped per provider, model, or group

## Install

```bash
pnpm install
pnpm build
```

## Authenticate a provider

Credentials are stored in a YAML file you choose.

```bash
pnpm loa login anthropic --auth-file ./auth.yaml
pnpm loa login cerebras --auth-file ./auth.yaml
pnpm loa login github-copilot --auth-file ./auth.yaml
pnpm loa login google --auth-file ./auth.yaml
pnpm loa login nvidia --auth-file ./auth.yaml
pnpm loa login openai-codex --auth-file ./auth.yaml
pnpm loa login opencode-go --auth-file ./auth.yaml
pnpm loa login openrouter --auth-file ./auth.yaml
```

For Google/Gemini, this stores a Gemini API key. You can also provide it with the `GEMINI_API_KEY` environment variable.
For NVIDIA NIM, this stores an NVIDIA API key. You can also provide it with the `NVIDIA_API_KEY` environment variable.
For Cerebras, this stores a Cerebras API key. You can also provide it with the `CEREBRAS_API_KEY` environment variable.
For OpenRouter, the default login runs an OAuth flow that mints a durable key on your account. You
can also provide an existing key with the `OPENROUTER_API_KEY` environment variable.

Providers that support both flows default to OAuth. Use `--api-key` to store a key instead:

```bash
pnpm loa login openrouter --auth-file ./auth.yaml --api-key
```

List supported providers:

```bash
pnpm loa providers
```

Remove stored credentials:

```bash
pnpm loa logout anthropic --auth-file ./auth.yaml
```

## Run the server

```bash
export LLM_OAUTH_API_KEY=your-shared-api-key
pnpm loa serve --auth-file ./auth.yaml --port 3000
```

Optional provider filtering:

```bash
pnpm loa serve \
  --auth-file ./auth.yaml \
  --providers anthropic,cerebras,github-copilot,google,nvidia,openai-codex,openrouter
```

While the server is running, it checks stored OAuth credentials for enabled providers every 60
seconds and refreshes credentials that expire within 300 seconds. Refreshed credentials are written
back to the auth file. Change the timing with seconds-based options:

```bash
pnpm loa serve \
  --auth-file ./auth.yaml \
  --oauth-refresh-interval 30 \
  --oauth-refresh-before-expiry 120
```

Set `--oauth-refresh-before-expiry 0` to refresh only credentials that are already expired. Use
`--no-oauth-auto-refresh` to disable the background scheduler. API-key credentials are never
refreshed. A failed provider refresh is logged without token values and retried on a later check;
request-time refresh remains available as a fallback.

## Model groups

A group is a virtual model backed by an ordered list of real models. Because providers give the
same model different names, every member names its own model explicitly. Declare groups in a YAML
file that maps each group name to its members, and point the server at it with `--groups-file`:

```yaml
free:
  - github-copilot:gpt-5.4-mini
  - openai-codex:gpt-5-mini
```

```bash
pnpm loa serve --auth-file ./auth.yaml --groups-file ./groups.yaml
```

Without `--groups-file` the server exposes no groups. The file is read once at startup, and any
problem in it — an unknown provider, a provider left out of `--providers`, a reference to a group
that does not exist — stops the server with an error naming the file.

The group name is itself the model id. Requesting `free` tries `github-copilot:gpt-5.4-mini` first
and falls back to `openai-codex:gpt-5-mini` if that attempt fails:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $LLM_OAUTH_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"free","messages":[{"role":"user","content":"hi"}]}'
```

Add one key per virtual model to expose several of them. A member written without a `:` names
another group, which is spliced into its parent at that position:

```yaml
fast:
  - github-copilot:gpt-5-mini
  - openai-codex:gpt-5.4-mini
smart:
  - anthropic:claude-sonnet-4-5
  - opencode-go:claude-sonnet-4-5
all:
  - fast
  - google:gemini-2.5-pro
```

Here `all` tries `github-copilot:gpt-5-mini`, then `openai-codex:gpt-5.4-mini`, then
`google:gemini-2.5-pro`, and `fast` remains requestable on its own. Nesting may go any number of
levels deep and groups may be declared in any order; a cycle is rejected at startup.

A group name is also the model id clients request, so it is read case-insensitively and `_` reads
as `-`: a key written `FAST_TIER` is requested as `fast-tier`. A name that matches a provider, such
as `google`, is rejected.

### Cooldown for failing models

Rate limits, exhausted quotas, and provider outages last longer than one request, so a member that
fails is remembered: for the next 5 minutes the group passes over it and starts at the following
member instead of paying for the same failure again. Responding successfully clears the record, and
so does the window expiring. Change or disable the window with seconds:

```bash
pnpm loa serve --auth-file ./auth.yaml --model-cooldown 60
pnpm loa serve --auth-file ./auth.yaml --model-cooldown 0   # always try every member
```

## Redacting credentials

Agents paste secrets into a conversation without meaning to: a tool result holding `printenv`, a config file read into context, an `Authorization` header echoed back into a tool call. Redaction masks those substrings before the request leaves for the provider. Declare the patterns in a YAML file and point the server at it with `--redaction-file`:

```yaml
models:
  - anthropic
  - 'openai-codex:gpt-5*'
  - free
replacement: '<redacted:{name}>'
rules:
  - name: openai-key
    pattern: 'sk-[A-Za-z0-9]{16,}'
  - name: aws-access-key-id
    pattern: 'AKIA[0-9A-Z]{16}'
    replacement: '<aws-key>'
```

```bash
pnpm loa serve --auth-file ./auth.yaml --redaction-file ./redaction.yaml
```

Without `--redaction-file` nothing is masked. The file is read once at startup, and any problem in it — an invalid regular expression, an unknown provider, a rule name used twice — stops the server with an error naming the file.

`models` is the scope for the whole file: every rule applies to exactly those models, and nothing is masked for anything else. It is optional, and leaving it out masks on every model. Scope is a property of your deployment rather than of any one pattern — which providers you distrust does not change from one credential shape to the next — so it is stated once instead of on every rule. A rule that tries to set its own `models` is rejected at startup.

`replacement` is the mask every rule uses unless it names its own. It is optional and defaults to `[REDACTED:{name}]`. In both places `{name}` expands to the rule name, so a file-wide mask can still say which pattern fired; write a plain literal such as `<redacted>` to mask everything identically instead. An empty string deletes the match rather than standing in for it.

In the example above `openai-key` masks to `<redacted:openai-key>` from the file-wide setting, while `aws-access-key-id` overrides it and masks to `<aws-key>`.

Each rule takes these fields:

- `name` (required) — letters, digits, and hyphens; unique within the file, and the label used in logs.
- `pattern` (required) — a JavaScript regular expression. A pattern that can match the empty string is rejected, because it would rewrite every message instead of masking anything.
- `flags` (optional) — any of `i`, `m`, `s`, `u`, `v`. `g` is always applied, so every occurrence is masked, and `y` is rejected because it would make matching stateful.
- `replacement` (optional) — overrides the file-wide mask for this rule.
- `captureGroup` (optional) — mask only this capture group of the match instead of the whole match. See [Masking part of a match](#masking-part-of-a-match).

Rules apply in declaration order, so an earlier replacement is visible to a later pattern.

If two sets of patterns really do need different scopes, run them as what they are: separate concerns. Widen the file to cover both and write the narrower patterns so they only match what they should.

### Masking part of a match

Credentials are often recognised by what sits around them rather than by the secret itself: `FOO_API_KEY=`, the colon in a JDBC userinfo, a `password=` query parameter. A pattern can name that context, but by default the whole match is replaced, and the context goes with it:

```yaml
- name: env-secret
  pattern: '[A-Z0-9_]*_API_KEY\s*=\s*\S+'
```

```
export MY_APP_API_KEY=abc123secret  ->  export [REDACTED:env-secret]
```

Losing the variable name usually costs the model the ability to reason about the config at all. Set `captureGroup` to the group holding the secret and the surrounding text survives:

```yaml
rules:
  - name: env-secret
    pattern: '([A-Z0-9_]*(?:API_KEY|PASSWORD|TOKEN|SECRET)\s*=\s*)(\S+)'
    captureGroup: 2
  - name: jdbc-userinfo
    pattern: '(jdbc:[a-z]+://[^:/@\s]+:)([^@\s]+)(@)'
    captureGroup: 2
  - name: jdbc-password
    pattern: '([?&]password=)([^&\s]+)'
    captureGroup: 2
```

```
export MY_APP_API_KEY=abc123secret                   ->  export MY_APP_API_KEY=[REDACTED:env-secret]
DB_PASSWORD=p@ss STRIPE_API_KEY=sk_live_9999         ->  DB_PASSWORD=[REDACTED:env-secret] STRIPE_API_KEY=[REDACTED:env-secret]
jdbc:postgresql://dbuser:s3cr3tpw@db.host:5432/app   ->  jdbc:postgresql://dbuser:[REDACTED:jdbc-userinfo]@db.host:5432/app
jdbc:mysql://db/app?user=root&password=hunter2&ssl=true  ->  jdbc:mysql://db/app?user=root&password=[REDACTED:jdbc-password]&ssl=true
```

Groups are counted left to right by their opening parenthesis, starting at 1; `(?:...)` does not count. A `captureGroup` the pattern does not have is rejected at startup. If the group matches nothing on a given occurrence — an optional group that did not participate — that occurrence is left alone.

Capture references such as `$1` in a `replacement` are **not** expanded; they would be written out literally, so a replacement containing one is rejected at startup rather than silently producing `$1` in the transcript. `captureGroup` is the supported way to keep part of a match.

Lookbehind is an alternative for simple cases, since the proxy imposes no restriction on it: `(?<=[A-Z_]*_API_KEY=)\S+` masks the same value without any group. `captureGroup` is usually easier to read, and works where a variable-length lookbehind would be awkward.

### Choosing the models

An entry in `models` follows the convention of the groups file: an entry with a `:` names a model, and an entry without one names a provider or a group. `*` is a wildcard in either half.

| Entry                         | Matches                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| `*`                           | every model                                                 |
| `anthropic`                   | every model from that provider                              |
| `anthropic:claude-sonnet-4-5` | that one model                                              |
| `openai-codex:gpt-5*`         | models matching the glob within that provider               |
| `*:claude-sonnet-4-5`         | that model id from any provider that publishes it           |
| `free`                        | requests for the group `free`, whichever member serves them |

Listing several entries is a union: the scope matches if any of them matches. A group entry is matched against the model id the client asked for, so a scope written for `free` stays in force as the group falls back from one member to the next. Group names are read case-insensitively with `_` as `-`, exactly as they are in the groups file.

The scope is evaluated per attempt rather than once per request, so a group that falls back to a provider outside it stops masking, and one that falls back into it starts.

### What is and is not redacted

Redaction runs on the request path only, over everything the proxy is about to send upstream:

- the system prompt, including instructions merged from `system` and `developer` messages
- user turns, both plain strings and the text blocks of multi-part content
- assistant turns replayed as history, and the arguments of the tool calls inside them
- tool results
- tool descriptions

Assistant history is included on purpose. A model that quoted a key back in an earlier turn would otherwise re-send it in clear text on the next request, and a client is free to put anything it likes in an assistant turn.

These are left untouched:

- **the response stream.** Output flowing back to your client is not masked. If a credential appears there it already left your machine, so masking it protects nothing while breaking token counts and forcing every SSE chunk to be buffered.
- **reasoning content and thinking signatures.** The readable half is paired with an encrypted blob the provider validates, and rewriting one half invalidates the pair.
- **image bytes.** A credential pattern says nothing about base64 pixels, and a replacement inside them corrupts the image.
- **tool call ids.** They pair a call with its result; rewriting one breaks the conversation.
- **tool parameter schemas.** Providers validate them and build constrained sampling from them.

When a rule matches, the server logs the rule name and how many substrings it replaced. The matched text is never logged — writing the secret into a log line would move the leak rather than close it.

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
