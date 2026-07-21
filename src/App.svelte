<script lang="ts">
  import {
    Button,
    Checkbox,
    InlineLoading,
    InlineNotification,
    NumberInput,
    PasswordInput,
    Select,
    SelectItem,
    TextArea,
    Toggle,
  } from 'carbon-components-svelte';
  import Checkmark from 'carbon-icons-svelte/lib/Checkmark.svelte';
  import Copy from 'carbon-icons-svelte/lib/Copy.svelte';
  import Renew from 'carbon-icons-svelte/lib/Renew.svelte';
  import Send from 'carbon-icons-svelte/lib/Send.svelte';
  import StopFilled from 'carbon-icons-svelte/lib/StopFilled.svelte';
  import Terminal from 'carbon-icons-svelte/lib/Terminal.svelte';

  type Endpoint = '/chat/completions' | '/responses';
  type ReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';
  type RequestState = 'idle' | 'running' | 'complete' | 'error';

  type ModelInfo = {
    id: string;
    owned_by?: string;
  };

  type UsageSummary = {
    input: number;
    output: number;
    total: number;
  };

  type RequestSettings = {
    endpoint: Endpoint;
    model: string;
    prompt: string;
    includeTemperature: boolean;
    temperature: number | null;
    includeMaxTokens: boolean;
    maxTokens: number | null;
    includeReasoning: boolean;
    reasoningEffort: ReasoningEffort;
    stream: boolean;
  };

  const reasoningEfforts: ReasoningEffort[] = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'];

  let apiKey = '';
  let endpoint: Endpoint = '/chat/completions';
  let models: ModelInfo[] = [];
  let model = '';
  let prompt = '';
  let includeTemperature = false;
  let temperature: number | null = 0.7;
  let includeMaxTokens = false;
  let maxTokens: number | null = 2048;
  let includeReasoning = false;
  let reasoningEffort: ReasoningEffort = 'medium';
  let stream = false;

  let modelsLoading = false;
  let modelsMessage = '';
  let requestState: RequestState = 'idle';
  let requestError = '';
  let displayText = '';
  let rawResponse = '';
  let responseData: unknown = null;
  let responseStatus: number | null = null;
  let duration: number | null = null;
  let copied = false;
  let requestController: AbortController | null = null;
  let requestBody: Record<string, unknown> = {};
  let requestPreview = '';
  let usage: UsageSummary | null = null;

  $: requestBody = createRequestBody({
    endpoint,
    model,
    prompt,
    includeTemperature,
    temperature,
    includeMaxTokens,
    maxTokens,
    includeReasoning,
    reasoningEffort,
    stream,
  });
  $: requestPreview = JSON.stringify(requestBody, null, 2);
  $: usage = getUsage(responseData);

  function createRequestBody(settings: RequestSettings): Record<string, unknown> {
    const generationOptions = {
      ...(settings.includeTemperature && settings.temperature != null
        ? { temperature: settings.temperature }
        : {}),
      ...(settings.includeMaxTokens && settings.maxTokens != null
        ? settings.endpoint === '/responses'
          ? { max_output_tokens: settings.maxTokens }
          : { max_completion_tokens: settings.maxTokens }
        : {}),
      ...(settings.stream ? { stream: true } : {}),
    };

    if (settings.endpoint === '/responses') {
      return {
        model: settings.model,
        input: settings.prompt,
        ...generationOptions,
        ...(settings.includeReasoning ? { reasoning: { effort: settings.reasoningEffort } } : {}),
      };
    }

    return {
      model: settings.model,
      messages: [{ role: 'user', content: settings.prompt }],
      ...generationOptions,
      ...(settings.includeReasoning ? { reasoning_effort: settings.reasoningEffort } : {}),
    };
  }

  async function loadModels(): Promise<void> {
    if (!apiKey.trim()) {
      modelsMessage = 'Enter an API key before loading models.';
      return;
    }

    modelsLoading = true;
    modelsMessage = '';

    try {
      const response = await fetch('/v1/models', {
        headers: authorizationHeaders(),
      });
      const data = await readResponse(response);

      if (!response.ok) {
        throw new Error(errorMessage(data, response.status));
      }

      const available = isRecord(data) && Array.isArray(data.data) ? data.data : [];
      models = available.filter(isModelInfo);

      if (!models.length) {
        model = '';
        modelsMessage = 'No configured models were returned by the API.';
        return;
      }

      if (!models.some((item) => item.id === model)) {
        model = models[0]?.id ?? '';
      }
      modelsMessage = `${models.length} model${models.length === 1 ? '' : 's'} available.`;
    } catch (error) {
      models = [];
      model = '';
      modelsMessage = error instanceof Error ? error.message : String(error);
    } finally {
      modelsLoading = false;
    }
  }

  async function sendRequest(): Promise<void> {
    requestError = validateRequest();
    if (requestError) {
      requestState = 'error';
      return;
    }

    const controller = new AbortController();
    requestController = controller;
    requestState = 'running';
    requestError = '';
    displayText = '';
    rawResponse = '';
    responseData = null;
    responseStatus = null;
    duration = null;
    copied = false;
    const startedAt = performance.now();

    try {
      const response = await fetch(`/v1${endpoint}`, {
        method: 'POST',
        headers: {
          ...authorizationHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });
      responseStatus = response.status;

      if (!response.ok) {
        const data = await readResponse(response);
        rawResponse = formatValue(data);
        responseData = data;
        throw new Error(errorMessage(data, response.status));
      }

      if (stream) {
        await readEventStream(response);
      } else {
        const data = await readResponse(response);
        responseData = data;
        rawResponse = formatValue(data);
        displayText = assistantText(data);
      }

      requestState = 'complete';
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        requestError = 'Request cancelled.';
      } else {
        requestError = error instanceof Error ? error.message : String(error);
      }
      requestState = 'error';
    } finally {
      duration = performance.now() - startedAt;
      if (requestController === controller) requestController = null;
    }
  }

  async function readEventStream(response: Response): Promise<void> {
    if (!response.body) throw new Error('The response did not contain a readable stream.');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const events: unknown[] = [];
    let buffer = '';
    let completedResponse: unknown = null;

    const processFrame = (frame: string) => {
      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n');

      if (!data || data === '[DONE]') return;

      let payload: unknown;
      try {
        payload = JSON.parse(data);
      } catch {
        payload = data;
      }

      events.push(payload);
      rawResponse = JSON.stringify(events, null, 2);

      const delta = endpoint === '/chat/completions' ? chatDelta(payload) : responsesDelta(payload);
      if (delta) displayText += delta;

      if (isRecord(payload) && payload.type === 'response.completed') {
        completedResponse = payload.response;
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? '';
      for (const frame of frames) processFrame(frame);
      if (done) break;
    }

    if (buffer.trim()) processFrame(buffer);
    responseData = completedResponse ?? [...events].reverse().find(hasUsage) ?? events;
    if (!displayText) displayText = assistantText(responseData);
  }

  function abortRequest(): void {
    requestController?.abort();
  }

  async function copyResponse(): Promise<void> {
    const value = displayText || rawResponse;
    if (!value) return;

    await navigator.clipboard.writeText(value);
    copied = true;
    window.setTimeout(() => (copied = false), 1600);
  }

  function authorizationHeaders(): Record<string, string> {
    return { Authorization: `Bearer ${apiKey.trim()}` };
  }

  function validateRequest(): string {
    if (!apiKey.trim()) return 'API key is required.';
    if (!model) return 'Load and select a model.';
    if (!prompt.trim()) return 'Enter a user message.';
    return '';
  }

  async function readResponse(response: Response): Promise<unknown> {
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  function errorMessage(value: unknown, status: number): string {
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') {
      return `${status}: ${value.error.message}`;
    }
    if (typeof value === 'string' && value) return `${status}: ${value}`;
    return `${status}: The API request failed.`;
  }

  function assistantText(value: unknown): string {
    if (!isRecord(value)) return typeof value === 'string' ? value : '';
    if (typeof value.output_text === 'string') return value.output_text;

    if (Array.isArray(value.choices)) {
      const first = value.choices[0];
      if (isRecord(first) && isRecord(first.message) && typeof first.message.content === 'string') {
        return first.message.content;
      }
    }

    if (Array.isArray(value.output)) {
      return value.output
        .flatMap((item) => (isRecord(item) && Array.isArray(item.content) ? item.content : []))
        .map((item) => (isRecord(item) && typeof item.text === 'string' ? item.text : ''))
        .join('');
    }

    return '';
  }

  function chatDelta(value: unknown): string {
    if (!isRecord(value) || !Array.isArray(value.choices)) return '';
    const first = value.choices[0];
    return isRecord(first) && isRecord(first.delta) && typeof first.delta.content === 'string'
      ? first.delta.content
      : '';
  }

  function responsesDelta(value: unknown): string {
    return isRecord(value) &&
      value.type === 'response.output_text.delta' &&
      typeof value.delta === 'string'
      ? value.delta
      : '';
  }

  function getUsage(value: unknown): UsageSummary | null {
    if (!isRecord(value)) return null;
    if (value.type === 'response.completed') return getUsage(value.response);
    if (!isRecord(value.usage)) return null;

    const input = numberValue(value.usage.prompt_tokens ?? value.usage.input_tokens);
    const output = numberValue(value.usage.completion_tokens ?? value.usage.output_tokens);
    const total = numberValue(value.usage.total_tokens);
    return input == null || output == null || total == null ? null : { input, output, total };
  }

  function hasUsage(value: unknown): boolean {
    return getUsage(value) != null;
  }

  function numberValue(value: unknown): number | null {
    return typeof value === 'number' ? value : null;
  }

  function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
  }

  function isModelInfo(value: unknown): value is ModelInfo {
    return isRecord(value) && typeof value.id === 'string';
  }

  function formatValue(value: unknown): string {
    return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  }
</script>

<svelte:head>
  <title>API Playground · LLM OAuth API</title>
</svelte:head>

<header class="app-header">
  <a class="brand" href="/" aria-label="LLM OAuth API playground home">
    <span class="brand-mark"><Terminal size={20} /></span>
    <span class="brand-name"><strong>LLM OAuth</strong> API</span>
  </a>
  <div class="environment">
    <span class="environment-dot"></span>
    Local playground
  </div>
</header>

<main>
  <section class="intro" aria-labelledby="page-title">
    <div>
      <p class="eyebrow">OpenAI-compatible test client</p>
      <h1 id="page-title">API Playground</h1>
      <p class="intro-copy">
        Configure a request, send it to a connected model, and inspect the complete response.
      </p>
    </div>
    <div class="route-badge">
      <span>POST</span>
      <code>/v1{endpoint}</code>
    </div>
  </section>

  <div class="playground-grid">
    <section class="request-panel" aria-labelledby="request-heading">
      <div class="panel-heading">
        <div>
          <p class="section-number">01</p>
          <h2 id="request-heading">Request</h2>
        </div>
        <span class="panel-caption">Configure</span>
      </div>

      <form
        on:submit|preventDefault={() => {
          void sendRequest();
        }}
      >
        <fieldset>
          <legend>Connection</legend>
          <div class="field-stack">
            <PasswordInput
              id="api-key"
              labelText="API key"
              helperText="Kept in this browser tab only."
              placeholder="Enter your shared API key"
              autocomplete="off"
              bind:value={apiKey}
            />
            <div class="model-row">
              <Select
                id="model"
                labelText="Model"
                helperText={models.length
                  ? `${models.length} configured models`
                  : 'Load models to begin'}
                bind:selected={model}
                disabled={modelsLoading || !models.length}
              >
                <SelectItem value="" text="Select a model" hidden />
                {#each models as item}
                  <SelectItem value={item.id} text={item.id} />
                {/each}
              </Select>
              <Button
                kind="tertiary"
                size="field"
                type="button"
                icon={Renew}
                disabled={modelsLoading || !apiKey.trim()}
                on:click={() => void loadModels()}
              >
                {models.length ? 'Reload' : 'Load models'}
              </Button>
            </div>
            {#if modelsLoading}
              <InlineLoading description="Loading configured models…" />
            {:else if modelsMessage}
              <p class:notice-error={!models.length} class="field-notice">{modelsMessage}</p>
            {/if}
          </div>
        </fieldset>

        <fieldset>
          <legend>API & message</legend>
          <div class="field-stack">
            <Select id="endpoint" labelText="API type" bind:selected={endpoint}>
              <SelectItem value="/chat/completions" text="/chat/completions" />
              <SelectItem value="/responses" text="/responses" />
            </Select>
            <TextArea
              id="prompt"
              labelText="User message"
              placeholder="Ask the model anything…"
              rows={7}
              bind:value={prompt}
            />
          </div>
        </fieldset>

        <fieldset>
          <legend>Generation settings</legend>
          <p class="settings-helper">Optional parameters are omitted unless explicitly included.</p>
          <div class="settings-grid">
            <div class="optional-setting">
              <Checkbox
                id="include-temperature"
                labelText="Include temperature"
                bind:checked={includeTemperature}
              />
              <NumberInput
                id="temperature"
                labelText="Temperature"
                helperText="0–2"
                min={0}
                max={2}
                step={0.1}
                allowEmpty
                disabled={!includeTemperature}
                bind:value={temperature}
              />
            </div>
            <div class="optional-setting">
              <Checkbox
                id="include-max-tokens"
                labelText="Include max output tokens"
                bind:checked={includeMaxTokens}
              />
              <NumberInput
                id="max-tokens"
                labelText="Max output tokens"
                helperText="Leave empty to omit"
                min={1}
                max={128000}
                step={256}
                allowEmpty
                disabled={!includeMaxTokens}
                bind:value={maxTokens}
              />
            </div>
          </div>
          <div class="settings-grid settings-grid-bottom">
            <div class="optional-setting">
              <Checkbox
                id="include-reasoning"
                labelText="Include reasoning effort"
                bind:checked={includeReasoning}
              />
              <Select
                id="reasoning"
                labelText="Reasoning effort"
                disabled={!includeReasoning}
                bind:selected={reasoningEffort}
              >
                {#each reasoningEfforts as effort}
                  <SelectItem value={effort} text={effort[0]?.toUpperCase() + effort.slice(1)} />
                {/each}
              </Select>
            </div>
            <div class="toggle-wrap">
              <Toggle
                id="stream"
                labelText="Stream response"
                labelA="Off"
                labelB="On"
                bind:toggled={stream}
              />
            </div>
          </div>
        </fieldset>

        {#if requestError}
          <InlineNotification
            kind="error"
            lowContrast
            title="Request failed"
            subtitle={requestError}
            on:close={() => (requestError = '')}
          />
        {/if}

        <details class="request-preview">
          <summary>Preview request body</summary>
          <pre>{requestPreview}</pre>
        </details>

        <div class="form-actions">
          {#if requestState === 'running'}
            <Button kind="danger-tertiary" type="button" icon={StopFilled} on:click={abortRequest}>
              Cancel
            </Button>
          {/if}
          <Button kind="primary" type="submit" icon={Send} disabled={requestState === 'running'}>
            {requestState === 'running' ? 'Sending…' : 'Send request'}
          </Button>
        </div>
      </form>
    </section>

    <section class="response-panel" aria-labelledby="response-heading" aria-live="polite">
      <div class="panel-heading response-heading-row">
        <div>
          <p class="section-number">02</p>
          <h2 id="response-heading">Response</h2>
        </div>
        <div class="response-tools">
          <span
            class:active={requestState === 'running'}
            class:success={requestState === 'complete'}
            class:error={requestState === 'error'}
            class="status-label"
          >
            <span></span>
            {requestState === 'running'
              ? stream
                ? 'Streaming'
                : 'Waiting'
              : requestState === 'complete'
                ? 'Complete'
                : requestState === 'error'
                  ? 'Error'
                  : 'Ready'}
          </span>
          <Button
            kind="ghost"
            size="small"
            type="button"
            icon={copied ? Checkmark : Copy}
            iconDescription={copied ? 'Copied' : 'Copy response'}
            disabled={!displayText && !rawResponse}
            on:click={() => void copyResponse()}
          />
        </div>
      </div>

      {#if requestState === 'idle' && !rawResponse}
        <div class="empty-state">
          <div class="empty-icon"><Terminal size={28} /></div>
          <h3>Your response will appear here</h3>
          <p>Load a model, write a message, then send your first request.</p>
          <div class="empty-code"><span>$</span> waiting for request</div>
        </div>
      {:else}
        <div class="response-content">
          {#if requestState === 'running' && !displayText}
            <div class="waiting-state">
              <InlineLoading
                description={stream ? 'Waiting for the first token…' : 'Generating response…'}
              />
            </div>
          {/if}

          {#if displayText}
            <article class="answer">
              <p class="answer-label">Assistant</p>
              <div class="answer-text">{displayText}</div>
              {#if requestState === 'running'}<span class="cursor" aria-hidden="true"></span>{/if}
            </article>
          {:else if requestState === 'complete'}
            <div class="no-text">
              The API returned no assistant text. Inspect the raw response for tool calls or
              metadata.
            </div>
          {/if}

          {#if responseStatus || duration != null || usage}
            <dl class="metrics">
              {#if responseStatus}
                <div>
                  <dt>Status</dt>
                  <dd>{responseStatus}</dd>
                </div>
              {/if}
              {#if duration != null}
                <div>
                  <dt>Duration</dt>
                  <dd>{(duration / 1000).toFixed(2)}s</dd>
                </div>
              {/if}
              {#if usage}
                <div>
                  <dt>Input</dt>
                  <dd>{usage.input.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Output</dt>
                  <dd>{usage.output.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Total</dt>
                  <dd>{usage.total.toLocaleString()}</dd>
                </div>
              {/if}
            </dl>
          {/if}

          {#if rawResponse}
            <details class="raw-response" open={!displayText}>
              <summary>Raw {stream ? 'events' : 'JSON'} response</summary>
              <pre>{rawResponse}</pre>
            </details>
          {/if}
        </div>
      {/if}
    </section>
  </div>
</main>
