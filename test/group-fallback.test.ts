import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { AssistantMessage, Model, MutableModels } from '@earendil-works/pi-ai';

import type { ModelGroup } from '../src/groups.js';
import { chatAdapter, runCompletion } from '../src/server.js';

// Each provider names the same model differently; that is why groups are
// declared per model rather than per provider.
const group: ModelGroup = {
  name: 'free',
  members: [
    { providerId: 'github-copilot', modelId: 'gpt-5.4-mini' },
    { providerId: 'openai-codex', modelId: 'gpt-5-mini' },
  ],
};

function model(provider: string, id: string): Model<any> {
  return { provider, id, api: 'openai-completions' } as Model<any>;
}

const catalog = [model('github-copilot', 'gpt-5.4-mini'), model('openai-codex', 'gpt-5-mini')];

function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text: 'ok' }],
    api: 'openai-completions',
    provider: 'github-copilot',
    model: 'gpt-5.4-mini',
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 0,
    ...overrides,
  } as AssistantMessage;
}

type Behavior = (model: Model<any>) => Promise<AssistantMessage>;

function fakeModels(
  behavior: Behavior,
  options: { unconfigured?: string[]; stream?: (model: Model<any>) => AsyncIterable<any> } = {},
): { models: MutableModels; attempts: string[] } {
  const attempts: string[] = [];
  const unconfigured = new Set(options.unconfigured ?? []);

  const models = {
    getModels(providerId?: string) {
      return providerId ? catalog.filter((entry) => entry.provider === providerId) : catalog;
    },
    async getAuth(target: Model<any>) {
      return unconfigured.has(target.provider) ? undefined : { apiKey: 'k' };
    },
    async completeSimple(target: Model<any>) {
      attempts.push(target.provider);
      return behavior(target);
    },
    streamSimple(target: Model<any>) {
      attempts.push(target.provider);
      return options.stream!(target);
    },
  } as unknown as MutableModels;

  return { models, attempts };
}

function fakeRequest(body: unknown) {
  const warnings: string[] = [];
  const request = {
    body,
    log: {
      info() {},
      warn(_bindings: unknown, message: string) {
        warnings.push(message);
      },
      error() {},
    },
    raw: { once() {} },
  } as any;
  return { request, warnings };
}

function fakeReply() {
  const state: { status: number; payload: any; sse: string; ended: boolean } = {
    status: 200,
    payload: undefined,
    sse: '',
    ended: false,
  };
  const reply = {
    code(status: number) {
      state.status = status;
      return reply;
    },
    send(payload: unknown) {
      state.payload = payload;
      return reply;
    },
    raw: {
      headersSent: false,
      writableEnded: false,
      writeHead() {
        reply.raw.headersSent = true;
      },
      write(chunk: string) {
        state.sse += chunk;
        return true;
      },
      end() {
        state.ended = true;
        reply.raw.writableEnded = true;
      },
      once() {},
    },
  } as any;
  return { reply, state };
}

const chatBody = (extra: Record<string, unknown> = {}) => ({
  model: 'free',
  messages: [{ role: 'user', content: 'hi' }],
  ...extra,
});

test('falls back to the next provider when the first reports an upstream error', async () => {
  const { models, attempts } = fakeModels(async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex' }),
  );
  const { request, warnings } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['github-copilot', 'openai-codex']);
  assert.equal(state.status, 200);
  assert.equal(state.payload.model, 'openai-codex:gpt-5-mini');
  assert.ok(warnings.some((message) => message.includes('falling back')));
});

test('falls back when the first provider throws', async () => {
  const { models, attempts } = fakeModels(async (target) => {
    if (target.provider === 'github-copilot') throw new Error('connection reset');
    return assistantMessage({ provider: 'openai-codex' });
  });
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['github-copilot', 'openai-codex']);
  assert.equal(state.payload.model, 'openai-codex:gpt-5-mini');
});

test('skips unconfigured members without consuming an attempt', async () => {
  const { models, attempts } = fakeModels(
    async () => assistantMessage({ provider: 'openai-codex' }),
    {
      unconfigured: ['github-copilot'],
    },
  );
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['openai-codex']);
  assert.equal(state.payload.model, 'openai-codex:gpt-5-mini');
});

test('stops at the first success without trying later members', async () => {
  const { models, attempts } = fakeModels(async () => assistantMessage());
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['github-copilot']);
  assert.equal(state.payload.model, 'github-copilot:gpt-5.4-mini');
});

test('does not fall back when the client aborts', async () => {
  const { models, attempts } = fakeModels(async () =>
    assistantMessage({ stopReason: 'aborted', errorMessage: 'client went away' }),
  );
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['github-copilot']);
  assert.equal(state.status, 502);
});

test('reports every failure once the group is exhausted', async () => {
  const { models, attempts } = fakeModels(async (target) =>
    assistantMessage({ stopReason: 'error', errorMessage: `${target.provider} down` }),
  );
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['github-copilot', 'openai-codex']);
  assert.equal(state.status, 502);
  assert.match(state.payload.error.message, /github-copilot:gpt-5\.4-mini: github-copilot down/);
  assert.match(state.payload.error.message, /openai-codex:gpt-5-mini: openai-codex down/);
});

test('reports a configuration error when no member is configured', async () => {
  const { models, attempts } = fakeModels(async () => assistantMessage(), {
    unconfigured: ['github-copilot', 'openai-codex'],
  });
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, []);
  assert.equal(state.status, 400);
  assert.equal(state.payload.error.code, 'model_not_configured');
});

test('preserves single-model status codes for non-group requests', async () => {
  const unknown = fakeReply();
  const { request: unknownRequest } = fakeRequest(chatBody({ model: 'nope' }));
  await runCompletion(
    fakeModels(async () => assistantMessage()).models,
    [group],
    unknownRequest,
    unknown.reply,
    chatAdapter,
  );
  assert.equal(unknown.state.status, 404);
  assert.equal(unknown.state.payload.error.code, 'model_not_found');

  const unconfigured = fakeReply();
  const { request: unconfiguredRequest } = fakeRequest(
    chatBody({ model: 'github-copilot:gpt-5.4-mini' }),
  );
  await runCompletion(
    fakeModels(async () => assistantMessage(), { unconfigured: ['github-copilot'] }).models,
    [group],
    unconfiguredRequest,
    unconfigured.reply,
    chatAdapter,
  );
  assert.equal(unconfigured.state.status, 400);
  assert.equal(unconfigured.state.payload.error.code, 'model_not_configured');

  const upstream = fakeReply();
  const { request: upstreamRequest } = fakeRequest(
    chatBody({ model: 'github-copilot:gpt-5.4-mini' }),
  );
  await runCompletion(
    fakeModels(async () => assistantMessage({ stopReason: 'error', errorMessage: 'boom' })).models,
    [group],
    upstreamRequest,
    upstream.reply,
    chatAdapter,
  );
  assert.equal(upstream.state.status, 502);
  assert.equal(upstream.state.payload.error.message, 'boom');

  const thrown = fakeReply();
  const { request: thrownRequest } = fakeRequest(
    chatBody({ model: 'github-copilot:gpt-5.4-mini' }),
  );
  await runCompletion(
    fakeModels(async () => {
      throw new Error('kaput');
    }).models,
    [group],
    thrownRequest,
    thrown.reply,
    chatAdapter,
  );
  assert.equal(thrown.state.status, 500);
  assert.equal(thrown.state.payload.error.message, 'kaput');
});

test('streaming fails over when the first provider errors before sending anything', async () => {
  const message = assistantMessage({ provider: 'openai-codex' });
  const { models, attempts } = fakeModels(async () => message, {
    stream: (target) =>
      target.provider === 'github-copilot'
        ? (async function* () {
            yield {
              type: 'error',
              error: assistantMessage({ stopReason: 'error', errorMessage: 'upstream 500' }),
            };
          })()
        : (async function* () {
            yield { type: 'start', partial: message };
            yield { type: 'text_delta', contentIndex: 0, delta: 'hello', partial: message };
            yield { type: 'done', reason: 'stop', message };
          })(),
  });
  const { request } = fakeRequest(chatBody({ stream: true }));
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  assert.deepEqual(attempts, ['github-copilot', 'openai-codex']);
  // Nothing from the failed provider leaked into the stream.
  assert.ok(!state.sse.includes('upstream 500'));
  assert.ok(state.sse.includes('hello'));
  assert.ok(state.sse.includes('data: [DONE]'));
});

test('does not fail over once streaming output has been committed', async () => {
  const message = assistantMessage();
  const { models, attempts } = fakeModels(async () => message, {
    stream: (target) =>
      target.provider === 'github-copilot'
        ? (async function* () {
            yield { type: 'start', partial: message };
            yield { type: 'text_delta', contentIndex: 0, delta: 'partial', partial: message };
            yield {
              type: 'error',
              error: assistantMessage({ stopReason: 'error', errorMessage: 'died mid-stream' }),
            };
          })()
        : (async function* () {
            yield { type: 'done', reason: 'stop', message };
          })(),
  });
  const { request } = fakeRequest(chatBody({ stream: true }));
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  // The second provider must not be invoked; the client already has bytes.
  assert.deepEqual(attempts, ['github-copilot']);
  assert.ok(state.sse.includes('partial'));
  assert.ok(state.sse.includes('died mid-stream'));
  assert.equal(state.ended, true);
});

test('sends each provider the model id that provider actually uses', async () => {
  const seen: string[] = [];
  const { models } = fakeModels(async (target) => {
    seen.push(`${target.provider}:${target.id}`);
    return target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' });
  });
  const { request } = fakeRequest(chatBody());
  const { reply, state } = fakeReply();

  await runCompletion(models, [group], request, reply, chatAdapter);

  // The group is addressed by one name, but each upstream keeps its own id.
  assert.deepEqual(seen, ['github-copilot:gpt-5.4-mini', 'openai-codex:gpt-5-mini']);
  assert.equal(state.payload.model, 'openai-codex:gpt-5-mini');
});
