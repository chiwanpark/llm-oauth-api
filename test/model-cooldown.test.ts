import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { AssistantMessage, Model, MutableModels } from '@earendil-works/pi-ai';

import type { ModelGroup } from '../src/groups.js';
import {
  createModelCooldown,
  parseCooldownSeconds,
  type ModelCooldown,
} from '../src/model-cooldown.js';
import { chatAdapter, runCompletion } from '../src/server.js';

/**
 * The failures a group protects against outlive a single request, so a member
 * that just failed is passed over instead of being asked again immediately.
 */
const group: ModelGroup = {
  name: 'free',
  members: [
    { providerId: 'github-copilot', modelId: 'gpt-5.4-mini' },
    { providerId: 'openai-codex', modelId: 'gpt-5-mini' },
  ],
};

const COOLDOWN_MS = 300_000;

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
  const logs: string[] = [];
  const request = {
    body,
    log: {
      info(_bindings: unknown, message: string) {
        logs.push(message);
      },
      warn(_bindings: unknown, message: string) {
        logs.push(message);
      },
      error() {},
    },
    raw: { once() {} },
  } as any;
  return { request, logs };
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

/** One request against the shared cooldown store, returning what the client saw. */
async function runOnce(cooldown: ModelCooldown, fakes: { models: MutableModels }, body: unknown) {
  const { request } = fakeRequest(body);
  const { reply, state } = fakeReply();
  await runCompletion(fakes.models, [group], request, reply, chatAdapter, cooldown);
  return state;
}

/** A clock the tests move by hand, so no test has to wait out a cooldown. */
function clock(start = 1_000) {
  let current = start;
  return {
    now: () => current,
    advance(ms: number) {
      current += ms;
    },
  };
}

test('passes over a member that failed on an earlier request', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const failFirst: Behavior = async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' });

  const first = fakeModels(failFirst);
  const firstRequest = fakeRequest(chatBody());
  const firstReply = fakeReply();
  await runCompletion(
    first.models,
    [group],
    firstRequest.request,
    firstReply.reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(first.attempts, ['github-copilot', 'openai-codex']);

  // Same failure, one second later: the group starts where it left off.
  time.advance(1_000);
  const second = fakeModels(failFirst);
  const secondRequest = fakeRequest(chatBody());
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    secondRequest.request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['openai-codex']);
  assert.equal(secondReply.state.status, 200);
  assert.equal(secondReply.state.payload.model, 'openai-codex:gpt-5-mini');
  assert.ok(secondRequest.logs.some((message) => message.includes('recently failed')));
});

test('tries the model again once the cooldown window passes', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });

  const first = fakeModels(async () =>
    assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' }),
  );
  const firstRequest = fakeRequest(chatBody());
  await runCompletion(
    first.models,
    [group],
    firstRequest.request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(first.attempts, ['github-copilot', 'openai-codex']);

  time.advance(COOLDOWN_MS);
  const second = fakeModels(async () => assistantMessage());
  const secondRequest = fakeRequest(chatBody());
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    secondRequest.request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['github-copilot']);
  assert.equal(secondReply.state.payload.model, 'github-copilot:gpt-5.4-mini');

  // Responding again clears the record, so the next request needs no window.
  const third = fakeModels(async () => assistantMessage());
  const thirdRequest = fakeRequest(chatBody());
  await runCompletion(
    third.models,
    [group],
    thirdRequest.request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(third.attempts, ['github-copilot']);
});

test('tries every member when they are all cooling down', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const failAll: Behavior = async (target) =>
    assistantMessage({ stopReason: 'error', errorMessage: `${target.provider} down` });

  const first = fakeModels(failAll);
  await runCompletion(
    first.models,
    [group],
    fakeRequest(chatBody()).request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(first.attempts, ['github-copilot', 'openai-codex']);

  // Skipping everything would take the group offline for the rest of the
  // window, so a request with nothing ready probes the members instead.
  time.advance(1_000);
  const second = fakeModels(async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'still down' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' }),
  );
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    fakeRequest(chatBody()).request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['github-copilot', 'openai-codex']);
  assert.equal(secondReply.state.payload.model, 'openai-codex:gpt-5-mini');
});

test('reports the skipped member when the rest of the group fails', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });

  const first = fakeModels(async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' }),
  );
  await runCompletion(
    first.models,
    [group],
    fakeRequest(chatBody()).request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );

  time.advance(60_000);
  const second = fakeModels(async () =>
    assistantMessage({
      provider: 'openai-codex',
      stopReason: 'error',
      errorMessage: 'upstream 500',
    }),
  );
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    fakeRequest(chatBody()).request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['openai-codex']);
  assert.equal(secondReply.state.status, 502);
  const message = secondReply.state.payload.error.message;
  // The skipped member is part of the explanation, with what is left to wait.
  assert.match(message, /github-copilot:gpt-5\.4-mini: skipped for another 4m/);
  assert.match(message, /rate limited/);
  assert.match(message, /openai-codex:gpt-5-mini: upstream 500/);
});

test('a client disconnect does not put the model on cooldown', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });

  const first = fakeModels(async () =>
    assistantMessage({ stopReason: 'aborted', errorMessage: 'client went away' }),
  );
  await runCompletion(
    first.models,
    [group],
    fakeRequest(chatBody()).request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(first.attempts, ['github-copilot']);

  const second = fakeModels(async () => assistantMessage());
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    fakeRequest(chatBody()).request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['github-copilot']);
  assert.equal(secondReply.state.payload.model, 'github-copilot:gpt-5.4-mini');
});

test('an unconfigured member is not put on cooldown', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });

  const first = fakeModels(async () => assistantMessage({ provider: 'openai-codex' }), {
    unconfigured: ['github-copilot'],
  });
  await runCompletion(
    first.models,
    [group],
    fakeRequest(chatBody()).request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );

  // Credentials can appear at any time, and checking for them costs nothing.
  const second = fakeModels(async () => assistantMessage());
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    fakeRequest(chatBody()).request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['github-copilot']);
  assert.equal(secondReply.state.payload.model, 'github-copilot:gpt-5.4-mini');
});

test('a stream that fails before sending anything puts the model on cooldown', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const message = assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' });
  const streams = (target: Model<any>) =>
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
        })();

  const first = fakeModels(async () => message, { stream: streams });
  await runCompletion(
    first.models,
    [group],
    fakeRequest(chatBody({ stream: true })).request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(first.attempts, ['github-copilot', 'openai-codex']);

  const second = fakeModels(async () => message, { stream: streams });
  const secondReply = fakeReply();
  await runCompletion(
    second.models,
    [group],
    fakeRequest(chatBody({ stream: true })).request,
    secondReply.reply,
    chatAdapter,
    cooldown,
  );

  assert.deepEqual(second.attempts, ['openai-codex']);
  assert.ok(secondReply.state.sse.includes('hello'));
});

test('a single-model request is never skipped but still records the failure', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const failing = fakeModels(async () =>
    assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' }),
  );

  const direct = fakeReply();
  await runCompletion(
    failing.models,
    [group],
    fakeRequest(chatBody({ model: 'github-copilot:gpt-5.4-mini' })).request,
    direct.reply,
    chatAdapter,
    cooldown,
  );
  assert.equal(direct.state.status, 502);

  // Nothing to fall back to, so the request is still attempted...
  const again = fakeReply();
  await runCompletion(
    failing.models,
    [group],
    fakeRequest(chatBody({ model: 'github-copilot:gpt-5.4-mini' })).request,
    again.reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(failing.attempts, ['github-copilot', 'github-copilot']);
  assert.equal(again.state.status, 502);
  assert.equal(again.state.payload.error.message, 'rate limited');

  // ...but the group knows the same upstream model is unhealthy.
  const grouped = fakeModels(async () => assistantMessage({ provider: 'openai-codex' }));
  await runCompletion(
    grouped.models,
    [group],
    fakeRequest(chatBody()).request,
    fakeReply().reply,
    chatAdapter,
    cooldown,
  );
  assert.deepEqual(grouped.attempts, ['openai-codex']);
});

test('a plain model request runs even while that model is cooling down', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });

  const failing = fakeModels(async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' }),
  );
  await runOnce(cooldown, failing, chatBody());
  assert.deepEqual(failing.attempts, ['github-copilot', 'openai-codex']);

  // The group skips it, but a request that names it has nowhere else to go, so
  // skipping would turn the cooldown into a 5-minute outage for that model.
  time.advance(1_000);
  const direct = fakeModels(async () => assistantMessage());
  const state = await runOnce(cooldown, direct, chatBody({ model: 'github-copilot:gpt-5.4-mini' }));
  assert.deepEqual(direct.attempts, ['github-copilot']);
  assert.equal(state.status, 200);
  assert.equal(state.payload.model, 'github-copilot:gpt-5.4-mini');

  const streamed = fakeModels(async () => assistantMessage(), {
    stream: () =>
      (async function* () {
        yield { type: 'done', reason: 'stop', message: assistantMessage() };
      })(),
  });
  await runOnce(
    cooldown,
    streamed,
    chatBody({ model: 'github-copilot:gpt-5.4-mini', stream: true }),
  );
  assert.deepEqual(streamed.attempts, ['github-copilot']);
});

test('a plain model request that succeeds takes the model off cooldown', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });

  const failing = fakeModels(async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' }),
  );
  await runOnce(cooldown, failing, chatBody());

  const skipping = fakeModels(async () => assistantMessage({ provider: 'openai-codex' }));
  await runOnce(cooldown, skipping, chatBody());
  assert.deepEqual(skipping.attempts, ['openai-codex']);

  // A direct request proves the model answers again, well inside the window.
  time.advance(1_000);
  const recovered = fakeModels(async () => assistantMessage());
  await runOnce(cooldown, recovered, chatBody({ model: 'github-copilot:gpt-5.4-mini' }));
  assert.deepEqual(recovered.attempts, ['github-copilot']);

  // So the group stops passing over it, without waiting out the rest of the
  // window.
  time.advance(1_000);
  const grouped = fakeModels(async () => assistantMessage());
  const state = await runOnce(cooldown, grouped, chatBody());
  assert.deepEqual(grouped.attempts, ['github-copilot']);
  assert.equal(state.payload.model, 'github-copilot:gpt-5.4-mini');
});

test('a zero window disables skipping', async () => {
  const cooldown = createModelCooldown({ cooldownMs: 0 });
  const behavior: Behavior = async (target) =>
    target.provider === 'github-copilot'
      ? assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' })
      : assistantMessage({ provider: 'openai-codex', model: 'gpt-5-mini' });

  for (let attempt = 0; attempt < 2; attempt += 1) {
    const round = fakeModels(behavior);
    await runCompletion(
      round.models,
      [group],
      fakeRequest(chatBody()).request,
      fakeReply().reply,
      chatAdapter,
      cooldown,
    );
    assert.deepEqual(round.attempts, ['github-copilot', 'openai-codex']);
  }
});

test('records a streaming failure reported on a plain model request', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const failing = fakeModels(async () => assistantMessage(), {
    stream: () =>
      (async function* () {
        yield {
          type: 'error',
          error: assistantMessage({ stopReason: 'error', errorMessage: 'upstream 500' }),
        };
      })(),
  });

  // Without a member to fall back to, the failure is reported on the stream
  // rather than thrown, which is the case that used to go unnoticed.
  const state = await runOnce(
    cooldown,
    failing,
    chatBody({ model: 'github-copilot:gpt-5.4-mini', stream: true }),
  );
  assert.deepEqual(failing.attempts, ['github-copilot']);
  assert.ok(state.sse.includes('upstream 500'));

  const grouped = fakeModels(async () => assistantMessage({ provider: 'openai-codex' }));
  await runOnce(cooldown, grouped, chatBody());
  assert.deepEqual(grouped.attempts, ['openai-codex']);
});

test('records a stream that broke after output reached the client', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const message = assistantMessage();
  const first = fakeModels(async () => message, {
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

  const state = await runOnce(cooldown, first, chatBody({ stream: true }));
  // The client keeps the partial answer; no other member is tried.
  assert.deepEqual(first.attempts, ['github-copilot']);
  assert.ok(state.sse.includes('died mid-stream'));

  // The next request has no reason to expect better from that model.
  const second = fakeModels(async () => assistantMessage({ provider: 'openai-codex' }));
  await runOnce(cooldown, second, chatBody());
  assert.deepEqual(second.attempts, ['openai-codex']);
});

test('a completed stream clears an earlier failure', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const message = assistantMessage();

  // Fail both members so neither is preferred over the other.
  const broken = fakeModels(async (target) =>
    assistantMessage({ stopReason: 'error', errorMessage: `${target.provider} down` }),
  );
  await runOnce(cooldown, broken, chatBody());
  assert.deepEqual(broken.attempts, ['github-copilot', 'openai-codex']);

  time.advance(1_000);
  const recovered = fakeModels(async () => message, {
    stream: () =>
      (async function* () {
        yield { type: 'start', partial: message };
        yield { type: 'text_delta', contentIndex: 0, delta: 'hello', partial: message };
        yield { type: 'done', reason: 'stop', message };
      })(),
  });
  await runOnce(cooldown, recovered, chatBody({ stream: true }));
  assert.deepEqual(recovered.attempts, ['github-copilot']);

  // Only the model that answered is ready again; the other is still cooling.
  time.advance(1_000);
  const third = fakeModels(async () => message);
  await runOnce(cooldown, third, chatBody());
  assert.deepEqual(third.attempts, ['github-copilot']);
});

test('a disconnect reported on the stream does not put the model on cooldown', async () => {
  const time = clock();
  const cooldown = createModelCooldown({ cooldownMs: COOLDOWN_MS, now: time.now });
  const message = assistantMessage();
  const first = fakeModels(async () => message, {
    stream: () =>
      (async function* () {
        yield { type: 'start', partial: message };
        yield { type: 'text_delta', contentIndex: 0, delta: 'partial', partial: message };
        yield {
          type: 'error',
          error: assistantMessage({ stopReason: 'aborted', errorMessage: 'client went away' }),
        };
      })(),
  });

  await runOnce(cooldown, first, chatBody({ stream: true }));
  assert.deepEqual(first.attempts, ['github-copilot']);

  const second = fakeModels(async () => message);
  const state = await runOnce(cooldown, second, chatBody());
  assert.deepEqual(second.attempts, ['github-copilot']);
  assert.equal(state.payload.model, 'github-copilot:gpt-5.4-mini');
});

test('parseCooldownSeconds accepts zero and rejects nonsense', () => {
  assert.equal(parseCooldownSeconds('300', '--model-cooldown'), 300_000);
  assert.equal(parseCooldownSeconds('0', '--model-cooldown'), 0);
  assert.throws(
    () => parseCooldownSeconds('-1', '--model-cooldown'),
    /non-negative number of seconds/,
  );
  assert.throws(() => parseCooldownSeconds('later', '--model-cooldown'));
});
