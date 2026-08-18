import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { AssistantMessage, Model, MutableModels } from '@earendil-works/pi-ai';

import {
  buildRenderOptions,
  createChatCompletionResponse,
  createResponsesResponse,
} from '../src/openai-compat.js';
import { streamChatCompletions, streamResponses } from '../src/server.js';

const model = {
  api: 'openai-responses',
  provider: 'openai',
  id: 'gpt-5.4-mini',
} as Model<any>;

function assistantMessage(content: AssistantMessage['content'] = []): AssistantMessage {
  return {
    role: 'assistant',
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 18,
      output: 64,
      cacheRead: 0,
      cacheWrite: 0,
      reasoning: 54,
      totalTokens: 82,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 1_787_030_519_000,
  };
}

function captureStream(events: any[]) {
  let body = '';
  const reply = {
    raw: {
      writeHead() {},
      write(chunk: string) {
        body += chunk;
        return true;
      },
      end() {},
    },
  } as any;
  const stream = () =>
    (async function* () {
      for (const event of events) yield event;
    })();

  return {
    reply,
    models: { stream, streamSimple: stream } as unknown as MutableModels,
    logger: { info() {} } as any,
    chunks: () =>
      body
        .split('\n')
        .filter((line) => line.startsWith('data: {'))
        .map((line) => JSON.parse(line.slice('data: '.length))),
    events: () =>
      body
        .trim()
        .split('\n\n')
        .map((block) => {
          const lines = block.split('\n');
          return {
            event: lines.find((line) => line.startsWith('event: '))?.slice('event: '.length),
            data: JSON.parse(
              lines.find((line) => line.startsWith('data: '))!.slice('data: '.length),
            ),
          };
        }),
  };
}

const textEvents = (message: AssistantMessage) => [
  { type: 'start', partial: message },
  { type: 'text_start', contentIndex: 0, partial: message },
  { type: 'text_delta', contentIndex: 0, delta: 'Hello!', partial: message },
  { type: 'text_end', contentIndex: 0, content: 'Hello!', partial: message },
  { type: 'done', reason: 'stop', message },
];

test('A chat completion carries exactly the fields OpenAI reports', () => {
  const response = createChatCompletionResponse(
    model,
    assistantMessage([{ type: 'text', text: 'Hello! How can I help you today?' }]),
  );

  assert.deepEqual(Object.keys(response), [
    'id',
    'object',
    'created',
    'model',
    'choices',
    'usage',
    'service_tier',
    'system_fingerprint',
  ]);
  assert.match(response.id, /^chatcmpl-/);
  assert.deepEqual(Object.keys(response.choices[0]!), [
    'index',
    'message',
    'logprobs',
    'finish_reason',
  ]);
  assert.deepEqual(Object.keys(response.choices[0]!.message), [
    'role',
    'content',
    'refusal',
    'annotations',
  ]);
  assert.deepEqual(response.usage, {
    prompt_tokens: 18,
    completion_tokens: 64,
    total_tokens: 82,
    prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
    completion_tokens_details: {
      reasoning_tokens: 54,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0,
    },
  });
  assert.equal(response.service_tier, 'default');
  assert.equal(response.system_fingerprint, null);
});

test('tool_calls appears only when the model called a tool', () => {
  const withCall = createChatCompletionResponse(
    model,
    assistantMessage([
      { type: 'toolCall', id: 'call_1', name: 'weather', arguments: { city: 'Oslo' } },
    ]),
  );

  assert.deepEqual(withCall.choices[0]!.message.tool_calls, [
    {
      id: 'call_1',
      type: 'function',
      function: { name: 'weather', arguments: '{"city":"Oslo"}' },
    },
  ]);
});

test('Chat streaming chunks match the OpenAI chunk shape', async () => {
  const message = assistantMessage([{ type: 'text', text: 'Hello!' }]);
  const capture = captureStream(textEvents(message));

  await streamChatCompletions(capture.models, model, {}, {}, capture.reply, capture.logger);

  const chunks = capture.chunks();
  assert.deepEqual(Object.keys(chunks[0]!), [
    'id',
    'object',
    'created',
    'model',
    'service_tier',
    'system_fingerprint',
    'choices',
  ]);
  assert.deepEqual(chunks[0]!.choices[0], {
    index: 0,
    delta: { role: 'assistant', content: '', refusal: null },
    logprobs: null,
    finish_reason: null,
  });
  assert.equal(chunks.at(-1)!.choices[0].finish_reason, 'stop');
  // Without stream_options the usage key is absent from every chunk.
  assert.ok(chunks.every((chunk) => !('usage' in chunk)));
});

test('Stream usage is reported only under stream_options.include_usage', async () => {
  const message = assistantMessage([{ type: 'text', text: 'Hello!' }]);
  const capture = captureStream(textEvents(message));

  await streamChatCompletions(
    capture.models,
    model,
    {},
    {},
    capture.reply,
    capture.logger,
    false,
    buildRenderOptions({ stream_options: { include_usage: true } }),
  );

  const chunks = capture.chunks();
  const usageChunk = chunks.at(-1)!;
  assert.deepEqual(usageChunk.choices, []);
  assert.equal(usageChunk.usage.total_tokens, 82);
  // Every earlier chunk announces the key with no value yet.
  assert.ok(chunks.slice(0, -1).every((chunk) => chunk.usage === null));
});

test('A response carries the request parameters OpenAI echoes back', () => {
  const render = buildRenderOptions({
    instructions: 'Be brief.',
    max_output_tokens: 256,
    metadata: { trace: 'abc' },
    previous_response_id: 'resp_prev',
    reasoning: { effort: 'high', summary: 'auto' },
    store: false,
    temperature: 0.2,
    tool_choice: 'required',
    tools: [{ type: 'function', name: 'weather' }],
    top_p: 0.5,
    truncation: 'auto',
    user: 'user-1',
  });

  const response = createResponsesResponse(
    model,
    assistantMessage([{ type: 'text', text: 'Hi.' }]),
    render,
  );

  assert.deepEqual(Object.keys(response), [
    'id',
    'object',
    'created_at',
    'status',
    'error',
    'incomplete_details',
    'instructions',
    'max_output_tokens',
    'model',
    'output',
    'parallel_tool_calls',
    'previous_response_id',
    'reasoning',
    'service_tier',
    'store',
    'temperature',
    'text',
    'tool_choice',
    'tools',
    'top_p',
    'truncation',
    'usage',
    'user',
    'metadata',
  ]);
  assert.equal(response.instructions, 'Be brief.');
  assert.equal(response.max_output_tokens, 256);
  assert.equal(response.previous_response_id, 'resp_prev');
  assert.deepEqual(response.reasoning, { effort: 'high', summary: 'auto' });
  assert.equal(response.store, false);
  assert.equal(response.temperature, 0.2);
  assert.equal(response.tool_choice, 'required');
  assert.deepEqual(response.tools, [{ type: 'function', name: 'weather' }]);
  assert.equal(response.top_p, 0.5);
  assert.equal(response.truncation, 'auto');
  assert.equal(response.user, 'user-1');
  assert.deepEqual(response.metadata, { trace: 'abc' });
  assert.deepEqual(response.usage, {
    input_tokens: 18,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 64,
    output_tokens_details: { reasoning_tokens: 54 },
    total_tokens: 82,
  });
});

test('A response left at its defaults reports OpenAI defaults', () => {
  const response = createResponsesResponse(
    model,
    assistantMessage([{ type: 'text', text: 'Hi.' }]),
    buildRenderOptions({}),
  );

  assert.equal(response.instructions, null);
  assert.equal(response.max_output_tokens, null);
  assert.equal(response.parallel_tool_calls, true);
  assert.equal(response.previous_response_id, null);
  assert.deepEqual(response.reasoning, { effort: null, summary: null });
  assert.equal(response.store, true);
  assert.equal(response.temperature, 1);
  assert.deepEqual(response.text, { format: { type: 'text' } });
  assert.equal(response.tool_choice, 'auto');
  assert.equal(response.top_p, 1);
  assert.equal(response.truncation, 'disabled');
  assert.equal(response.user, null);
  assert.deepEqual(response.metadata, {});
});

test('Responses streaming numbers its events and repeats the full response object', async () => {
  const message = assistantMessage([{ type: 'text', text: 'Hello!' }]);
  const capture = captureStream(textEvents(message));

  await streamResponses(
    capture.models,
    model,
    {},
    {},
    capture.reply,
    capture.logger,
    false,
    buildRenderOptions({ temperature: 0.7 }),
  );

  const events = capture.events();
  assert.deepEqual(
    events.map(({ data }) => data.sequence_number),
    events.map((_, index) => index),
  );
  // `response_id` is not part of the API; item events are keyed by item_id.
  assert.ok(events.every(({ data }) => !('response_id' in data)));

  const created = events.find(({ event }) => event === 'response.created')!;
  assert.equal(created.data.response.status, 'in_progress');
  assert.equal(created.data.response.temperature, 0.7);
  assert.deepEqual(created.data.response.output, []);
  assert.equal(created.data.response.usage, null);

  const completed = events.find(({ event }) => event === 'response.completed')!;
  assert.equal(completed.data.response.status, 'completed');
  assert.equal(completed.data.response.temperature, 0.7);
  assert.equal(completed.data.response.usage.total_tokens, 82);

  const textDelta = events.find(({ event }) => event === 'response.output_text.delta')!;
  assert.deepEqual(Object.keys(textDelta.data), [
    'type',
    'item_id',
    'output_index',
    'content_index',
    'delta',
    'sequence_number',
  ]);
});
