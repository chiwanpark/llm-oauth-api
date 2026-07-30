import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { AssistantMessage, Model, MutableModels } from '@earendil-works/pi-ai';

import {
  assistantReasoning,
  createChatCompletionResponse,
  createResponsesResponse,
} from '../src/openai-compat.js';
import { streamChatCompletions, streamResponses } from '../src/server.js';

const model = {
  api: 'openai-responses',
  provider: 'openai',
  id: 'gpt-5.4-mini',
} as Model<any>;

function assistantMessage(content: AssistantMessage['content']): AssistantMessage {
  return {
    role: 'assistant',
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    responseId: 'resp_upstream',
    usage: {
      input: 5,
      output: 7,
      cacheRead: 1,
      cacheWrite: 0,
      reasoning: 4,
      totalTokens: 12,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 1_700_000_000_000,
  };
}

function streamingModels(events: any[]): MutableModels {
  return {
    stream() {
      return (async function* () {
        for (const event of events) yield event;
      })();
    },
  } as unknown as MutableModels;
}

function captureReply() {
  let body = '';
  let ended = false;
  const reply = {
    raw: {
      writeHead() {},
      write(chunk: string) {
        body += chunk;
        return true;
      },
      end() {
        ended = true;
      },
    },
  } as any;

  return {
    reply,
    body: () => body,
    ended: () => ended,
  };
}

const logger = { info() {} } as any;

test('Chat Completions exposes reasoning content for a reasoning-only message', () => {
  const message = assistantMessage([
    { type: 'thinking', thinking: 'Check the constraints. ' },
    { type: 'thinking', thinking: 'The answer is 42.' },
  ]);

  assert.equal(assistantReasoning(message), 'Check the constraints. The answer is 42.');

  const response = createChatCompletionResponse(model, message);
  assert.equal(response.choices[0]?.message.content, null);
  assert.equal(
    response.choices[0]?.message.reasoning_content,
    'Check the constraints. The answer is 42.',
  );
  assert.equal(response.usage.completion_tokens_details.reasoning_tokens, 4);
});

test('Responses API emits a native reasoning item alongside assistant text', () => {
  const response = createResponsesResponse(
    model,
    assistantMessage([
      { type: 'thinking', thinking: 'Compare both options.' },
      { type: 'text', text: 'Option B is better.' },
    ]),
  );

  assert.deepEqual(
    response.output.map((item) => item.type),
    ['reasoning', 'message'],
  );
  assert.deepEqual(response.output[0].summary, [
    { type: 'summary_text', text: 'Compare both options.' },
  ]);
  assert.equal(response.output[1].content[0].text, 'Option B is better.');
  assert.equal(response.output_text, 'Option B is better.');
  assert.equal(response.usage.output_tokens_details.reasoning_tokens, 4);
});

test('Chat Completions streaming emits reasoning_content deltas', async () => {
  const message = assistantMessage([
    { type: 'thinking', thinking: 'Check facts.' },
    { type: 'text', text: 'Done.' },
  ]);
  const events = [
    { type: 'start', partial: message },
    { type: 'thinking_start', contentIndex: 0, partial: message },
    { type: 'thinking_delta', contentIndex: 0, delta: 'Check ', partial: message },
    { type: 'thinking_delta', contentIndex: 0, delta: 'facts.', partial: message },
    { type: 'thinking_end', contentIndex: 0, content: 'Check facts.', partial: message },
    { type: 'text_start', contentIndex: 1, partial: message },
    { type: 'text_delta', contentIndex: 1, delta: 'Done.', partial: message },
    { type: 'text_end', contentIndex: 1, content: 'Done.', partial: message },
    { type: 'done', reason: 'stop', message },
  ];
  const capture = captureReply();

  await streamChatCompletions(streamingModels(events), model, {}, {}, capture.reply, logger);

  const payloads = capture
    .body()
    .split('\n')
    .filter((line) => line.startsWith('data: {'))
    .map((line) => JSON.parse(line.slice('data: '.length)));
  assert.deepEqual(
    payloads
      .map((payload) => payload.choices?.[0]?.delta?.reasoning_content)
      .filter((delta) => delta != null),
    ['Check ', 'facts.'],
  );
  assert.ok(capture.body().includes('data: [DONE]\n\n'));
  assert.equal(capture.ended(), true);
});

test('Responses streaming emits native reasoning summary events and final output', async () => {
  const message = assistantMessage([
    { type: 'thinking', thinking: 'Check facts.' },
    { type: 'text', text: 'Done.' },
  ]);
  const events = [
    { type: 'start', partial: message },
    { type: 'thinking_start', contentIndex: 0, partial: message },
    { type: 'thinking_delta', contentIndex: 0, delta: 'Check ', partial: message },
    { type: 'thinking_delta', contentIndex: 0, delta: 'facts.', partial: message },
    { type: 'thinking_end', contentIndex: 0, content: 'Check facts.', partial: message },
    { type: 'text_start', contentIndex: 1, partial: message },
    { type: 'text_delta', contentIndex: 1, delta: 'Done.', partial: message },
    { type: 'text_end', contentIndex: 1, content: 'Done.', partial: message },
    { type: 'done', reason: 'stop', message },
  ];
  const capture = captureReply();

  await streamResponses(streamingModels(events), model, {}, {}, capture.reply, logger);

  const sseEvents = capture
    .body()
    .trim()
    .split('\n\n')
    .map((block) => {
      const lines = block.split('\n');
      return {
        event: lines.find((line) => line.startsWith('event: '))?.slice('event: '.length),
        data: JSON.parse(lines.find((line) => line.startsWith('data: '))!.slice('data: '.length)),
      };
    });

  const added = sseEvents.find(
    ({ event, data }) => event === 'response.output_item.added' && data.item?.type === 'reasoning',
  );
  assert.deepEqual(added?.data.item.summary, []);

  assert.deepEqual(
    sseEvents
      .filter(({ event }) => event === 'response.reasoning_summary_text.delta')
      .map(({ data }) => data.delta),
    ['Check ', 'facts.'],
  );

  const reasoningDone = sseEvents.find(
    ({ event }) => event === 'response.reasoning_summary_text.done',
  );
  assert.equal(reasoningDone?.data.text, 'Check facts.');

  const itemDone = sseEvents.find(
    ({ event, data }) => event === 'response.output_item.done' && data.item?.type === 'reasoning',
  );
  assert.deepEqual(itemDone?.data.item.summary, [{ type: 'summary_text', text: 'Check facts.' }]);

  const completed = sseEvents.find(({ event }) => event === 'response.completed');
  assert.deepEqual(completed?.data.response.output[0].summary, [
    { type: 'summary_text', text: 'Check facts.' },
  ]);
  assert.equal(capture.ended(), true);
});
