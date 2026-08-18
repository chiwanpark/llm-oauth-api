import assert from 'node:assert/strict';
import { test } from 'node:test';

import type {
  AssistantMessage,
  Model,
  MutableModels,
  ThinkingContent,
} from '@earendil-works/pi-ai';

import {
  buildChatContext,
  buildResponsesContext,
  createChatCompletionResponse,
  createResponsesResponse,
} from '../src/openai-compat.js';
import { streamChatCompletions, streamResponses } from '../src/server.js';

const openai = {
  api: 'openai-responses',
  provider: 'openai',
  id: 'gpt-5.4-mini',
} as Model<any>;

const anthropic = {
  api: 'anthropic-messages',
  provider: 'anthropic',
  id: 'claude-opus-5',
} as Model<any>;

/** What an OpenAI reasoning item looks like once pi-ai has stored it. */
const upstreamItem = {
  id: 'rs_upstream_1',
  type: 'reasoning',
  summary: [{ type: 'summary_text', text: 'Compare both options.' }],
  encrypted_content: 'gAAAAABmZXJuZXQtc3R5bGUtcGF5bG9hZA',
};

function assistantMessage(
  model: Model<any>,
  content: AssistantMessage['content'],
): AssistantMessage {
  return {
    role: 'assistant',
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
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

function thinkingOf(message: unknown): ThinkingContent[] {
  return (message as AssistantMessage).content.filter(
    (block): block is ThinkingContent => block.type === 'thinking',
  );
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
    body: () => body,
  };
}

function sseEvents(body: string) {
  return body
    .trim()
    .split('\n\n')
    .map((block) => {
      const lines = block.split('\n');
      return {
        event: lines.find((line) => line.startsWith('event: '))?.slice('event: '.length),
        data: JSON.parse(lines.find((line) => line.startsWith('data: '))!.slice('data: '.length)),
      };
    });
}

test('Responses omits encrypted content until the client asks for it', () => {
  const message = assistantMessage(openai, [
    {
      type: 'thinking',
      thinking: 'Compare both options.',
      thinkingSignature: JSON.stringify(upstreamItem),
    },
    { type: 'text', text: 'Option B.' },
  ]);

  const plain = createResponsesResponse(openai, message);
  assert.equal(plain.output[0].encrypted_content, undefined);
  // The provider's own item id is reported either way, so clients can pair the
  // reasoning with the turn it belongs to.
  assert.equal(plain.output[0].id, 'rs_upstream_1');

  const withEncrypted = createResponsesResponse(openai, message, {
    includeEncryptedReasoning: true,
  });
  assert.equal(withEncrypted.output[0].encrypted_content, upstreamItem.encrypted_content);
  assert.deepEqual(withEncrypted.output[0].summary, [
    { type: 'summary_text', text: 'Compare both options.' },
  ]);
});

test('An OpenAI reasoning item is replayed exactly as the provider issued it', async () => {
  const response = createResponsesResponse(
    openai,
    assistantMessage(openai, [
      {
        type: 'thinking',
        thinking: 'Compare both options.',
        thinkingSignature: JSON.stringify(upstreamItem),
      },
      { type: 'text', text: 'Option B.' },
    ]),
    { includeEncryptedReasoning: true },
  );

  const context = await buildResponsesContext(openai, {
    input: [
      { role: 'user', content: 'Which one?' },
      response.output[0],
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Option B.' }] },
      { role: 'user', content: 'Why?' },
    ],
  });

  const [thinking] = thinkingOf(context.messages[1]);
  assert.equal(thinking?.thinking, 'Compare both options.');
  assert.deepEqual(JSON.parse(thinking!.thinkingSignature!), {
    id: 'rs_upstream_1',
    type: 'reasoning',
    summary: [{ type: 'summary_text', text: 'Compare both options.' }],
    encrypted_content: upstreamItem.encrypted_content,
  });
  // Thinking has to lead the assistant turn for providers that require it.
  assert.equal((context.messages[1] as AssistantMessage).content[0]?.type, 'thinking');
});

test('An OpenAI reasoning item without ciphertext is replayed as plain reasoning', async () => {
  const context = await buildResponsesContext(openai, {
    input: [
      { role: 'user', content: 'Which one?' },
      {
        type: 'reasoning',
        id: 'rs_upstream_1',
        summary: [{ type: 'summary_text', text: 'Compare both options.' }],
      },
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Option B.' }] },
    ],
  });

  // Replaying a bare id would make the provider look up an item it never
  // stored, so the reasoning goes back as text only.
  assert.deepEqual(thinkingOf(context.messages[1]), [
    { type: 'thinking', thinking: 'Compare both options.' },
  ]);
});

test('Reasoning that precedes a tool call rides along with that tool call', async () => {
  const context = await buildResponsesContext(openai, {
    input: [
      { role: 'user', content: 'Weather?' },
      { type: 'reasoning', ...upstreamItem },
      { type: 'function_call', call_id: 'call_1', name: 'weather', arguments: '{"city":"Oslo"}' },
      { type: 'function_call_output', call_id: 'call_1', output: 'sunny' },
    ],
  });

  const assistant = context.messages[1] as AssistantMessage;
  assert.deepEqual(
    assistant.content.map((block) => block.type),
    ['thinking', 'toolCall'],
  );
  assert.equal(
    JSON.parse(thinkingOf(assistant)[0]!.thinkingSignature!).encrypted_content,
    upstreamItem.encrypted_content,
  );
});

test('Reasoning without a following assistant turn is dropped', async () => {
  const context = await buildResponsesContext(openai, {
    input: [
      { role: 'user', content: 'Which one?' },
      { type: 'reasoning', ...upstreamItem },
    ],
  });

  assert.deepEqual(
    context.messages.map((message) => message.role),
    ['user'],
  );
});

test('An Anthropic thinking signature travels as encrypted content', async () => {
  const response = createResponsesResponse(
    anthropic,
    assistantMessage(anthropic, [
      { type: 'thinking', thinking: 'Weigh the tradeoffs.', thinkingSignature: 'ErUBCkYIBBgCKkA' },
      { type: 'text', text: 'Option B.' },
    ]),
    { includeEncryptedReasoning: true },
  );

  assert.equal(response.output[0].encrypted_content, 'ErUBCkYIBBgCKkA');

  const context = await buildResponsesContext(anthropic, {
    input: [
      { role: 'user', content: 'Which one?' },
      response.output[0],
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Option B.' }] },
    ],
  });

  assert.deepEqual(thinkingOf(context.messages[1]), [
    { type: 'thinking', thinking: 'Weigh the tradeoffs.', thinkingSignature: 'ErUBCkYIBBgCKkA' },
  ]);
});

test('Redacted reasoning keeps its opaque payload and stays redacted', async () => {
  const response = createResponsesResponse(
    anthropic,
    assistantMessage(anthropic, [
      {
        type: 'thinking',
        thinking: '[Reasoning redacted]',
        thinkingSignature: 'EroBCkYIBRgCKkBd',
        redacted: true,
      },
      { type: 'text', text: 'Done.' },
    ]),
    { includeEncryptedReasoning: true },
  );

  // Redacted reasoning has no readable summary; the payload is the whole item.
  assert.deepEqual(response.output[0].summary, []);
  assert.equal(response.output[0].encrypted_content, 'EroBCkYIBRgCKkBd');

  const context = await buildResponsesContext(anthropic, {
    input: [
      { role: 'user', content: 'Go' },
      response.output[0],
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Done.' }] },
    ],
  });

  assert.deepEqual(thinkingOf(context.messages[1]), [
    { type: 'thinking', thinking: '', thinkingSignature: 'EroBCkYIBRgCKkBd', redacted: true },
  ]);
});

test('Chat Completions returns reasoning_details and takes them back', async () => {
  const response = createChatCompletionResponse(
    anthropic,
    assistantMessage(anthropic, [
      { type: 'thinking', thinking: 'Step one.', thinkingSignature: 'sig-a' },
      { type: 'text', text: 'Answer.' },
    ]),
  );

  const assistant = response.choices[0]!.message;
  assert.equal(assistant.reasoning_content, 'Step one.');
  assert.deepEqual(assistant.reasoning_details, [
    {
      type: 'reasoning.text',
      index: 0,
      format: 'anthropic-claude-v1',
      text: 'Step one.',
      signature: 'sig-a',
    },
  ]);

  const context = await buildChatContext(anthropic, {
    messages: [
      { role: 'user', content: 'Question?' },
      { role: 'assistant', content: 'Answer.', ...assistant },
      { role: 'user', content: 'And now?' },
    ],
  });

  assert.deepEqual(thinkingOf(context.messages[1]), [
    { type: 'thinking', thinking: 'Step one.', thinkingSignature: 'sig-a' },
  ]);
});

test('Chat Completions carries an OpenAI reasoning item as summary plus ciphertext', async () => {
  const response = createChatCompletionResponse(
    openai,
    assistantMessage(openai, [
      {
        type: 'thinking',
        thinking: 'Compare both options.',
        thinkingSignature: JSON.stringify(upstreamItem),
      },
      { type: 'text', text: 'Option B.' },
    ]),
  );

  const assistant = response.choices[0]!.message;
  assert.deepEqual(assistant.reasoning_details, [
    {
      type: 'reasoning.summary',
      index: 0,
      format: 'openai-responses-v1',
      summary: 'Compare both options.',
      id: 'rs_upstream_1',
    },
    {
      type: 'reasoning.encrypted',
      index: 0,
      format: 'openai-responses-v1',
      data: upstreamItem.encrypted_content,
      id: 'rs_upstream_1',
    },
  ]);

  const context = await buildChatContext(openai, {
    messages: [
      { role: 'user', content: 'Which one?' },
      { role: 'assistant', content: 'Option B.', reasoning_details: assistant.reasoning_details },
    ],
  });

  assert.deepEqual(JSON.parse(thinkingOf(context.messages[1])[0]!.thinkingSignature!), {
    id: 'rs_upstream_1',
    type: 'reasoning',
    summary: [{ type: 'summary_text', text: 'Compare both options.' }],
    encrypted_content: upstreamItem.encrypted_content,
  });
});

test('Reasoning from another provider is replayed as text only', async () => {
  const context = await buildChatContext(openai, {
    messages: [
      { role: 'user', content: 'Question?' },
      {
        role: 'assistant',
        content: 'Answer.',
        reasoning_details: [
          {
            type: 'reasoning.text',
            index: 0,
            format: 'anthropic-claude-v1',
            text: 'Step one.',
            signature: 'sig-a',
          },
        ],
      },
    ],
  });

  assert.deepEqual(thinkingOf(context.messages[1]), [{ type: 'thinking', thinking: 'Step one.' }]);
});

test('Chat Completions still accepts a bare reasoning_content string', async () => {
  const context = await buildChatContext(anthropic, {
    messages: [
      { role: 'user', content: 'Question?' },
      { role: 'assistant', content: 'Answer.', reasoning_content: 'Unsigned thought.' },
    ],
  });

  assert.deepEqual(thinkingOf(context.messages[1]), [
    { type: 'thinking', thinking: 'Unsigned thought.' },
  ]);
});

test('Responses streaming reports the provider reasoning item once it closes', async () => {
  const message = assistantMessage(openai, [
    {
      type: 'thinking',
      thinking: 'Compare both options.',
      thinkingSignature: JSON.stringify(upstreamItem),
    },
    { type: 'text', text: 'Option B.' },
  ]);
  const capture = captureStream([
    { type: 'start', partial: message },
    { type: 'thinking_start', contentIndex: 0, partial: message },
    { type: 'thinking_delta', contentIndex: 0, delta: 'Compare both options.', partial: message },
    { type: 'thinking_end', contentIndex: 0, content: 'Compare both options.', partial: message },
    { type: 'text_start', contentIndex: 1, partial: message },
    { type: 'text_delta', contentIndex: 1, delta: 'Option B.', partial: message },
    { type: 'text_end', contentIndex: 1, content: 'Option B.', partial: message },
    { type: 'done', reason: 'stop', message },
  ]);

  await streamResponses(capture.models, openai, {}, {}, capture.reply, capture.logger, false, {
    includeEncryptedReasoning: true,
  });

  const events = sseEvents(capture.body());
  const itemDone = events.find(
    ({ event, data }) => event === 'response.output_item.done' && data.item?.type === 'reasoning',
  );
  assert.equal(itemDone?.data.item.id, 'rs_upstream_1');
  assert.equal(itemDone?.data.item.encrypted_content, upstreamItem.encrypted_content);

  const completed = events.find(({ event }) => event === 'response.completed');
  assert.equal(completed?.data.response.output[0].id, 'rs_upstream_1');
  assert.equal(
    completed?.data.response.output[0].encrypted_content,
    upstreamItem.encrypted_content,
  );
});

test('Chat streaming reports reasoning_details on the final chunk', async () => {
  const message = assistantMessage(anthropic, [
    { type: 'thinking', thinking: 'Check facts.', thinkingSignature: 'sig-a' },
    { type: 'text', text: 'Done.' },
  ]);
  const capture = captureStream([
    { type: 'start', partial: message },
    { type: 'thinking_start', contentIndex: 0, partial: message },
    { type: 'thinking_delta', contentIndex: 0, delta: 'Check facts.', partial: message },
    { type: 'thinking_end', contentIndex: 0, content: 'Check facts.', partial: message },
    { type: 'text_start', contentIndex: 1, partial: message },
    { type: 'text_delta', contentIndex: 1, delta: 'Done.', partial: message },
    { type: 'text_end', contentIndex: 1, content: 'Done.', partial: message },
    { type: 'done', reason: 'stop', message },
  ]);

  await streamChatCompletions(capture.models, anthropic, {}, {}, capture.reply, capture.logger);

  const chunks = capture
    .body()
    .split('\n')
    .filter((line) => line.startsWith('data: {'))
    .map((line) => JSON.parse(line.slice('data: '.length)));
  const final = chunks.find((chunk) => chunk.choices?.[0]?.finish_reason);
  assert.deepEqual(final.choices[0].delta.reasoning_details, [
    {
      type: 'reasoning.text',
      index: 0,
      format: 'anthropic-claude-v1',
      text: 'Check facts.',
      signature: 'sig-a',
    },
  ]);
});
