import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { Model } from '@earendil-works/pi-ai';

import { buildChatContext, buildResponsesContext } from '../src/openai-compat.js';

const model = {
  api: 'openai-responses',
  provider: 'openai',
  id: 'gpt-5.4-mini',
} as Model<any>;

const parameters = {
  type: 'object',
  properties: { city: { type: 'string' } },
  required: ['city'],
};

/** Chat completions nests the definition under `function`. */
const chatTool = {
  type: 'function',
  function: { name: 'weather', description: 'Get the weather.', parameters },
};

/** The responses API spells the same tool flat on the item. */
const responsesTool = {
  type: 'function',
  name: 'weather',
  description: 'Get the weather.',
  parameters,
};

const chatBody = (extra: Record<string, unknown> = {}) => ({
  messages: [{ role: 'user', content: 'Weather in Lisbon?' }],
  ...extra,
});

const responsesBody = (extra: Record<string, unknown> = {}) => ({
  input: 'Weather in Lisbon?',
  ...extra,
});

test('Chat completions tools reach the pi-ai context', async () => {
  const context = await buildChatContext(model, chatBody({ tools: [chatTool] }));

  assert.deepEqual(context.tools, [
    { name: 'weather', description: 'Get the weather.', parameters },
  ]);
});

test('Responses tools reach the pi-ai context despite their flat shape', async () => {
  const context = await buildResponsesContext(model, responsesBody({ tools: [responsesTool] }));

  assert.deepEqual(context.tools, [
    { name: 'weather', description: 'Get the weather.', parameters },
  ]);
});

test('A tool without a description or parameters gets usable defaults', async () => {
  const context = await buildResponsesContext(
    model,
    responsesBody({ tools: [{ type: 'function', name: 'ping' }] }),
  );

  assert.deepEqual(context.tools, [
    { name: 'ping', description: '', parameters: { type: 'object', properties: {} } },
  ]);
});

test('`tool_choice: "none"` withholds the tools from the provider', async () => {
  const chat = await buildChatContext(model, chatBody({ tools: [chatTool], tool_choice: 'none' }));
  const responses = await buildResponsesContext(
    model,
    responsesBody({ tools: [responsesTool], tool_choice: 'none' }),
  );

  assert.equal(chat.tools, undefined);
  assert.equal(responses.tools, undefined);
});

test('A forced tool choice narrows the context to that tool in either shape', async () => {
  const other = { type: 'function', name: 'clock' };
  const chat = await buildChatContext(
    model,
    chatBody({
      tools: [chatTool, { type: 'function', function: { name: 'clock' } }],
      tool_choice: { type: 'function', function: { name: 'weather' } },
    }),
  );
  const responses = await buildResponsesContext(
    model,
    responsesBody({
      tools: [responsesTool, other],
      tool_choice: { type: 'function', name: 'weather' },
    }),
  );

  assert.deepEqual(
    chat.tools?.map((tool) => tool.name),
    ['weather'],
  );
  assert.deepEqual(
    responses.tools?.map((tool) => tool.name),
    ['weather'],
  );
});

test('A forced tool choice naming no declared tool leaves the field unset', async () => {
  // An empty `tools` array is rejected by some providers, so it must not be
  // sent in place of "no tools".
  const chat = await buildChatContext(
    model,
    chatBody({
      tools: [chatTool],
      tool_choice: { type: 'function', function: { name: 'missing' } },
    }),
  );
  const responses = await buildResponsesContext(
    model,
    responsesBody({
      tools: [responsesTool],
      tool_choice: { type: 'function', name: 'missing' },
    }),
  );

  assert.equal(chat.tools, undefined);
  assert.equal(responses.tools, undefined);
});

test('Tools pi-ai cannot express are dropped', async () => {
  const context = await buildResponsesContext(
    model,
    responsesBody({
      tools: [{ type: 'web_search' }, { type: 'custom', name: 'grammar' }, responsesTool],
    }),
  );

  assert.deepEqual(
    context.tools?.map((tool) => tool.name),
    ['weather'],
  );
});

test('A request whose tools are all unsupported carries no tools', async () => {
  const context = await buildResponsesContext(
    model,
    responsesBody({ tools: [{ type: 'web_search' }] }),
  );

  assert.equal(context.tools, undefined);
});
