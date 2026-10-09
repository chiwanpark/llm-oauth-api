import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  collapseSystemMessages,
  getCurrentSystemPrompt,
  normalizeContext,
  type Context,
  type Model,
} from '@earendil-works/pi-ai';

import { buildChatContext, buildResponsesContext } from '../src/openai-compat.js';

const model = {
  api: 'openai-responses',
  provider: 'openai',
  id: 'gpt-5.4-mini',
} as Model<any>;

function roles(context: Context): string[] {
  return context.messages.map((message) => message.role);
}

function systemTexts(context: Context): unknown[] {
  return context.messages
    .filter((message) => message.role === 'system')
    .map((message) => message.content);
}

test('chat: leading system and developer messages form the system prompt', async () => {
  const context = await buildChatContext(model, {
    messages: [
      { role: 'system', content: 'Be brief.' },
      { role: 'developer', content: [{ type: 'text', text: 'Answer in English.' }] },
      { role: 'user', content: 'hi' },
    ],
  });

  assert.equal(context.systemPrompt, 'Be brief.\n\nAnswer in English.');
  assert.deepEqual(roles(context), ['user']);
});

test('chat: a system message after the first turn stays in place', async () => {
  const context = await buildChatContext(model, {
    messages: [
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'developer', content: 'Now be thorough.' },
      { role: 'user', content: 'explain' },
    ],
  });

  assert.equal(context.systemPrompt, 'Be brief.');
  assert.deepEqual(roles(context), ['user', 'assistant', 'system', 'user']);
  assert.deepEqual(systemTexts(context), ['Now be thorough.']);
});

test('chat: an empty mid-conversation system message is dropped', async () => {
  const context = await buildChatContext(model, {
    messages: [
      { role: 'user', content: 'hi' },
      { role: 'system', content: '   ' },
    ],
  });

  assert.deepEqual(roles(context), ['user']);
});

test('chat: without a leading system message, a later one is not promoted', async () => {
  const context = await buildChatContext(model, {
    messages: [
      { role: 'user', content: 'hi' },
      { role: 'system', content: 'Be brief.' },
    ],
  });

  assert.equal(context.systemPrompt, undefined);
  assert.deepEqual(roles(context), ['user', 'system']);
});

test('responses: instructions and leading items form the prompt, later items stay in place', async () => {
  const context = await buildResponsesContext(model, {
    instructions: 'Be brief.',
    input: [
      { type: 'message', role: 'developer', content: [{ type: 'input_text', text: 'No emoji.' }] },
      { type: 'message', role: 'user', content: 'hi' },
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'hello' }] },
      { role: 'system', content: 'Now be thorough.' },
      { role: 'user', content: 'explain' },
    ],
  });

  assert.equal(context.systemPrompt, 'Be brief.\n\nNo emoji.');
  assert.deepEqual(roles(context), ['user', 'assistant', 'system', 'user']);
  assert.deepEqual(systemTexts(context), ['Now be thorough.']);
});

test('models without mid-conversation support see the previous combined prompt', async () => {
  const context = await buildChatContext(model, {
    messages: [
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'hi' },
      { role: 'developer', content: 'Now be thorough.' },
    ],
  });

  const collapsed = collapseSystemMessages(normalizeContext(context));
  assert.equal(getCurrentSystemPrompt(collapsed.messages), 'Be brief.\n\nNow be thorough.');
  assert.deepEqual(
    collapsed.messages.map((message) => message.role),
    ['system', 'user'],
  );
});
