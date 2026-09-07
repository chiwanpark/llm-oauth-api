import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { Context, Model } from '@earendil-works/pi-ai';

import {
  conversationSessionId,
  requestSessionId,
  SESSION_ID_HEADER,
  withSessionAffinity,
} from '../src/session.js';

const openrouter = { provider: 'openrouter' } as Model<any>;
const anthropic = { provider: 'anthropic' } as Model<any>;

const request = (headers: Record<string, string> = {}, body: unknown = {}) =>
  ({ headers, body }) as any;

const context = (extra: Partial<Context> = {}): Context => ({
  systemPrompt: 'You are a helpful assistant.',
  messages: [{ role: 'user', content: 'Hello', timestamp: 1 }],
  ...extra,
});

test('a client header becomes the session id', () => {
  assert.equal(requestSessionId(request({ [SESSION_ID_HEADER]: 'abc-123' })), 'abc-123');
});

test('a body `session_id` is used when no header is sent', () => {
  assert.equal(requestSessionId(request({}, { session_id: 'from-body' })), 'from-body');
});

test('the header wins over the body', () => {
  assert.equal(
    requestSessionId(request({ [SESSION_ID_HEADER]: 'header' }, { session_id: 'body' })),
    'header',
  );
});

test('a blank or missing session id is ignored', () => {
  assert.equal(requestSessionId(request({ [SESSION_ID_HEADER]: '   ' })), undefined);
  assert.equal(requestSessionId(request({}, { session_id: 42 })), undefined);
  assert.equal(requestSessionId(request()), undefined);
});

test('an oversized or non-ASCII session id is hashed into a sendable one', () => {
  const long = requestSessionId(request({}, { session_id: 'x'.repeat(300) }));
  const unicode = requestSessionId(request({}, { session_id: '세션' }));

  for (const id of [long, unicode]) {
    assert.match(id!, /^[0-9a-f]{32}$/);
  }
});

test('the derived id is stable as the conversation grows', () => {
  const first = conversationSessionId(context());
  const later = conversationSessionId(
    context({
      messages: [
        { role: 'user', content: 'Hello', timestamp: 1 },
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'Hi' }],
          timestamp: 2,
        } as any,
        { role: 'user', content: 'And now?', timestamp: 3 },
      ],
    }),
  );

  assert.equal(first, later);
  assert.match(first!, /^[0-9a-f]{32}$/);
});

test('a different conversation derives a different id', () => {
  assert.notEqual(
    conversationSessionId(context()),
    conversationSessionId(context({ systemPrompt: 'You are a pirate.' })),
  );
});

test('image parts do not take part in the derived id', () => {
  const text = conversationSessionId(
    context({
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }], timestamp: 1 }],
    }),
  );
  const withImage = conversationSessionId(
    context({
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Hello' },
            { type: 'image', data: 'AAAA', mimeType: 'image/png' },
          ],
          timestamp: 1,
        },
      ],
    }),
  );

  assert.equal(text, withImage);
});

test('a conversation without text derives no id', () => {
  assert.equal(conversationSessionId({ messages: [] }), undefined);
  assert.equal(
    conversationSessionId({
      messages: [
        {
          role: 'user',
          content: [{ type: 'image', data: 'AAAA', mimeType: 'image/png' }],
          timestamp: 1,
        },
      ],
    }),
    undefined,
  );
});

test('OpenRouter requests carry the sticky routing header', () => {
  const options = withSessionAffinity(
    { temperature: 0.2 },
    openrouter,
    request({ [SESSION_ID_HEADER]: 'abc-123' }),
    context(),
  );

  assert.deepEqual(options, { temperature: 0.2, headers: { [SESSION_ID_HEADER]: 'abc-123' } });
});

test('OpenRouter requests without a client session id fall back to the derived one', () => {
  const options = withSessionAffinity({}, openrouter, request(), context()) as any;

  assert.equal(options.headers[SESSION_ID_HEADER], conversationSessionId(context()));
});

test('other providers are left untouched', () => {
  const options = { temperature: 0.2 };

  assert.equal(
    withSessionAffinity(options, anthropic, request({ [SESSION_ID_HEADER]: 'abc' }), context()),
    options,
  );
});

test('explicit option headers override the session header', () => {
  const options = withSessionAffinity(
    { headers: { [SESSION_ID_HEADER]: 'explicit' } },
    openrouter,
    request({ [SESSION_ID_HEADER]: 'from-client' }),
    context(),
  );

  assert.deepEqual(options.headers, { [SESSION_ID_HEADER]: 'explicit' });
});
