import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';

import { createRequestSignal } from '../src/server.js';

/**
 * Both streams emit `close` during a perfectly healthy request, so these tests
 * pin down which of those closes actually mean "the client is gone".
 */
function harness() {
  const rawRequest = Object.assign(new EventEmitter(), { complete: false });
  const rawReply = Object.assign(new EventEmitter(), { writableEnded: false });
  const request = { raw: rawRequest, socket: { destroyed: false } } as any;
  const reply = { raw: rawReply } as any;

  return { rawRequest, rawReply, signal: createRequestSignal(request, reply) };
}

test('stays live when the request body finishes arriving', () => {
  const { rawRequest, signal } = harness();

  rawRequest.complete = true;
  rawRequest.emit('close');

  assert.equal(signal.aborted, false);
});

test('stays live when the response closes after being fully written', () => {
  const { rawReply, signal } = harness();

  rawReply.writableEnded = true;
  rawReply.emit('close');

  assert.equal(signal.aborted, false);
});

test('aborts when the client disappears mid-upload', () => {
  const { rawRequest, signal } = harness();

  rawRequest.complete = false;
  rawRequest.emit('close');

  assert.equal(signal.aborted, true);
});

test('aborts when the response closes before it was written', () => {
  const { rawReply, signal } = harness();

  rawReply.writableEnded = false;
  rawReply.emit('close');

  assert.equal(signal.aborted, true);
});

test('aborts on an explicit request abort', () => {
  const { rawRequest, signal } = harness();

  rawRequest.emit('aborted');

  assert.equal(signal.aborted, true);
});

test('survives a full healthy request lifecycle', () => {
  const { rawRequest, rawReply, signal } = harness();

  rawRequest.complete = true;
  rawRequest.emit('close');
  rawReply.writableEnded = true;
  rawReply.emit('close');

  assert.equal(signal.aborted, false);
});
