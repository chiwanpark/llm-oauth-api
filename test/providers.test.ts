import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { OAuthCredential, Provider } from '@earendil-works/pi-ai';

import {
  createSupportedProvider,
  DEVICE_ID_KEY,
  getSupportedProviderIds,
  withPreservedDeviceId,
} from '../src/providers.js';

const DEVICE_ID = '7c9e6679-7425-40de-944b-e07fc1f90ae7';

function fakeProvider(): Provider {
  return {
    id: 'fake',
    name: 'Fake',
    auth: {
      oauth: {
        name: 'Fake OAuth',
        login: async () => ({ type: 'oauth', access: 'a0', refresh: 'r0', expires: 0 }),
        refresh: async (credential: OAuthCredential) => ({
          type: 'oauth',
          access: `${credential.access}+`,
          refresh: `${credential.refresh}+`,
          expires: credential.expires + 1,
        }),
        toAuth: async (credential: OAuthCredential) => ({ apiKey: credential.access }),
      },
    },
  } as unknown as Provider;
}

const signal = new AbortController().signal;

test('openai replaces the legacy openai-codex provider', () => {
  const ids = getSupportedProviderIds();
  assert.ok(ids.includes('openai'));
  assert.ok(!ids.includes('openai-codex' as never));
  const provider = createSupportedProvider('openai');
  assert.equal(provider.id, 'openai');
  assert.ok(provider.auth.oauth);
  assert.ok(provider.auth.apiKey);
});

test('a refreshed credential keeps the stored device id', async () => {
  const oauth = withPreservedDeviceId(fakeProvider()).auth.oauth!;
  const refreshed = await oauth.refresh(
    { type: 'oauth', access: 'a0', refresh: 'r0', expires: 0, [DEVICE_ID_KEY]: DEVICE_ID },
    signal,
  );
  assert.deepEqual(refreshed, {
    type: 'oauth',
    access: 'a0+',
    refresh: 'r0+',
    expires: 1,
    [DEVICE_ID_KEY]: DEVICE_ID,
  });
});

test('a credential without a device id is refreshed unchanged', async () => {
  const oauth = withPreservedDeviceId(fakeProvider()).auth.oauth!;
  const refreshed = await oauth.refresh(
    { type: 'oauth', access: 'a0', refresh: 'r0', expires: 0 },
    signal,
  );
  assert.equal(DEVICE_ID_KEY in refreshed, false);
});
