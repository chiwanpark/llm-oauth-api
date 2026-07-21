import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import type {
  Credential,
  CredentialInfo,
  CredentialStore,
  OAuthCredential,
  Provider,
} from '@earendil-works/pi-ai';

import { JsonCredentialStore } from '../src/credential-store.js';
import {
  parseRefreshSeconds,
  refreshExpiringOAuthCredentials,
  startOAuthRefreshScheduler,
  type OAuthRefreshLogger,
} from '../src/oauth-refresh.js';

const silentLogger: OAuthRefreshLogger = {
  error() {},
};

class MemoryCredentialStore implements CredentialStore {
  readonly values = new Map<string, Credential>();
  readCalls = 0;
  beforeModify?: () => void | Promise<void>;

  constructor(entries: Record<string, Credential> = {}) {
    for (const [providerId, credential] of Object.entries(entries)) {
      this.values.set(providerId, credential);
    }
  }

  async read(providerId: string): Promise<Credential | undefined> {
    this.readCalls += 1;
    return this.values.get(providerId);
  }

  async list(): Promise<readonly CredentialInfo[]> {
    return [...this.values].map(([providerId, credential]) => ({
      providerId,
      type: credential.type,
    }));
  }

  async modify(
    providerId: string,
    fn: (current: Credential | undefined) => Promise<Credential | undefined>,
  ): Promise<Credential | undefined> {
    await this.beforeModify?.();
    const next = await fn(this.values.get(providerId));
    if (next !== undefined) this.values.set(providerId, next);
    return this.values.get(providerId);
  }

  async delete(providerId: string): Promise<void> {
    this.values.delete(providerId);
  }
}

function oauthProvider(
  id: string,
  refresh: (credential: OAuthCredential) => Promise<OAuthCredential>,
): Provider {
  return {
    id,
    name: id,
    auth: {
      oauth: {
        name: id,
        async login() {
          throw new Error('not implemented in test');
        },
        refresh,
        async toAuth(credential) {
          return { apiKey: credential.access };
        },
      },
    },
  } as Provider;
}

function oauthCredential(expires: number, suffix = 'old'): OAuthCredential {
  return {
    type: 'oauth',
    access: `access-${suffix}`,
    refresh: `refresh-${suffix}`,
    expires,
  };
}

test('parseRefreshSeconds accepts positive intervals and zero pre-expiry windows', () => {
  assert.equal(parseRefreshSeconds('30', '--interval', false), 30_000);
  assert.equal(parseRefreshSeconds('0', '--before-expiry', true), 0);
  assert.throws(
    () => parseRefreshSeconds('0', '--interval', false),
    /--interval must be a positive number of seconds/,
  );
  assert.throws(
    () => parseRefreshSeconds('-1', '--before-expiry', true),
    /--before-expiry must be a non-negative number of seconds/,
  );
  assert.throws(() => parseRefreshSeconds('later', '--interval', false));
});

test('refreshes and persists a credential inside the pre-expiry window', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'llm-oauth-refresh-'));
  const authFile = join(directory, 'auth.json');
  const credentials = new JsonCredentialStore(authFile);
  const initial = oauthCredential(1_100);
  const refreshed = oauthCredential(10_000, 'new');
  let refreshCalls = 0;

  try {
    await credentials.modify('oauth-test', async () => initial);
    await refreshExpiringOAuthCredentials({
      credentials,
      providers: [
        oauthProvider('oauth-test', async () => {
          refreshCalls += 1;
          return refreshed;
        }),
      ],
      logger: silentLogger,
      refreshBeforeExpiryMs: 200,
      now: () => 1_000,
    });

    assert.equal(refreshCalls, 1);
    assert.deepEqual(await credentials.read('oauth-test'), refreshed);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('skips valid, API-key, missing, and non-OAuth provider credentials', async () => {
  const credentials = new MemoryCredentialStore({
    future: oauthCredential(5_000),
    api: { type: 'api_key', key: 'test-key' },
    'no-oauth': oauthCredential(900),
  });
  let refreshCalls = 0;
  const refresh = async (credential: OAuthCredential) => {
    refreshCalls += 1;
    return credential;
  };
  const noOAuthProvider = {
    id: 'no-oauth',
    name: 'no-oauth',
    auth: { apiKey: {} },
  } as unknown as Provider;

  await refreshExpiringOAuthCredentials({
    credentials,
    providers: [
      oauthProvider('future', refresh),
      oauthProvider('api', refresh),
      oauthProvider('missing', refresh),
      noOAuthProvider,
    ],
    logger: silentLogger,
    refreshBeforeExpiryMs: 100,
    now: () => 1_000,
  });

  assert.equal(refreshCalls, 0);
});

test('rechecks expiry under the store lock before refreshing', async () => {
  const credentials = new MemoryCredentialStore({ provider: oauthCredential(900) });
  const alreadyRefreshed = oauthCredential(10_000, 'concurrent');
  let refreshCalls = 0;
  credentials.beforeModify = () => {
    credentials.values.set('provider', alreadyRefreshed);
    credentials.beforeModify = undefined;
  };

  await refreshExpiringOAuthCredentials({
    credentials,
    providers: [
      oauthProvider('provider', async (credential) => {
        refreshCalls += 1;
        return credential;
      }),
    ],
    logger: silentLogger,
    refreshBeforeExpiryMs: 0,
    now: () => 1_000,
  });

  assert.equal(refreshCalls, 0);
  assert.deepEqual(await credentials.read('provider'), alreadyRefreshed);
});

test('isolates provider failures and preserves the failed credential', async () => {
  const failedCredential = oauthCredential(900, 'failed');
  const credentials = new MemoryCredentialStore({
    failed: failedCredential,
    successful: oauthCredential(900),
  });
  const refreshed = oauthCredential(10_000, 'successful');
  const loggedProviders: unknown[] = [];
  const logger: OAuthRefreshLogger = {
    error(bindings) {
      loggedProviders.push(bindings.providerId);
    },
  };

  await refreshExpiringOAuthCredentials({
    credentials,
    providers: [
      oauthProvider('failed', async () => {
        throw new Error('refresh rejected');
      }),
      oauthProvider('successful', async () => refreshed),
    ],
    logger,
    refreshBeforeExpiryMs: 0,
    now: () => 1_000,
  });

  assert.deepEqual(credentials.values.get('failed'), failedCredential);
  assert.deepEqual(credentials.values.get('successful'), refreshed);
  assert.deepEqual(loggedProviders, ['failed']);
});

test('scheduler shares an in-flight run and stops without scheduling more work', async () => {
  const credentials = new MemoryCredentialStore({ provider: oauthCredential(900) });
  let refreshCalls = 0;
  let markStarted!: () => void;
  let finishRefresh!: (credential: OAuthCredential) => void;
  const started = new Promise<void>((resolve) => {
    markStarted = resolve;
  });
  const pendingRefresh = new Promise<OAuthCredential>((resolve) => {
    finishRefresh = resolve;
  });

  const scheduler = startOAuthRefreshScheduler({
    credentials,
    providers: [
      oauthProvider('provider', async () => {
        refreshCalls += 1;
        markStarted();
        return pendingRefresh;
      }),
    ],
    logger: silentLogger,
    intervalMs: 5,
    refreshBeforeExpiryMs: 0,
    now: () => 1_000,
  });

  await started;
  const first = scheduler.runNow();
  const second = scheduler.runNow();
  assert.strictEqual(first, second);
  assert.equal(refreshCalls, 1);

  const stop = scheduler.stop();
  finishRefresh(oauthCredential(10_000, 'new'));
  await stop;
  const readsAfterStop = credentials.readCalls;
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.equal(refreshCalls, 1);
  assert.equal(credentials.readCalls, readsAfterStop);
});
