import type { CredentialStore, Provider } from '@earendil-works/pi-ai';

export const DEFAULT_OAUTH_REFRESH_INTERVAL_MS = 60_000;
export const DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS = 300_000;

export type OAuthRefreshLogger = {
  error(bindings: Record<string, unknown>, message: string): void;
};

export type OAuthRefreshSweepOptions = {
  credentials: CredentialStore;
  providers: readonly Provider[];
  logger: OAuthRefreshLogger;
  refreshBeforeExpiryMs: number;
  now?: () => number;
};

export type OAuthRefreshSchedulerOptions = OAuthRefreshSweepOptions & {
  intervalMs: number;
};

export type OAuthRefreshScheduler = {
  runNow(): Promise<void>;
  stop(): Promise<void>;
};

export function parseRefreshSeconds(value: string, optionName: string, allowZero: boolean): number {
  const seconds = Number(value);
  const valid = Number.isFinite(seconds) && (allowZero ? seconds >= 0 : seconds > 0);
  const milliseconds = seconds * 1000;

  if (!valid || !Number.isFinite(milliseconds)) {
    const expected = allowZero ? 'a non-negative' : 'a positive';
    throw new Error(`${optionName} must be ${expected} number of seconds`);
  }

  return milliseconds;
}

export async function refreshExpiringOAuthCredentials(
  options: OAuthRefreshSweepOptions,
): Promise<void> {
  assertRefreshBeforeExpiry(options.refreshBeforeExpiryMs);
  await Promise.all(options.providers.map((provider) => refreshProvider(provider, options)));
}

export function startOAuthRefreshScheduler(
  options: OAuthRefreshSchedulerOptions,
): OAuthRefreshScheduler {
  assertInterval(options.intervalMs);
  assertRefreshBeforeExpiry(options.refreshBeforeExpiryMs);

  let stopped = false;
  let timer: NodeJS.Timeout | undefined;
  let currentRun: Promise<void> | undefined;

  const schedule = () => {
    if (stopped) return;
    timer = setTimeout(() => {
      timer = undefined;
      const run = runNow();
      void run.then(schedule, schedule);
    }, options.intervalMs);
    timer.unref();
  };

  const runNow = (): Promise<void> => {
    if (currentRun) return currentRun;
    if (stopped) return Promise.resolve();

    const run = refreshExpiringOAuthCredentials(options);
    currentRun = run;
    const clearCurrentRun = () => {
      if (currentRun === run) currentRun = undefined;
    };
    void run.then(clearCurrentRun, clearCurrentRun);
    return run;
  };

  const initialRun = runNow();
  void initialRun.then(schedule, schedule);

  return {
    runNow,
    async stop() {
      stopped = true;
      if (timer) {
        clearTimeout(timer);
        timer = undefined;
      }
      await currentRun;
    },
  };
}

async function refreshProvider(
  provider: Provider,
  options: OAuthRefreshSweepOptions,
): Promise<void> {
  const oauth = provider.auth.oauth;
  if (!oauth) return;

  let stored;
  try {
    stored = await options.credentials.read(provider.id);
  } catch (error) {
    options.logger.error(
      { err: error, providerId: provider.id },
      'Automatic OAuth credential read failed',
    );
    return;
  }

  const now = options.now ?? Date.now;
  if (
    stored?.type !== 'oauth' ||
    !Number.isFinite(stored.expires) ||
    stored.expires > now() + options.refreshBeforeExpiryMs
  ) {
    return;
  }

  try {
    await options.credentials.modify(provider.id, async (current) => {
      if (
        current?.type !== 'oauth' ||
        !Number.isFinite(current.expires) ||
        current.expires > now() + options.refreshBeforeExpiryMs
      ) {
        return undefined;
      }
      return oauth.refresh(current);
    });
  } catch (error) {
    options.logger.error(
      { err: error, providerId: provider.id },
      'Automatic OAuth credential refresh failed; will retry',
    );
  }
}

function assertInterval(intervalMs: number): void {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
    throw new Error('OAuth refresh interval must be a positive number of milliseconds');
  }
}

function assertRefreshBeforeExpiry(refreshBeforeExpiryMs: number): void {
  if (!Number.isFinite(refreshBeforeExpiryMs) || refreshBeforeExpiryMs < 0) {
    throw new Error('OAuth pre-expiry window must be a non-negative number of milliseconds');
  }
}
