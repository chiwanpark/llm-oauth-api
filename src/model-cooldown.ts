import type { Model } from '@earendil-works/pi-ai';

/**
 * How long a model that failed to respond is passed over.
 *
 * The failures groups exist for — rate limits, quota exhaustion, provider
 * outages — persist for a while, so retrying the same model on every request
 * only spends the client's latency budget to learn what the last request
 * already established.
 */
export const DEFAULT_MODEL_COOLDOWN_MS = 300_000;

/** A model that is being passed over, and why. */
export type SkippedModel = {
  model: Model<any>;
  /** The failure that started the cooldown. */
  reason: string;
  /** How much of the cooldown window is left. */
  remainingMs: number;
};

export type ModelCooldown = {
  /** Zero when cooldowns are disabled. */
  readonly cooldownMs: number;
  /** Starts (or restarts) the cooldown for a model that failed to respond. */
  record(model: Model<any>, reason: string): void;
  /**
   * Forgets a model's failure.
   *
   * Every successful response calls this, whatever the request asked for, so a
   * model that is working again is never passed over for the rest of the window
   * on the strength of one stale failure.
   */
  clear(model: Model<any>): void;
  /**
   * Splits candidates into the ones to try and the ones to pass over.
   *
   * Cooling models are only skipped while some other candidate can serve the
   * request: taking the group offline for the rest of the window would outlast
   * the outage it is reacting to, so a request with nothing ready tries every
   * candidate and doubles as the probe that ends the cooldown.
   *
   * This is also what keeps a request that names one model working. It resolves
   * to a single candidate with nothing to fall back to, so it is always
   * attempted, cooling or not, and its own result is what updates the record.
   */
  select(candidates: readonly Model<any>[]): { attempts: Model<any>[]; skipped: SkippedModel[] };
};

export type ModelCooldownOptions = {
  /** Length of the window. Zero or less disables skipping entirely. */
  cooldownMs: number;
  now?: () => number;
};

export function createModelCooldown(options: ModelCooldownOptions): ModelCooldown {
  const cooldownMs = options.cooldownMs > 0 ? options.cooldownMs : 0;
  const now = options.now ?? Date.now;
  // Keyed by provider and model because a group routes to one named model per
  // provider; the same model id failing on one provider says nothing about the
  // others, and one provider can serve several models.
  const cooling = new Map<string, { until: number; reason: string }>();

  const remaining = (model: Model<any>): { reason: string; remainingMs: number } | undefined => {
    const key = modelKey(model);
    const entry = cooling.get(key);
    if (!entry) return undefined;

    const remainingMs = entry.until - now();
    if (remainingMs <= 0) {
      cooling.delete(key);
      return undefined;
    }
    return { reason: entry.reason, remainingMs };
  };

  return {
    cooldownMs,

    record(model, reason) {
      if (!cooldownMs) return;
      cooling.set(modelKey(model), { until: now() + cooldownMs, reason });
    },

    clear(model) {
      cooling.delete(modelKey(model));
    },

    select(candidates) {
      if (!cooldownMs) return { attempts: [...candidates], skipped: [] };

      const skipped: SkippedModel[] = [];
      const attempts = candidates.filter((model) => {
        const entry = remaining(model);
        if (!entry) return true;
        skipped.push({ model, ...entry });
        return false;
      });

      return attempts.length ? { attempts, skipped } : { attempts: [...candidates], skipped: [] };
    },
  };
}

/** Explains a skip in logs and in the aggregated group error. */
export function describeSkip(skip: SkippedModel): string {
  return `skipped for another ${formatDuration(skip.remainingMs)} after: ${skip.reason}`;
}

function modelKey(model: Model<any>): string {
  return `${model.provider}:${model.id}`;
}

/** Cooldowns are configured in seconds, so report what is left the same way. */
function formatDuration(milliseconds: number): string {
  const seconds = Math.ceil(milliseconds / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

/**
 * Parses a seconds-based cooldown option. Zero disables skipping, which is the
 * only way to ask for the previous behavior of always retrying every member.
 */
export function parseCooldownSeconds(value: string, optionName: string): number {
  const seconds = Number(value);
  const milliseconds = seconds * 1000;

  if (!Number.isFinite(seconds) || seconds < 0 || !Number.isFinite(milliseconds)) {
    throw new Error(`${optionName} must be a non-negative number of seconds (0 disables it)`);
  }

  return milliseconds;
}
