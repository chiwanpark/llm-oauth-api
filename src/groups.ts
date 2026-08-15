import type { Model, MutableModels } from '@earendil-works/pi-ai';

import { resolveModelByName, type OpenAIModelInfo } from './openai-compat.js';
import { resolveProviderId, type SupportedProviderId } from './providers.js';

/** Environment variable prefix that declares a model group. */
export const GROUP_ENV_PREFIX = 'LLM_OAUTH_GROUP_';

/** One concrete upstream model that a group can route to. */
export type GroupMember = {
  providerId: SupportedProviderId;
  modelId: string;
};

/**
 * A virtual model backed by an ordered list of upstream models.
 *
 * Providers name equivalent models differently, so each member names its own
 * model explicitly. The group name is itself the model id clients request; a
 * request for it tries each member in order until one succeeds.
 *
 * Members are always concrete by this point: groups may reference other groups
 * in configuration, but those are flattened during parsing.
 */
export type ModelGroup = {
  name: string;
  members: GroupMember[];
};

/** A member as written in configuration, before nested groups are flattened. */
type RawMember =
  | { kind: 'model'; providerId: SupportedProviderId; modelId: string }
  | { kind: 'group'; name: string; text: string };

type RawGroup = {
  name: string;
  /** Environment variable that declared this group, for error messages. */
  key: string;
  members: RawMember[];
};

/**
 * Derives a group name from an environment variable suffix.
 *
 * `LLM_OAUTH_GROUP_FAST_TIER` becomes `fast-tier` so that group names look like
 * the model ids they sit alongside.
 */
function groupNameFromEnvKey(key: string): string {
  return normalizeGroupName(key.slice(GROUP_ENV_PREFIX.length));
}

/** Group names are written as env suffixes, so accept either separator. */
function normalizeGroupName(value: string): string {
  return value.trim().toLowerCase().replaceAll('_', '-');
}

const VALID_GROUP_NAME = /^[a-z0-9][a-z0-9-]*$/;

/** Splits `provider:model` on the first separator; model ids may contain `:`. */
function parseMember(entry: string): { provider: string; modelId: string } | undefined {
  const separator = entry.indexOf(':');
  if (separator <= 0) return undefined;
  const provider = entry.slice(0, separator).trim();
  const modelId = entry.slice(separator + 1).trim();
  if (!provider || !modelId) return undefined;
  return { provider, modelId };
}

/**
 * Reads `LLM_OAUTH_GROUP_*` variables into validated groups.
 *
 * Every failure is a startup-time configuration error: a silently dropped or
 * partially applied group would surface much later as a confusing "unknown
 * model" or an unexpected provider serving the request.
 */
export function parseModelGroups(
  env: Record<string, string | undefined>,
  enabledProviderIds: readonly SupportedProviderId[],
): ModelGroup[] {
  const raw = parseRawGroups(env, enabledProviderIds);
  assertGroupReferencesExist(raw);
  return flattenGroups(raw);
}

/** First pass: read each variable without resolving references between groups. */
function parseRawGroups(
  env: Record<string, string | undefined>,
  enabledProviderIds: readonly SupportedProviderId[],
): Map<string, RawGroup> {
  const enabled = new Set<string>(enabledProviderIds);
  const groups = new Map<string, RawGroup>();
  const sources = new Map<string, string>();

  for (const key of Object.keys(env).sort()) {
    if (!key.startsWith(GROUP_ENV_PREFIX)) continue;

    const raw = env[key];
    const name = groupNameFromEnvKey(key);

    if (!name) {
      throw new Error(`${key} does not specify a group name (expected ${GROUP_ENV_PREFIX}<NAME>)`);
    }
    if (!VALID_GROUP_NAME.test(name)) {
      throw new Error(
        `${key} defines an invalid group name "${name}"; the part after ` +
          `${GROUP_ENV_PREFIX} may only contain letters, digits, and underscores`,
      );
    }
    if (resolveProviderId(name)) {
      throw new Error(
        `${key} defines the group "${name}", which collides with a provider name; ` +
          'pick a name that reads as a model id',
      );
    }
    const duplicate = sources.get(name);
    if (duplicate) {
      throw new Error(`${key} and ${duplicate} both define the group "${name}"`);
    }

    const entries = (raw ?? '')
      .split(',')
      .map((part) => part.trim())
      .filter(Boolean);
    if (!entries.length) {
      throw new Error(`${key} must list at least one provider:model entry`);
    }

    const members: RawMember[] = entries.map((entry) => {
      // No separator at all means the entry names another group; whether it
      // exists is checked once every variable has been read, so declaration
      // order does not matter. An entry that does contain a separator is a
      // model reference and must be well formed.
      if (!entry.includes(':')) {
        return { kind: 'group', name: normalizeGroupName(entry), text: entry };
      }

      const parsed = parseMember(entry);
      if (!parsed) {
        throw new Error(
          `${key} entry "${entry}" must be written as <provider>:<model>, ` +
            'because providers name the same model differently',
        );
      }

      const providerId = resolveProviderId(parsed.provider);
      if (!providerId) {
        throw new Error(`${key} refers to an unsupported provider: ${parsed.provider}`);
      }
      if (!enabled.has(providerId)) {
        throw new Error(
          `${key} refers to the provider "${providerId}", which is not enabled; ` +
            'add it to --providers or remove it from the group',
        );
      }

      return { kind: 'model', providerId, modelId: parsed.modelId };
    });

    sources.set(name, key);
    groups.set(name, { name, key, members });
  }

  return groups;
}

/** Second pass: every referenced group must exist, in any declaration order. */
function assertGroupReferencesExist(groups: ReadonlyMap<string, RawGroup>): void {
  for (const group of groups.values()) {
    for (const member of group.members) {
      if (member.kind !== 'group' || groups.has(member.name)) continue;

      // A bare provider name is the most likely mistake, so name the fix.
      if (resolveProviderId(member.text)) {
        throw new Error(
          `${group.key} entry "${member.text}" names a provider, not a group; ` +
            'list a concrete model as <provider>:<model>',
        );
      }
      throw new Error(
        `${group.key} refers to the group "${member.name}", which is not declared; ` +
          `define ${GROUP_ENV_PREFIX}${member.name.replaceAll('-', '_').toUpperCase()} ` +
          'or write a concrete model as <provider>:<model>',
      );
    }
  }
}

/**
 * Third pass: splice nested groups into their parents.
 *
 * Flattening once at startup keeps request handling free of recursion and
 * turns cycle detection into a by-product of the walk, so a cyclic
 * configuration can never reach a request.
 */
function flattenGroups(groups: ReadonlyMap<string, RawGroup>): ModelGroup[] {
  const resolved = new Map<string, GroupMember[]>();
  const visiting: string[] = [];

  const visit = (name: string): GroupMember[] => {
    const done = resolved.get(name);
    if (done) return done;

    const cycleStart = visiting.indexOf(name);
    if (cycleStart !== -1) {
      const cycle = [...visiting.slice(cycleStart), name].join(' -> ');
      throw new Error(`${GROUP_ENV_PREFIX}* groups form a cycle: ${cycle}`);
    }

    const group = groups.get(name)!;
    visiting.push(name);

    const members: GroupMember[] = [];
    const add = (candidate: GroupMember) => {
      // Nesting can surface the same model twice; keep the earliest position so
      // the declared preference order still holds.
      const already = members.some(
        (member) =>
          member.providerId === candidate.providerId && member.modelId === candidate.modelId,
      );
      if (!already) members.push(candidate);
    };

    for (const member of group.members) {
      if (member.kind === 'model') {
        add({ providerId: member.providerId, modelId: member.modelId });
      } else {
        for (const nested of visit(member.name)) add(nested);
      }
    }

    visiting.pop();
    resolved.set(name, members);
    return members;
  };

  return [...groups.keys()].map((name) => ({ name, members: visit(name) }));
}

export function findGroup(groups: readonly ModelGroup[], name: string): ModelGroup | undefined {
  const normalized = name.trim().toLowerCase();
  return groups.find((group) => group.name === normalized);
}

/**
 * Resolves a requested model into the ordered list of models to try.
 *
 * A group name yields one model per member that its provider actually
 * publishes, in declaration order. Anything else resolves to at most one model.
 */
export function resolveModelCandidates(
  models: MutableModels,
  groups: readonly ModelGroup[],
  requested: string,
): Model<any>[] {
  const group = findGroup(groups, requested);
  if (group) {
    return group.members.flatMap((member) => {
      const match = models
        .getModels(member.providerId)
        .find((model) => model.id === member.modelId);
      return match ? [match] : [];
    });
  }

  const single = resolveModelByName(models, requested);
  return single ? [single] : [];
}

/**
 * Builds the synthetic entries advertised by /v1/models.
 *
 * A group is listed as a single virtual model, and only when at least one
 * member can serve it, so the catalog matches what a request would accept.
 */
export function groupModelEntries(
  groups: readonly ModelGroup[],
  availableModels: readonly Model<any>[],
): OpenAIModelInfo[] {
  return groups
    .filter((group) =>
      group.members.some((member) =>
        availableModels.some(
          (model) => model.provider === member.providerId && model.id === member.modelId,
        ),
      ),
    )
    .map((group) => ({
      id: group.name,
      object: 'model' as const,
      created: 0,
      owned_by: 'group',
    }));
}

/** Human-readable member list for configuration errors. */
export function describeGroup(group: ModelGroup): string {
  return group.members.map((member) => `${member.providerId}:${member.modelId}`).join(', ');
}
