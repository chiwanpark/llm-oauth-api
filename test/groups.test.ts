import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { Model, MutableModels } from '@earendil-works/pi-ai';

import {
  GROUP_ENV_PREFIX,
  describeGroup,
  findGroup,
  groupModelEntries,
  parseModelGroups,
  resolveModelCandidates,
  type ModelGroup,
} from '../src/groups.js';
import { getSupportedProviderIds, resolveSupportedProviderIds } from '../src/providers.js';

const allProviders = getSupportedProviderIds();

function model(provider: string, id: string): Model<any> {
  return { provider, id, api: 'openai-completions' } as Model<any>;
}

function modelsWith(entries: readonly Model<any>[]): MutableModels {
  return {
    getModels(providerId?: string) {
      return providerId ? entries.filter((entry) => entry.provider === providerId) : entries;
    },
  } as unknown as MutableModels;
}

// The whole point of model-level groups: the same model is named differently
// by each provider.
const freeGroup: ModelGroup = {
  name: 'free',
  members: [
    { providerId: 'github-copilot', modelId: 'gpt-5.4-mini' },
    { providerId: 'openai-codex', modelId: 'gpt-5-mini' },
  ],
};

test('parses per-provider model names in declaration order', () => {
  const groups = parseModelGroups(
    { [`${GROUP_ENV_PREFIX}FREE`]: 'github-copilot:gpt-5.4-mini, openai-codex:gpt-5-mini' },
    allProviders,
  );

  assert.deepEqual(groups, [freeGroup]);
});

test('derives hyphenated group names and ignores unrelated variables', () => {
  const groups = parseModelGroups(
    {
      PATH: '/usr/bin',
      LLM_OAUTH_API_KEY: 'secret',
      [`${GROUP_ENV_PREFIX}FAST_TIER`]: 'google:gemini-2.5-pro',
    },
    allProviders,
  );

  assert.deepEqual(groups, [
    { name: 'fast-tier', members: [{ providerId: 'google', modelId: 'gemini-2.5-pro' }] },
  ]);
});

test('keeps distinct models from one provider but drops exact duplicates', () => {
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}FREE`]:
        'google:gemini-2.5-pro,google:gemini-2.5-pro,google:gemini-2.5-flash',
    },
    allProviders,
  );

  assert.deepEqual(groups[0]?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
    { providerId: 'google', modelId: 'gemini-2.5-flash' },
  ]);
});

test('preserves model ids that contain separators', () => {
  const groups = parseModelGroups(
    { [`${GROUP_ENV_PREFIX}LLAMA`]: 'nvidia:meta/llama-3.3-70b-instruct' },
    allProviders,
  );

  assert.deepEqual(groups[0]?.members, [
    { providerId: 'nvidia', modelId: 'meta/llama-3.3-70b-instruct' },
  ]);
});

test('requires a model entry to name both a provider and a model', () => {
  // An entry containing a separator is a model reference, so a missing half is
  // a malformed model rather than a group reference.
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}FREE`]: 'github-copilot:' }, allProviders),
    /must be written as <provider>:<model>/,
  );
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}FREE`]: ':gpt-5.4-mini' }, allProviders),
    /must be written as <provider>:<model>/,
  );
});

test('rejects a bare provider name by pointing at the model syntax', () => {
  // Bare entries are group references now, so a provider name is not one.
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}FREE`]: 'github-copilot' }, allProviders),
    /names a provider, not a group; list a concrete model as <provider>:<model>/,
  );
});

test('rejects a reference to a group that was never declared', () => {
  assert.throws(
    () =>
      parseModelGroups(
        { [`${GROUP_ENV_PREFIX}FREE`]: 'google:gemini-2.5-pro,missing-tier' },
        allProviders,
      ),
    /refers to the group "missing-tier", which is not declared/,
  );
});

test('rejects a group whose name collides with a provider name', () => {
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}GOOGLE`]: 'nvidia:x' }, allProviders),
    /collides with a provider name/,
  );
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}OPENAI_CODEX`]: 'nvidia:x' }, allProviders),
    /collides with a provider name/,
  );
});

test('rejects unknown, empty, and disabled members', () => {
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}FREE`]: 'not-a-provider:x' }, allProviders),
    /unsupported provider: not-a-provider/,
  );
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}FREE`]: '  ,  ' }, allProviders),
    /must list at least one provider:model entry/,
  );
  assert.throws(
    () =>
      parseModelGroups(
        { [`${GROUP_ENV_PREFIX}FREE`]: 'google:a,nvidia:b' },
        resolveSupportedProviderIds(['google']),
      ),
    /"nvidia", which is not enabled/,
  );
});

test('rejects a group name that cannot be used as a model id', () => {
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}BAD.NAME`]: 'google:x' }, allProviders),
    /invalid group name/,
  );
  assert.throws(
    () => parseModelGroups({ [GROUP_ENV_PREFIX]: 'google:x' }, allProviders),
    /does not specify a group name/,
  );
});

test('resolves a group name to each member model in order', () => {
  const copilot = model('github-copilot', 'gpt-5.4-mini');
  const codex = model('openai-codex', 'gpt-5-mini');

  const candidates = resolveModelCandidates(modelsWith([codex, copilot]), [freeGroup], 'free');

  // Group order wins over catalog order, and each provider keeps its own name.
  assert.deepEqual(
    candidates.map((entry) => `${entry.provider}:${entry.id}`),
    ['github-copilot:gpt-5.4-mini', 'openai-codex:gpt-5-mini'],
  );
});

test('skips members whose provider does not publish the named model', () => {
  const candidates = resolveModelCandidates(
    modelsWith([model('openai-codex', 'gpt-5-mini'), model('github-copilot', 'some-other-model')]),
    [freeGroup],
    'free',
  );

  assert.deepEqual(
    candidates.map((entry) => entry.provider),
    ['openai-codex'],
  );
});

test('falls back to plain resolution for non-group references', () => {
  const copilot = model('github-copilot', 'gpt-5.4-mini');
  const catalog = modelsWith([copilot]);

  assert.deepEqual(resolveModelCandidates(catalog, [freeGroup], 'github-copilot:gpt-5.4-mini'), [
    copilot,
  ]);
  assert.deepEqual(resolveModelCandidates(catalog, [freeGroup], 'gpt-5.4-mini'), [copilot]);
  assert.deepEqual(resolveModelCandidates(catalog, [freeGroup], 'nope'), []);
  // The old provider-group syntax must not silently resolve.
  assert.deepEqual(resolveModelCandidates(catalog, [freeGroup], 'free:gpt-5.4-mini'), []);
});

test('matches group names case-insensitively', () => {
  assert.equal(findGroup([freeGroup], 'FREE')?.name, 'free');
  assert.equal(findGroup([freeGroup], ' free ')?.name, 'free');
  assert.equal(findGroup([freeGroup], 'other'), undefined);
});

test('lists a group as one virtual model when a member is available', () => {
  const entries = groupModelEntries(
    [freeGroup],
    [model('openai-codex', 'gpt-5-mini'), model('google', 'gemini-2.5-pro')],
  );

  assert.deepEqual(entries, [{ id: 'free', object: 'model', created: 0, owned_by: 'group' }]);
});

test('hides a group when no member can serve it', () => {
  // Right provider, wrong model id.
  assert.deepEqual(groupModelEntries([freeGroup], [model('github-copilot', 'gpt-4o')]), []);
  assert.deepEqual(groupModelEntries([freeGroup], []), []);
});

test('describes a group as its provider:model members', () => {
  assert.equal(describeGroup(freeGroup), 'github-copilot:gpt-5.4-mini, openai-codex:gpt-5-mini');
});

test('flattens a nested group into its parent in declaration order', () => {
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}FAST`]: 'github-copilot:gpt-5-mini,openai-codex:gpt-5.4-mini',
      [`${GROUP_ENV_PREFIX}ALL`]: 'google:gemini-2.5-pro,fast,nvidia:some-model',
    },
    allProviders,
  );

  // The nested members are spliced in at the position of the reference.
  assert.deepEqual(findGroup(groups, 'all')?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
    { providerId: 'github-copilot', modelId: 'gpt-5-mini' },
    { providerId: 'openai-codex', modelId: 'gpt-5.4-mini' },
    { providerId: 'nvidia', modelId: 'some-model' },
  ]);
  // The nested group stays independently requestable.
  assert.deepEqual(findGroup(groups, 'fast')?.members, [
    { providerId: 'github-copilot', modelId: 'gpt-5-mini' },
    { providerId: 'openai-codex', modelId: 'gpt-5.4-mini' },
  ]);
});

test('resolves references declared in any order', () => {
  // "all" is read before "fast" because variables are processed sorted.
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}ALL`]: 'fast',
      [`${GROUP_ENV_PREFIX}FAST`]: 'google:gemini-2.5-pro',
    },
    allProviders,
  );

  assert.deepEqual(findGroup(groups, 'all')?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
  ]);
});

test('flattens nesting several levels deep', () => {
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}A`]: 'b',
      [`${GROUP_ENV_PREFIX}B`]: 'c',
      [`${GROUP_ENV_PREFIX}C`]: 'google:gemini-2.5-pro',
    },
    allProviders,
  );

  assert.deepEqual(findGroup(groups, 'a')?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
  ]);
});

test('matches nested references by their normalized name', () => {
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}FAST_TIER`]: 'google:gemini-2.5-pro',
      [`${GROUP_ENV_PREFIX}ALL`]: 'FAST_TIER',
      [`${GROUP_ENV_PREFIX}ALSO`]: 'fast-tier',
    },
    allProviders,
  );

  const expected = [{ providerId: 'google', modelId: 'gemini-2.5-pro' }];
  assert.deepEqual(findGroup(groups, 'all')?.members, expected);
  assert.deepEqual(findGroup(groups, 'also')?.members, expected);
});

test('keeps the earliest position when nesting repeats a model', () => {
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}FAST`]: 'google:gemini-2.5-pro,nvidia:some-model',
      [`${GROUP_ENV_PREFIX}ALL`]: 'fast,google:gemini-2.5-pro',
    },
    allProviders,
  );

  assert.deepEqual(findGroup(groups, 'all')?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
    { providerId: 'nvidia', modelId: 'some-model' },
  ]);
});

test('lets two groups share a nested group without duplicating it', () => {
  const groups = parseModelGroups(
    {
      [`${GROUP_ENV_PREFIX}SHARED`]: 'google:gemini-2.5-pro',
      [`${GROUP_ENV_PREFIX}LEFT`]: 'shared,nvidia:a',
      [`${GROUP_ENV_PREFIX}RIGHT`]: 'shared,nvidia:b',
      [`${GROUP_ENV_PREFIX}BOTH`]: 'left,right',
    },
    allProviders,
  );

  // A diamond is not a cycle; the repeated model collapses to one attempt.
  assert.deepEqual(findGroup(groups, 'both')?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
    { providerId: 'nvidia', modelId: 'a' },
    { providerId: 'nvidia', modelId: 'b' },
  ]);
});

test('detects a self-referencing group at startup', () => {
  assert.throws(
    () => parseModelGroups({ [`${GROUP_ENV_PREFIX}LOOP`]: 'loop' }, allProviders),
    /groups form a cycle: loop -> loop/,
  );
});

test('detects a mutual cycle at startup', () => {
  assert.throws(
    () =>
      parseModelGroups(
        { [`${GROUP_ENV_PREFIX}A`]: 'b', [`${GROUP_ENV_PREFIX}B`]: 'a' },
        allProviders,
      ),
    /groups form a cycle: a -> b -> a/,
  );
});

test('detects a deep cycle and reports the path', () => {
  assert.throws(
    () =>
      parseModelGroups(
        {
          [`${GROUP_ENV_PREFIX}A`]: 'google:gemini-2.5-pro,b',
          [`${GROUP_ENV_PREFIX}B`]: 'c',
          [`${GROUP_ENV_PREFIX}C`]: 'b',
        },
        allProviders,
      ),
    /groups form a cycle: b -> c -> b/,
  );
});
