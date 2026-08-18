import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import type { Model, MutableModels } from '@earendil-works/pi-ai';

import {
  describeGroup,
  findGroup,
  groupModelEntries,
  loadModelGroups,
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

function tempDir(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), 'loa-groups-'));
}

async function writeGroupsFile(config: unknown): Promise<string> {
  const file = path.join(await tempDir(), 'groups.json');
  await writeFile(file, JSON.stringify(config, null, 2), 'utf8');
  return file;
}

function escapeRegExp(value: string): string {
  return value.replaceAll(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
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
    { free: ['github-copilot:gpt-5.4-mini', 'openai-codex:gpt-5-mini'] },
    allProviders,
  );

  assert.deepEqual(groups, [freeGroup]);
});

test('normalizes a declared group name into a model id', () => {
  // A name is written however the author likes; it is read as one model id.
  const groups = parseModelGroups({ FAST_TIER: ['google:gemini-2.5-pro'] }, allProviders);

  assert.deepEqual(groups, [
    { name: 'fast-tier', members: [{ providerId: 'google', modelId: 'gemini-2.5-pro' }] },
  ]);
});

test('keeps groups in the order they are declared in the file', () => {
  const groups = parseModelGroups(
    {
      smart: ['google:gemini-2.5-pro'],
      free: ['nvidia:some-model'],
    },
    allProviders,
  );

  assert.deepEqual(
    groups.map((group) => group.name),
    ['smart', 'free'],
  );
});

test('keeps distinct models from one provider but drops exact duplicates', () => {
  const groups = parseModelGroups(
    {
      free: ['google:gemini-2.5-pro', 'google:gemini-2.5-pro', 'google:gemini-2.5-flash'],
    },
    allProviders,
  );

  assert.deepEqual(groups[0]?.members, [
    { providerId: 'google', modelId: 'gemini-2.5-pro' },
    { providerId: 'google', modelId: 'gemini-2.5-flash' },
  ]);
});

test('preserves model ids that contain separators', () => {
  const groups = parseModelGroups({ llama: ['nvidia:meta/llama-3.3-70b-instruct'] }, allProviders);

  assert.deepEqual(groups[0]?.members, [
    { providerId: 'nvidia', modelId: 'meta/llama-3.3-70b-instruct' },
  ]);
});

test('keeps a variant suffix that is part of the model id', () => {
  // OpenRouter names variants with a second colon, so only the first one may
  // be read as the provider separator.
  const groups = parseModelGroups(
    { cheap: ['openrouter:deepseek/deepseek-r1:free'] },
    allProviders,
  );

  assert.deepEqual(groups[0]?.members, [
    { providerId: 'openrouter', modelId: 'deepseek/deepseek-r1:free' },
  ]);
});

test('routes a group to an OpenRouter model whose id contains a colon', () => {
  const variant = model('openrouter', 'deepseek/deepseek-r1:free');
  const candidates = resolveModelCandidates(
    modelsWith([variant]),
    [
      {
        name: 'cheap',
        members: [{ providerId: 'openrouter', modelId: 'deepseek/deepseek-r1:free' }],
      },
    ],
    'cheap',
  );

  assert.deepEqual(candidates, [variant]);
});

test('requires a model entry to name both a provider and a model', () => {
  // An entry containing a separator is a model reference, so a missing half is
  // a malformed model rather than a group reference.
  assert.throws(
    () => parseModelGroups({ free: ['github-copilot:'] }, allProviders),
    /must be written as <provider>:<model>/,
  );
  assert.throws(
    () => parseModelGroups({ free: [':gpt-5.4-mini'] }, allProviders),
    /must be written as <provider>:<model>/,
  );
});

test('rejects a bare provider name by pointing at the model syntax', () => {
  // Bare entries are group references now, so a provider name is not one.
  assert.throws(
    () => parseModelGroups({ free: ['github-copilot'] }, allProviders),
    /names a provider, not a group; list a concrete model as <provider>:<model>/,
  );
});

test('rejects a reference to a group that was never declared', () => {
  assert.throws(
    () => parseModelGroups({ free: ['google:gemini-2.5-pro', 'missing-tier'] }, allProviders),
    /refers to the group "missing-tier", which is not declared/,
  );
});

test('rejects a group whose name collides with a provider name', () => {
  assert.throws(
    () => parseModelGroups({ google: ['nvidia:x'] }, allProviders),
    /collides with a provider name/,
  );
  // The collision is judged on the normalized name, not the raw key.
  assert.throws(
    () => parseModelGroups({ OPENAI_CODEX: ['nvidia:x'] }, allProviders),
    /collides with a provider name/,
  );
});

test('rejects unknown, empty, and disabled members', () => {
  assert.throws(
    () => parseModelGroups({ free: ['not-a-provider:x'] }, allProviders),
    /unsupported provider: not-a-provider/,
  );
  assert.throws(
    () => parseModelGroups({ free: ['  ', ' '] }, allProviders),
    /must list at least one provider:model entry/,
  );
  assert.throws(
    () => parseModelGroups({ free: [] }, allProviders),
    /must list at least one provider:model entry/,
  );
  assert.throws(
    () =>
      parseModelGroups({ free: ['google:a', 'nvidia:b'] }, resolveSupportedProviderIds(['google'])),
    /"nvidia", which is not enabled/,
  );
});

test('rejects a group name that cannot be used as a model id', () => {
  assert.throws(
    () => parseModelGroups({ 'BAD.NAME': ['google:x'] }, allProviders),
    /invalid group name/,
  );
  assert.throws(() => parseModelGroups({ '': ['google:x'] }, allProviders), /an empty name/);
  assert.throws(() => parseModelGroups({ '  ': ['google:x'] }, allProviders), /an empty name/);
});

test('rejects two keys that name the same group', () => {
  assert.throws(
    () =>
      parseModelGroups(
        { FAST_TIER: ['google:gemini-2.5-pro'], 'fast-tier': ['nvidia:some-model'] },
        allProviders,
      ),
    /declares the group "fast-tier" twice/,
  );
});

test('rejects a file that is not an object of member arrays', () => {
  for (const config of [null, 'free', 42, ['google:x']]) {
    assert.throws(
      () => parseModelGroups(config, allProviders),
      /must contain a JSON object mapping group names to arrays/,
    );
  }

  // A single string is the shape the old environment variables used.
  assert.throws(
    () => parseModelGroups({ free: 'google:gemini-2.5-pro' }, allProviders),
    /must be an array of "<provider>:<model>" strings/,
  );
  assert.throws(
    () => parseModelGroups({ free: ['google:gemini-2.5-pro', 7] }, allProviders),
    /must be an array of "<provider>:<model>" strings/,
  );
});

test('rejects a comma-separated entry by pointing at the array syntax', () => {
  assert.throws(
    () => parseModelGroups({ free: ['google:gemini-2.5-pro,nvidia:some-model'] }, allProviders),
    /contains a comma; list each member as its own array element/,
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
      fast: ['github-copilot:gpt-5-mini', 'openai-codex:gpt-5.4-mini'],
      all: ['google:gemini-2.5-pro', 'fast', 'nvidia:some-model'],
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
  // "all" refers to "fast" before the file declares it.
  const groups = parseModelGroups(
    {
      all: ['fast'],
      fast: ['google:gemini-2.5-pro'],
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
      a: ['b'],
      b: ['c'],
      c: ['google:gemini-2.5-pro'],
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
      'fast-tier': ['google:gemini-2.5-pro'],
      all: ['FAST_TIER'],
      also: ['fast-tier'],
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
      fast: ['google:gemini-2.5-pro', 'nvidia:some-model'],
      all: ['fast', 'google:gemini-2.5-pro'],
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
      shared: ['google:gemini-2.5-pro'],
      left: ['shared', 'nvidia:a'],
      right: ['shared', 'nvidia:b'],
      both: ['left', 'right'],
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
    () => parseModelGroups({ loop: ['loop'] }, allProviders),
    /form a cycle: loop -> loop/,
  );
});

test('detects a mutual cycle at startup', () => {
  assert.throws(
    () => parseModelGroups({ a: ['b'], b: ['a'] }, allProviders),
    /form a cycle: a -> b -> a/,
  );
});

test('detects a deep cycle and reports the path', () => {
  assert.throws(
    () =>
      parseModelGroups(
        {
          a: ['google:gemini-2.5-pro', 'b'],
          b: ['c'],
          c: ['b'],
        },
        allProviders,
      ),
    /form a cycle: b -> c -> b/,
  );
});

test('reads groups from a JSON file', async () => {
  const file = await writeGroupsFile({
    fast: ['github-copilot:gpt-5-mini'],
    all: ['fast', 'google:gemini-2.5-pro'],
  });

  assert.deepEqual(await loadModelGroups(file, allProviders), [
    { name: 'fast', members: [{ providerId: 'github-copilot', modelId: 'gpt-5-mini' }] },
    {
      name: 'all',
      members: [
        { providerId: 'github-copilot', modelId: 'gpt-5-mini' },
        { providerId: 'google', modelId: 'gemini-2.5-pro' },
      ],
    },
  ]);
});

test('names the file in a configuration error', async () => {
  const file = await writeGroupsFile({ free: ['not-a-provider:x'] });

  await assert.rejects(
    () => loadModelGroups(file, allProviders),
    (error: Error) =>
      error.message.startsWith(file) && /unsupported provider: not-a-provider/.test(error.message),
  );
});

test('reports a missing or malformed file instead of starting without groups', async () => {
  const missing = path.join(await tempDir(), 'absent.json');
  await assert.rejects(
    () => loadModelGroups(missing, allProviders),
    new RegExp(`Groups file not found: ${escapeRegExp(missing)}`),
  );

  const broken = path.join(await tempDir(), 'groups.json');
  await writeFile(broken, '{ "free": [ ', 'utf8');
  await assert.rejects(() => loadModelGroups(broken, allProviders), /is not valid JSON/);
});
