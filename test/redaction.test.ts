import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { stringify as stringifyYaml } from 'yaml';

import type { AssistantMessage, Context, Model, MutableModels } from '@earendil-works/pi-ai';

import type { ModelGroup } from '../src/groups.js';
import { getSupportedProviderIds } from '../src/providers.js';
import {
  loadRedactionConfig,
  NO_REDACTION,
  parseRedactionConfig,
  type RedactionTarget,
  type Redactor,
} from '../src/redaction.js';
import { chatAdapter, runCompletion } from '../src/server.js';

const allProviders = getSupportedProviderIds();

const freeGroup: ModelGroup = {
  name: 'free',
  members: [
    { providerId: 'github-copilot', modelId: 'gpt-5.4-mini' },
    { providerId: 'openai-codex', modelId: 'gpt-5-mini' },
  ],
};

const KEY = 'sk-abcdefghijklmnopqrstuvwx';

function build(rules: unknown, groups: readonly ModelGroup[] = [freeGroup]): Redactor {
  return parseRedactionConfig({ rules }, allProviders, groups);
}

/** A file with an explicit scope; `models` is the whole file's, not a rule's. */
function scoped(
  models: unknown,
  rules: unknown,
  groups: readonly ModelGroup[] = [freeGroup],
): Redactor {
  return parseRedactionConfig({ models, rules }, allProviders, groups);
}

function openaiKeyRule(extra: Record<string, unknown> = {}) {
  return { name: 'openai-key', pattern: String.raw`sk-[A-Za-z0-9]{16,}`, ...extra };
}

function target(providerId: string, modelId: string, requestedModel?: string): RedactionTarget {
  return { providerId, modelId, ...(requestedModel ? { requestedModel } : {}) };
}

function textContext(text: string): Context {
  return { messages: [{ role: 'user', content: text, timestamp: 1 }] };
}

function userText(context: Context): string {
  const message = context.messages[0];
  assert.equal(message.role, 'user');
  return message.role === 'user' && typeof message.content === 'string' ? message.content : '';
}

async function writeRedactionFile(config: unknown): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), 'loa-redaction-'));
  const file = path.join(directory, 'redaction.yaml');
  await writeFile(file, stringifyYaml(config), 'utf8');
  return file;
}

test('masks a matching credential and leaves the rest of the text alone', () => {
  const redactor = build([openaiKeyRule()]);
  const { context, summary } = redactor.redact(
    textContext(`use ${KEY} to call the api`),
    target('anthropic', 'claude-sonnet-4-5'),
  );

  assert.equal(userText(context), 'use [REDACTED:openai-key] to call the api');
  assert.equal(summary.total, 1);
  assert.deepEqual(summary.byRule, { 'openai-key': 1 });
});

test('a custom replacement is used verbatim', () => {
  const redactor = build([openaiKeyRule({ replacement: '<hidden>' })]);
  const { context } = redactor.redact(textContext(KEY), target('anthropic', 'claude-sonnet-4-5'));
  assert.equal(userText(context), '<hidden>');
});

/**
 * The mask, like the scope, is usually a property of the whole file, so it can
 * be stated once. A rule may still override it, because which text stands in for
 * a secret can genuinely differ between one credential shape and the next.
 */

function masked(config: Record<string, unknown>, rules: unknown): string {
  const redactor = parseRedactionConfig({ ...config, rules }, allProviders, [freeGroup]);
  const { context } = redactor.redact(textContext(KEY), target('anthropic', 'claude-sonnet-4-5'));
  return userText(context);
}

test('a top-level replacement is the default mask', () => {
  assert.equal(masked({ replacement: '<hidden>' }, [openaiKeyRule()]), '<hidden>');
});

test('a rule replacement overrides the top-level one', () => {
  assert.equal(
    masked({ replacement: '<hidden>' }, [openaiKeyRule({ replacement: '<mine>' })]),
    '<mine>',
  );
});

test('{name} in a top-level replacement expands to the rule name', () => {
  assert.equal(masked({ replacement: '<<{name}>>' }, [openaiKeyRule()]), '<<openai-key>>');
});

test('{name} expands in a rule replacement too', () => {
  assert.equal(masked({}, [openaiKeyRule({ replacement: '[{name}]' })]), '[openai-key]');
});

test('the built-in default names the rule', () => {
  assert.equal(masked({}, [openaiKeyRule()]), '[REDACTED:openai-key]');
});

test('a top-level replacement applies to every rule that omits one', () => {
  const redactor = parseRedactionConfig(
    {
      replacement: '<{name}>',
      rules: [openaiKeyRule(), { name: 'pw', pattern: 'hunter2', replacement: 'x' }],
    },
    allProviders,
  );
  const { context } = redactor.redact(
    textContext(`${KEY} hunter2`),
    target('anthropic', 'claude-sonnet-4-5'),
  );
  assert.equal(userText(context), '<openai-key> x');
});

test('an empty replacement deletes the match', () => {
  assert.equal(masked({ replacement: '' }, [openaiKeyRule()]), '');
});

test('rejects a non-string top-level replacement', () => {
  assert.throws(
    () => parseRedactionConfig({ replacement: 5, rules: [openaiKeyRule()] }, allProviders),
    /"replacement" must be a string/,
  );
});

test('every occurrence is masked, not just the first', () => {
  const redactor = build([openaiKeyRule()]);
  const { summary } = redactor.redact(
    textContext(`${KEY} and ${KEY} and ${KEY}`),
    target('anthropic', 'claude-sonnet-4-5'),
  );
  assert.equal(summary.total, 3);
});

/**
 * Credentials are usually recognised by what surrounds them, and that context is
 * what keeps the rest of the message readable. `captureGroup` lets a pattern name
 * the context and still mask only the secret inside it.
 */

function maskedText(rule: Record<string, unknown>, text: string): string {
  const redactor = build([rule]);
  const { context } = redactor.redact(textContext(text), target('anthropic', 'claude-sonnet-4-5'));
  return userText(context);
}

const ENV_RULE = {
  name: 'env-secret',
  pattern: String.raw`([A-Z0-9_]*(?:API_KEY|PASSWORD)\s*=\s*)(\S+)`,
  captureGroup: 2,
};

test('captureGroup masks the value and keeps the variable name', () => {
  assert.equal(
    maskedText(ENV_RULE, 'export MY_APP_API_KEY=abc123secret'),
    'export MY_APP_API_KEY=[REDACTED:env-secret]',
  );
});

test('without captureGroup the whole match goes, context included', () => {
  assert.equal(
    maskedText({ name: 'env-secret', pattern: ENV_RULE.pattern }, 'export MY_APP_API_KEY=abc'),
    'export [REDACTED:env-secret]',
  );
});

test('captureGroup masks every occurrence on a line', () => {
  assert.equal(
    maskedText(ENV_RULE, 'DB_PASSWORD=p@ss STRIPE_API_KEY=sk_live_9999'),
    'DB_PASSWORD=[REDACTED:env-secret] STRIPE_API_KEY=[REDACTED:env-secret]',
  );
});

test('captureGroup masks the password in a JDBC url', () => {
  const rule = {
    name: 'jdbc',
    pattern: String.raw`(jdbc:[a-z]+://[^:/@\s]+:)([^@\s]+)(@)`,
    captureGroup: 2,
  };
  assert.equal(
    maskedText(rule, 'jdbc:postgresql://dbuser:s3cr3tpw@db.host:5432/app'),
    'jdbc:postgresql://dbuser:[REDACTED:jdbc]@db.host:5432/app',
  );
});

test('captureGroup masks a password query parameter', () => {
  const rule = {
    name: 'jdbcqp',
    pattern: String.raw`([?&]password=)([^&\s]+)`,
    captureGroup: 2,
  };
  assert.equal(
    maskedText(rule, 'jdbc:mysql://db/app?user=root&password=hunter2&ssl=true'),
    'jdbc:mysql://db/app?user=root&password=[REDACTED:jdbcqp]&ssl=true',
  );
});

test('captureGroup 1 works and text outside the match survives', () => {
  assert.equal(
    maskedText({ name: 'g', pattern: '(secret)-(keep)', captureGroup: 1 }, 'a secret-keep b'),
    'a [REDACTED:g]-keep b',
  );
});

test('a repeated context picks the right occurrence to mask', () => {
  // Locating the group by text rather than by position would mask the wrong `x`.
  assert.equal(
    maskedText({ name: 'g', pattern: '(x=)(x)', captureGroup: 2 }, 'x=x'),
    'x=[REDACTED:g]',
  );
});

test('a group that matched nothing is left alone', () => {
  assert.equal(
    maskedText({ name: 'g', pattern: 'keep(secret)?', captureGroup: 1 }, 'keep and keepsecret'),
    'keep and keep[REDACTED:g]',
  );
});

test('captureGroup counts hits per masked group', () => {
  const redactor = build([ENV_RULE]);
  const { summary } = redactor.redact(
    textContext('A_API_KEY=1 B_PASSWORD=2'),
    target('anthropic', 'claude-sonnet-4-5'),
  );
  assert.deepEqual(summary.byRule, { 'env-secret': 2 });
});

test('rejects a captureGroup that is not a whole number of 1 or more', () => {
  for (const value of [0, -1, 1.5, '1', null]) {
    assert.throws(
      () => build([{ name: 'g', pattern: '(a)', captureGroup: value }]),
      /"captureGroup" as a whole number of 1 or more/,
    );
  }
});

test('rejects a captureGroup the pattern does not have', () => {
  assert.throws(
    () => build([{ name: 'g', pattern: '(a)(b)', captureGroup: 3 }]),
    /sets "captureGroup" to 3 but its pattern has 2 capture groups/,
  );
  assert.throws(
    () => build([{ name: 'g', pattern: '(a)', captureGroup: 2 }]),
    /but its pattern has 1 capture group/,
  );
});

test('a non-capturing group does not count toward captureGroup', () => {
  assert.throws(
    () => build([{ name: 'g', pattern: '(?:a)(b)', captureGroup: 2 }]),
    /but its pattern has 1 capture group/,
  );
});

/**
 * `$1` is written out literally rather than expanded, so leaving it in place
 * would produce output that looks redacted but carries a stray `$1`.
 */
test('rejects a replacement containing a capture reference', () => {
  assert.throws(
    () => build([{ name: 'g', pattern: '(a)(b)', replacement: '$1<masked>' }]),
    /containing \$1, which is written out literally/,
  );
  assert.throws(
    () => build([{ name: 'g', pattern: '(a)', replacement: 'x$&y' }]),
    /containing \$&/,
  );
  assert.throws(
    () => parseRedactionConfig({ replacement: '$1', rules: [openaiKeyRule()] }, allProviders),
    /top-level has a "replacement" containing \$1/,
  );
});

test('a plain dollar sign in a replacement is fine', () => {
  assert.equal(maskedText({ name: 'g', pattern: 'a', replacement: '$$$' }, 'a'), '$$$');
});

test('a context with nothing to mask is returned unchanged', () => {
  const redactor = build([openaiKeyRule()]);
  const original = textContext('no secrets here');
  const { context, summary } = redactor.redact(original, target('anthropic', 'claude-sonnet-4-5'));

  assert.equal(userText(context), 'no secrets here');
  assert.equal(summary.total, 0);
  assert.deepEqual(summary.byRule, {});
});

test('the input context is not mutated', () => {
  const redactor = build([openaiKeyRule()]);
  const original = textContext(`key ${KEY}`);
  redactor.redact(original, target('anthropic', 'claude-sonnet-4-5'));
  assert.equal(userText(original), `key ${KEY}`);
});

test('rules apply in declaration order and can stack', () => {
  const redactor = build([
    { name: 'first', pattern: 'alpha', replacement: 'beta' },
    { name: 'second', pattern: 'beta', replacement: 'gamma' },
  ]);
  const { context, summary } = redactor.redact(
    textContext('alpha'),
    target('anthropic', 'claude-sonnet-4-5'),
  );

  assert.equal(userText(context), 'gamma');
  assert.deepEqual(summary.byRule, { first: 1, second: 1 });
});

test('flags are honoured and g is always implied', () => {
  const redactor = build([{ name: 'secret', pattern: 'PASSWORD', flags: 'i' }]);
  const { summary } = redactor.redact(
    textContext('password Password PASSWORD'),
    target('anthropic', 'claude-sonnet-4-5'),
  );
  assert.equal(summary.total, 3);
});

/**
 * The scope is stated once for the whole file, so these check `covers` rather
 * than any per-rule notion of reach.
 */

test('a provider entry covers that provider only', () => {
  const redactor = scoped(['anthropic'], [openaiKeyRule()]);

  assert.equal(redactor.covers(target('anthropic', 'claude-sonnet-4-5')), true);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-pro')), false);
});

test('a provider:model entry matches one model', () => {
  const redactor = scoped(['openai-codex:gpt-5.4'], [openaiKeyRule()]);

  assert.equal(redactor.covers(target('openai-codex', 'gpt-5.4')), true);
  assert.equal(redactor.covers(target('openai-codex', 'gpt-5-mini')), false);
});

test('a model glob matches within one provider', () => {
  const redactor = scoped(['openai-codex:gpt-5*'], [openaiKeyRule()]);

  assert.equal(redactor.covers(target('openai-codex', 'gpt-5.4')), true);
  assert.equal(redactor.covers(target('openai-codex', 'gpt-5-mini')), true);
  assert.equal(redactor.covers(target('openai-codex', 'o3')), false);
});

test('a wildcard provider matches a model id across providers', () => {
  const redactor = scoped(['*:claude-sonnet-4-5'], [openaiKeyRule()]);

  assert.equal(redactor.covers(target('anthropic', 'claude-sonnet-4-5')), true);
  assert.equal(redactor.covers(target('opencode-go', 'claude-sonnet-4-5')), true);
  assert.equal(redactor.covers(target('anthropic', 'claude-haiku-4-5')), false);
});

test('a bare * covers everything', () => {
  const redactor = scoped(['*'], [openaiKeyRule()]);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-pro')), true);
});

test('omitting models covers everything', () => {
  const redactor = build([openaiKeyRule()]);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-pro')), true);
});

test('a group entry follows the requested group across its members', () => {
  const redactor = scoped(['free'], [openaiKeyRule()]);

  // Either member serves the group, so the scope has to hold for both.
  assert.equal(redactor.covers(target('github-copilot', 'gpt-5.4-mini', 'free')), true);
  assert.equal(redactor.covers(target('openai-codex', 'gpt-5-mini', 'free')), true);
  // The same model requested directly is not the group.
  assert.equal(redactor.covers(target('openai-codex', 'gpt-5-mini')), false);
});

test('a group entry is read case-insensitively with _ as -', () => {
  const groups: ModelGroup[] = [{ name: 'fast-tier', members: freeGroup.members }];
  const redactor = scoped(['FAST_TIER'], [openaiKeyRule()], groups);
  assert.equal(redactor.covers(target('openai-codex', 'gpt-5-mini', 'fast-tier')), true);
});

test('several entries are a union', () => {
  const redactor = scoped(['anthropic', 'google:gemini-2.5-pro'], [openaiKeyRule()]);

  assert.equal(redactor.covers(target('anthropic', 'claude-sonnet-4-5')), true);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-pro')), true);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-flash')), false);
});

test('every rule applies inside the scope and none outside it', () => {
  const redactor = scoped(['anthropic'], [openaiKeyRule(), { name: 'pw', pattern: 'hunter2' }]);

  const inside = redactor.redact(
    textContext(`${KEY} hunter2`),
    target('anthropic', 'claude-sonnet-4-5'),
  );
  assert.equal(userText(inside.context), '[REDACTED:openai-key] [REDACTED:pw]');

  const outside = redactor.redact(
    textContext(`${KEY} hunter2`),
    target('google', 'gemini-2.5-pro'),
  );
  assert.equal(userText(outside.context), `${KEY} hunter2`);
  assert.equal(outside.summary.total, 0);
});

test('masks system prompt, user, assistant, and tool result turns', () => {
  const redactor = build([openaiKeyRule()]);
  const context: Context = {
    systemPrompt: `system ${KEY}`,
    messages: [
      { role: 'user', content: `user ${KEY}`, timestamp: 1 },
      {
        role: 'assistant',
        content: [{ type: 'text', text: `assistant ${KEY}` }],
        api: 'openai-completions',
        provider: 'openai-codex',
        model: 'gpt-5.4',
        usage: {} as any,
        stopReason: 'stop',
        timestamp: 2,
      },
      {
        role: 'toolResult',
        toolCallId: 'call_1',
        toolName: 'read_file',
        content: [{ type: 'text', text: `tool ${KEY}` }],
        isError: false,
        timestamp: 3,
      },
    ],
  };

  const { context: out, summary } = redactor.redact(
    context,
    target('anthropic', 'claude-sonnet-4-5'),
  );

  assert.equal(out.systemPrompt, 'system [REDACTED:openai-key]');
  assert.equal(userText(out), 'user [REDACTED:openai-key]');
  const assistant = out.messages[1];
  assert.equal(
    assistant.role === 'assistant' && (assistant.content[0] as any).text,
    'assistant [REDACTED:openai-key]',
  );
  const tool = out.messages[2];
  assert.equal(
    tool.role === 'toolResult' && (tool.content[0] as any).text,
    'tool [REDACTED:openai-key]',
  );
  assert.equal(summary.total, 4);
});

test('masks user text blocks but never image data', () => {
  const redactor = build([openaiKeyRule()]);
  const context: Context = {
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: `look at ${KEY}` },
          { type: 'image', data: 'sk-abcdefghijklmnopqrstuvwx', mimeType: 'image/png' },
        ],
        timestamp: 1,
      },
    ],
  };

  const { context: out, summary } = redactor.redact(
    context,
    target('anthropic', 'claude-sonnet-4-5'),
  );
  const blocks = out.messages[0].role === 'user' ? (out.messages[0].content as any[]) : [];

  assert.equal(blocks[0].text, 'look at [REDACTED:openai-key]');
  // Image bytes are opaque to a credential pattern and rewriting them corrupts the image.
  assert.equal(blocks[1].data, 'sk-abcdefghijklmnopqrstuvwx');
  assert.equal(summary.total, 1);
});

test('masks tool call arguments at any depth while keeping the shape', () => {
  const redactor = build([openaiKeyRule()]);
  const context: Context = {
    messages: [
      {
        role: 'assistant',
        content: [
          {
            type: 'toolCall',
            id: 'call_1',
            name: 'http_request',
            arguments: {
              url: 'https://example.com',
              headers: { authorization: `Bearer ${KEY}` },
              retries: 3,
              tags: [`tag-${KEY}`, 'plain'],
            },
          },
        ],
        api: 'openai-completions',
        provider: 'openai-codex',
        model: 'gpt-5.4',
        usage: {} as any,
        stopReason: 'toolUse',
        timestamp: 1,
      },
    ],
  };

  const { context: out } = redactor.redact(context, target('anthropic', 'claude-sonnet-4-5'));
  const call = (out.messages[0] as any).content[0];

  assert.equal(call.arguments.headers.authorization, 'Bearer [REDACTED:openai-key]');
  assert.equal(call.arguments.tags[0], 'tag-[REDACTED:openai-key]');
  assert.equal(call.arguments.tags[1], 'plain');
  assert.equal(call.arguments.retries, 3);
  assert.equal(call.arguments.url, 'https://example.com');
  // The correlation id pairs the call with its result and must survive intact.
  assert.equal(call.id, 'call_1');
});

test('never rewrites thinking content or its signature', () => {
  const redactor = build([{ name: 'anything', pattern: '[a-z-]+' }]);
  const context: Context = {
    messages: [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'secret reasoning', thinkingSignature: 'sig-abc' },
          { type: 'text', text: 'hello' },
        ],
        api: 'anthropic-messages',
        provider: 'anthropic',
        model: 'claude-sonnet-4-5',
        usage: {} as any,
        stopReason: 'stop',
        timestamp: 1,
      },
    ],
  };

  const { context: out } = redactor.redact(context, target('anthropic', 'claude-sonnet-4-5'));
  const blocks = (out.messages[0] as any).content;

  // Thinking is paired with a signature the provider validates; masking one half breaks it.
  assert.equal(blocks[0].thinking, 'secret reasoning');
  assert.equal(blocks[0].thinkingSignature, 'sig-abc');
  assert.equal(blocks[1].text, '[REDACTED:anything]');
});

test('masks tool descriptions but leaves the parameter schema alone', () => {
  const redactor = build([openaiKeyRule()]);
  const context: Context = {
    messages: [],
    tools: [
      {
        name: 'call_api',
        description: `pass ${KEY} as the token`,
        parameters: { type: 'object', properties: { token: { type: 'string' } } } as any,
      },
    ],
  };

  const { context: out } = redactor.redact(context, target('anthropic', 'claude-sonnet-4-5'));

  assert.equal(out.tools?.[0].description, 'pass [REDACTED:openai-key] as the token');
  assert.equal(out.tools?.[0].name, 'call_api');
  assert.deepEqual(out.tools?.[0].parameters, {
    type: 'object',
    properties: { token: { type: 'string' } },
  });
});

test('NO_REDACTION passes a context through untouched', () => {
  const original = textContext(KEY);
  const { context, summary } = NO_REDACTION.redact(original, target('anthropic', 'claude-4'));

  assert.equal(context, original);
  assert.equal(summary.total, 0);
});

test('rejects a config that is not a mapping with rules', () => {
  assert.throws(() => parseRedactionConfig([], allProviders), /must contain a YAML mapping/);
  assert.throws(() => parseRedactionConfig({}, allProviders), /must contain a "rules" list/);
});

test('rejects a rule with no name or a bad name', () => {
  assert.throws(() => build([{ pattern: 'x' }]), /must have a non-empty "name"/);
  assert.throws(() => build([{ name: 'has space', pattern: 'x' }]), /invalid name/);
});

test('rejects a duplicate rule name', () => {
  assert.throws(
    () => build([openaiKeyRule(), openaiKeyRule()]),
    /declares the rule "openai-key" twice/,
  );
});

test('rejects a missing or invalid pattern', () => {
  assert.throws(() => build([{ name: 'a' }]), /must have a non-empty "pattern"/);
  assert.throws(() => build([{ name: 'a', pattern: '([' }]), /invalid pattern/);
});

test('rejects a pattern that matches the empty string', () => {
  // `.*` would splice the replacement between every character of every message.
  assert.throws(() => build([{ name: 'a', pattern: '.*' }]), /matches the empty string/);
});

test('rejects an unsupported regex flag', () => {
  assert.throws(() => build([openaiKeyRule({ flags: 'y' })]), /unsupported flag "y"/);
});

test('rejects a models entry naming an unknown provider or group', () => {
  assert.throws(() => scoped(['nope:x'], [openaiKeyRule()]), /unknown provider/);
  assert.throws(
    () => scoped(['nope'], [openaiKeyRule()]),
    /neither a known provider nor a declared group/,
  );
});

test('rejects a models entry naming a provider left out of --providers', () => {
  assert.throws(
    () => parseRedactionConfig({ models: ['google'], rules: [openaiKeyRule()] }, ['anthropic']),
    /not enabled/,
  );
});

test('rejects a malformed models list', () => {
  assert.throws(() => scoped('anthropic', [openaiKeyRule()]), /must be an array of strings/);
  assert.throws(() => scoped([], [openaiKeyRule()]), /lists no models/);
});

/**
 * Scope belongs to the file, not to a pattern. A leftover per-rule list is
 * refused outright, because ignoring it would silently widen that rule to every
 * model the file covers.
 */
test('rejects a rule that sets its own models', () => {
  assert.throws(
    () => build([openaiKeyRule({ models: ['anthropic'] })]),
    /rule "openai-key" sets "models", which is not a rule field/,
  );
  assert.throws(
    () => scoped(['anthropic'], [openaiKeyRule({ models: ['*'] })]),
    /move the model list to the top level/,
  );
});

test('loads rules and their scope from a file', async () => {
  const file = await writeRedactionFile({
    models: ['anthropic'],
    rules: [openaiKeyRule(), { name: 'pw', pattern: 'hunter2' }],
  });
  const redactor = await loadRedactionConfig(file, allProviders);

  assert.deepEqual(
    redactor.rules.map((rule) => rule.name),
    ['openai-key', 'pw'],
  );
  assert.equal(redactor.covers(target('anthropic', 'claude-sonnet-4-5')), true);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-pro')), false);
});

test('a file without models covers every model', async () => {
  const file = await writeRedactionFile({ rules: [openaiKeyRule()] });
  const redactor = await loadRedactionConfig(file, allProviders);

  assert.equal(redactor.rules.length, 1);
  assert.equal(redactor.covers(target('google', 'gemini-2.5-pro')), true);
});

test('a missing file is a startup error naming the path', async () => {
  const file = path.join(tmpdir(), 'loa-redaction-missing', 'redaction.yaml');
  await assert.rejects(loadRedactionConfig(file, allProviders), /Redaction file not found/);
});

test('a malformed file is a startup error', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'loa-redaction-'));
  const file = path.join(directory, 'redaction.yaml');
  await writeFile(file, 'rules: [\n  - name: broken', 'utf8');
  await assert.rejects(loadRedactionConfig(file, allProviders), /is not valid YAML/);
});

test('a group entry in a file resolves against the loaded groups', async () => {
  const file = await writeRedactionFile({ models: ['free'], rules: [openaiKeyRule()] });
  const redactor = await loadRedactionConfig(file, allProviders, [freeGroup]);

  assert.equal(redactor.covers(target('github-copilot', 'gpt-5.4-mini', 'free')), true);
  await assert.rejects(loadRedactionConfig(file, allProviders, []), /neither a known provider/);
});

/**
 * The unit tests above check the masking itself; these check that the masked
 * context is the one that actually reaches the provider, and that a fallback to
 * a second provider re-evaluates the scope against the new target.
 */

function fakeRequest(body: unknown) {
  return {
    body,
    log: { info() {}, warn() {}, error() {} },
    raw: { once() {} },
  } as any;
}

function fakeReply() {
  const reply: any = {
    code: () => reply,
    send: () => reply,
    raw: {
      headersSent: false,
      writableEnded: false,
      writeHead() {},
      write: () => true,
      end() {},
      once() {},
    },
  };
  return reply;
}

function fakeModels(
  catalog: readonly Model<any>[],
  onComplete: (model: Model<any>, context: Context) => AssistantMessage,
) {
  const seen: { model: Model<any>; context: Context }[] = [];
  const models = {
    getModels: (providerId?: string) =>
      providerId ? catalog.filter((entry) => entry.provider === providerId) : catalog,
    getAuth: async () => ({ apiKey: 'k' }),
    async completeSimple(model: Model<any>, context: Context) {
      seen.push({ model, context });
      return onComplete(model, context);
    },
  } as unknown as MutableModels;
  return { models, seen };
}

function assistantReply(model: Model<any>, stopReason: 'stop' | 'error'): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text: 'ok' }],
    api: 'openai-completions',
    provider: model.provider,
    model: model.id,
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    ...(stopReason === 'error' ? { errorMessage: 'boom' } : {}),
    timestamp: 0,
  } as AssistantMessage;
}

const catalog: Model<any>[] = [
  { provider: 'github-copilot', id: 'gpt-5.4-mini', api: 'openai-completions' } as Model<any>,
  { provider: 'openai-codex', id: 'gpt-5-mini', api: 'openai-completions' } as Model<any>,
];

test('the provider receives the masked context, not the original', async () => {
  const redactor = build([openaiKeyRule()]);
  const { models, seen } = fakeModels(catalog, (model) => assistantReply(model, 'stop'));

  await runCompletion(
    models,
    [],
    fakeRequest({
      model: 'openai-codex:gpt-5-mini',
      messages: [{ role: 'user', content: `my key is ${KEY}` }],
    }),
    fakeReply(),
    chatAdapter,
    undefined,
    redactor,
  );

  assert.equal(seen.length, 1);
  const sent = seen[0].context.messages[0];
  const text = sent.role === 'user' ? JSON.stringify(sent.content) : '';
  assert.ok(text.includes('[REDACTED:openai-key]'), text);
  assert.ok(!text.includes(KEY), 'the raw credential must not reach the provider');
});

test('fallback re-evaluates the scope against the second provider', async () => {
  // Only the fallback target is in scope, so the first attempt goes out unmasked
  // and the second must still be masked after the group moves on.
  const redactor = scoped(['openai-codex'], [openaiKeyRule()]);
  const { models, seen } = fakeModels(catalog, (model) =>
    assistantReply(model, model.provider === 'github-copilot' ? 'error' : 'stop'),
  );

  await runCompletion(
    models,
    [freeGroup],
    fakeRequest({ model: 'free', messages: [{ role: 'user', content: `key ${KEY}` }] }),
    fakeReply(),
    chatAdapter,
    undefined,
    redactor,
  );

  assert.equal(seen.length, 2);
  const first = JSON.stringify(seen[0].context.messages[0]);
  const second = JSON.stringify(seen[1].context.messages[0]);
  assert.ok(first.includes(KEY), 'github-copilot is outside the configured scope');
  assert.ok(second.includes('[REDACTED:openai-key]'));
  assert.ok(!second.includes(KEY));
});

test('a group scope masks whichever member serves the request', async () => {
  const redactor = scoped(['free'], [openaiKeyRule()]);
  const { models, seen } = fakeModels(catalog, (model) =>
    assistantReply(model, model.provider === 'github-copilot' ? 'error' : 'stop'),
  );

  await runCompletion(
    models,
    [freeGroup],
    fakeRequest({ model: 'free', messages: [{ role: 'user', content: `key ${KEY}` }] }),
    fakeReply(),
    chatAdapter,
    undefined,
    redactor,
  );

  assert.equal(seen.length, 2);
  for (const attempt of seen) {
    const text = JSON.stringify(attempt.context.messages[0]);
    assert.ok(!text.includes(KEY), `${attempt.model.provider} leaked the credential`);
  }
});

test('without a redactor the context is forwarded unchanged', async () => {
  const { models, seen } = fakeModels(catalog, (model) => assistantReply(model, 'stop'));

  await runCompletion(
    models,
    [],
    fakeRequest({
      model: 'openai-codex:gpt-5-mini',
      messages: [{ role: 'user', content: `key ${KEY}` }],
    }),
    fakeReply(),
    chatAdapter,
  );

  assert.ok(JSON.stringify(seen[0].context).includes(KEY));
});
