import { readFile } from 'node:fs/promises';
import path from 'node:path';

import type { Context, Message, Tool } from '@earendil-works/pi-ai';

import type { ModelGroup } from './groups.js';
import { resolveProviderId, type SupportedProviderId } from './providers.js';

/** Label used in errors when rules did not come from a file on disk. */
const DEFAULT_SOURCE = 'the redaction file';

/** Flags a rule may set; `g` is implied and `y` would make matching stateful. */
const ALLOWED_FLAGS = new Set(['i', 'm', 's', 'u', 'v']);

const VALID_RULE_NAME = /^[a-z0-9][a-z0-9-]*$/;

/**
 * The mask used when neither the file nor the rule names one.
 *
 * `{name}` expands to the rule name, so the default says which pattern fired.
 * A file-wide `replacement` can keep that by using the placeholder too, or drop
 * it by writing a plain literal.
 */
const DEFAULT_REPLACEMENT = '[REDACTED:{name}]';

const NAME_PLACEHOLDER = '{name}';

/**
 * One credential shape to mask.
 *
 * The pattern is compiled once at startup so a malformed regex fails the server
 * rather than silently letting a secret through on the first request that would
 * have matched it.
 */
export type RedactionRule = {
  name: string;
  pattern: RegExp;
  replacement: string;
  /**
   * Mask only this capture group instead of the whole match.
   *
   * Credentials are usually recognised by what surrounds them — `FOO_API_KEY=`,
   * a JDBC userinfo colon — but that context is what makes the rest of the
   * message readable, so a rule can name the context in its pattern and still
   * mask only the secret inside it.
   */
  captureGroup?: number;
};

/**
 * One entry of the file-wide `models` list.
 *
 * A `model` selector leaves out the half it does not constrain: no provider and
 * no pattern is the bare `*`, a provider alone covers that whole catalog, and a
 * pattern alone is `*:<glob>` for a model id several providers publish. A
 * `group` selector matches the requested group name instead, so the scope
 * follows a virtual model no matter which member ends up serving it.
 */
export type ModelSelector =
  | { kind: 'model'; providerId?: SupportedProviderId; modelPattern?: RegExp; text: string }
  | { kind: 'group'; name: string; text: string };

/** The model a request is actually about to be sent to. */
export type RedactionTarget = {
  providerId: string;
  modelId: string;
  /** The id the client asked for, which is the group name for a group request. */
  requestedModel?: string;
};

/** What a single redaction pass changed, for logging. */
export type RedactionSummary = {
  /** Total replaced substrings across every rule. */
  total: number;
  /** Replacement count per rule name, only for rules that matched. */
  byRule: Record<string, number>;
};

export type RedactionResult = {
  context: Context;
  summary: RedactionSummary;
};

/**
 * Compiled rules plus the one scope they all run in.
 *
 * Whether the scope covers a target is pure string work, so it is cheap enough
 * to run on every attempt; that matters because a group falls back between
 * providers and the scope may cover one member but not the next.
 */
export type Redactor = {
  rules: readonly RedactionRule[];
  /** The models these rules apply to; empty means no model at all. */
  selectors: readonly ModelSelector[];
  /** Whether the configured scope covers one target. */
  covers(target: RedactionTarget): boolean;
  /** Masks every outbound field of a context, leaving opaque data untouched. */
  redact(context: Context, target: RedactionTarget): RedactionResult;
};

const EMPTY_SUMMARY: RedactionSummary = { total: 0, byRule: {} };

/** A redactor that never changes anything, used when no file was configured. */
export const NO_REDACTION: Redactor = {
  rules: [],
  selectors: [],
  covers: () => false,
  redact: (context) => ({ context, summary: EMPTY_SUMMARY }),
};

/**
 * Reads a redaction file into compiled rules.
 *
 * Like the groups file, every problem is a startup error: a rule that was meant
 * to mask a credential but was quietly dropped is worse than not starting, because
 * nothing downstream would report the secret that went out in clear text.
 */
export async function loadRedactionConfig(
  filePath: string,
  enabledProviderIds: readonly SupportedProviderId[],
  groups: readonly ModelGroup[] = [],
): Promise<Redactor> {
  const resolved = path.resolve(filePath);

  let raw: string;
  try {
    raw = await readFile(resolved, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new Error(`Redaction file not found: ${resolved}`, { cause: error });
    }
    throw new Error(`Cannot read redaction file ${resolved}: ${describeError(error)}`, {
      cause: error,
    });
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${resolved} is not valid JSON: ${describeError(error)}`, { cause: error });
  }

  return parseRedactionConfig(parsed, enabledProviderIds, groups, resolved);
}

/**
 * Validates already-parsed redaction configuration.
 *
 * One `models` list at the top level scopes the whole file. Scope is a property
 * of the deployment rather than of any single pattern — which providers you
 * distrust does not change between one credential shape and the next — so it is
 * stated once instead of on every rule. A top-level `replacement` works the
 * same way, as the default mask each rule may still override.
 *
 * `source` only shapes error messages, so callers that did not read a file can
 * leave it out.
 */
export function parseRedactionConfig(
  config: unknown,
  enabledProviderIds: readonly SupportedProviderId[],
  groups: readonly ModelGroup[] = [],
  source: string = DEFAULT_SOURCE,
): Redactor {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error(`${source} must contain a JSON object with a "rules" array`);
  }

  const rulesValue = (config as Record<string, unknown>).rules;
  if (!Array.isArray(rulesValue)) {
    throw new Error(`${source} must contain a "rules" array`);
  }

  const enabled = new Set<string>(enabledProviderIds);
  const groupNames = new Set(groups.map((group) => group.name));

  const selectors = parseSelectors(
    (config as Record<string, unknown>).models,
    enabled,
    groupNames,
    source,
  );

  const raw = config as Record<string, unknown>;
  if (raw.replacement !== undefined && typeof raw.replacement !== 'string') {
    throw new Error(`${source} "replacement" must be a string`);
  }
  const defaultReplacement = raw.replacement ?? DEFAULT_REPLACEMENT;
  assertNoCaptureRefs(defaultReplacement, 'top-level', source);

  const seen = new Set<string>();
  const rules = rulesValue.map((entry, index) =>
    parseRule(entry, index, defaultReplacement, seen, source),
  );

  return createRedactor(rules, selectors);
}

/** Builds a redactor from rules and a scope that are already compiled. */
export function createRedactor(
  rules: readonly RedactionRule[],
  selectors: readonly ModelSelector[] = [{ kind: 'model', text: '*' }],
): Redactor {
  const covers = (target: RedactionTarget): boolean =>
    selectors.some((selector) => selectorMatches(selector, target));

  return {
    rules,
    selectors,
    covers,
    redact: (context, target) =>
      covers(target) ? redactContext(context, rules) : { context, summary: EMPTY_SUMMARY },
  };
}

function parseRule(
  entry: unknown,
  index: number,
  defaultReplacement: string,
  seen: Set<string>,
  source: string,
): RedactionRule {
  const position = `${source} rule #${index + 1}`;

  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error(`${position} must be an object`);
  }

  const raw = entry as Record<string, unknown>;
  const name = typeof raw.name === 'string' ? raw.name.trim().toLowerCase() : '';
  if (!name) {
    throw new Error(`${position} must have a non-empty "name"`);
  }
  if (!VALID_RULE_NAME.test(name)) {
    throw new Error(
      `${source} rule "${raw.name}" has an invalid name; a rule name may only contain ` +
        'letters, digits, and hyphens',
    );
  }
  if (seen.has(name)) {
    throw new Error(`${source} declares the rule "${name}" twice`);
  }
  seen.add(name);

  if (typeof raw.pattern !== 'string' || !raw.pattern) {
    throw new Error(`${source} rule "${name}" must have a non-empty "pattern"`);
  }

  const captureGroup = parseCaptureGroup(raw.captureGroup, name, source);
  // `d` exposes where each group matched, which is the only way to rewrite one
  // group and leave the surrounding context in place.
  const flags = parseFlags(raw.flags, name, source) + (captureGroup === undefined ? '' : 'd');

  let pattern: RegExp;
  try {
    pattern = new RegExp(raw.pattern, flags);
  } catch (error) {
    throw new Error(`${source} rule "${name}" has an invalid pattern: ${describeError(error)}`, {
      cause: error,
    });
  }

  // A pattern that matches the empty string would splice the replacement between
  // every character of every message instead of masking anything.
  if (pattern.test('')) {
    throw new Error(
      `${source} rule "${name}" matches the empty string, which would rewrite every message; ` +
        'make the pattern require at least one character',
    );
  }
  pattern.lastIndex = 0;

  if (captureGroup !== undefined) {
    const available = countCaptureGroups(raw.pattern, flags);
    if (captureGroup > available) {
      throw new Error(
        `${source} rule "${name}" sets "captureGroup" to ${captureGroup} but its pattern has ` +
          `${available === 1 ? '1 capture group' : `${available} capture groups`}`,
      );
    }
  }

  if (raw.replacement !== undefined && typeof raw.replacement !== 'string') {
    throw new Error(`${source} rule "${name}" must have a string "replacement"`);
  }
  const replacement = (raw.replacement ?? defaultReplacement).replaceAll(NAME_PLACEHOLDER, name);
  assertNoCaptureRefs(replacement, `rule "${name}"`, source);

  // A leftover per-rule scope is refused rather than ignored: silently widening
  // a rule to the whole file is exactly the mistake that ends with a credential
  // reaching a provider the operator meant to exclude.
  if (raw.models !== undefined) {
    throw new Error(
      `${source} rule "${name}" sets "models", which is not a rule field; move the model ` +
        'list to the top level of the file, where it scopes every rule',
    );
  }

  return { name, pattern, replacement, ...(captureGroup === undefined ? {} : { captureGroup }) };
}

function parseCaptureGroup(value: unknown, name: string, source: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new Error(
      `${source} rule "${name}" must have "captureGroup" as a whole number of 1 or more, ` +
        'counting capture groups left to right',
    );
  }
  return value;
}

/** Counts capture groups by making the pattern also match nothing. */
function countCaptureGroups(source: string, flags: string): number {
  return new RegExp(`${source}|`, flags.replace('g', '')).exec('')!.length - 1;
}

/**
 * `$1` and friends are inert here, because the replacement is written literally
 * rather than expanded. Left alone they fail open: the output looks redacted
 * while carrying a stray `$1`, so they are refused with a pointer to the field
 * that does what the author meant.
 */
const CAPTURE_REF = /\$(?:\d|&|`|'|<[^>]*>)/;

function assertNoCaptureRefs(replacement: string, label: string, source: string): void {
  const found = CAPTURE_REF.exec(replacement);
  if (found) {
    throw new Error(
      `${source} ${label} has a "replacement" containing ${found[0]}, which is written out ` +
        'literally rather than expanded; set "captureGroup" to mask only part of the match',
    );
  }
}

function parseFlags(value: unknown, name: string, source: string): string {
  if (value === undefined) return 'g';
  if (typeof value !== 'string') {
    throw new Error(`${source} rule "${name}" must have a string "flags"`);
  }

  const flags = new Set(value.trim().split(''));
  // `g` drives replaceAll; `y` anchors to lastIndex and would skip most matches.
  flags.delete('g');
  for (const flag of flags) {
    if (!ALLOWED_FLAGS.has(flag)) {
      throw new Error(
        `${source} rule "${name}" has an unsupported flag "${flag}"; allowed flags are ` +
          `${[...ALLOWED_FLAGS].join(', ')}`,
      );
    }
  }

  return `g${[...flags].join('')}`;
}

/**
 * Reads the file-wide `models` list.
 *
 * Leaving it out is the common case and means every model, so `undefined` is a
 * default rather than an error.
 */
function parseSelectors(
  value: unknown,
  enabledProviderIds: ReadonlySet<string>,
  groupNames: ReadonlySet<string>,
  source: string,
): ModelSelector[] {
  if (value === undefined) return [{ kind: 'model', text: '*' }];

  if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'string')) {
    throw new Error(`${source} "models" must be an array of strings`);
  }

  const entries = (value as string[]).map((entry) => entry.trim()).filter(Boolean);
  if (!entries.length) {
    throw new Error(
      `${source} "models" lists no models; drop it to apply the rules to every model`,
    );
  }

  return entries.map((entry) => parseSelector(entry, enabledProviderIds, groupNames, source));
}

function parseSelector(
  entry: string,
  enabledProviderIds: ReadonlySet<string>,
  groupNames: ReadonlySet<string>,
  source: string,
): ModelSelector {
  if (entry === '*') return { kind: 'model', text: entry };

  // The groups file reads an entry without a separator as a group name, so the
  // same text means the same thing here.
  if (!entry.includes(':')) {
    const normalized = entry.toLowerCase().replaceAll('_', '-');
    const providerId = resolveProviderId(normalized);
    if (providerId) {
      assertProviderEnabled(providerId, entry, enabledProviderIds, source);
      return { kind: 'model', providerId, text: entry };
    }
    if (groupNames.has(normalized)) {
      return { kind: 'group', name: normalized, text: entry };
    }
    throw new Error(
      `${source} "models" entry "${entry}" is neither a known provider nor a declared ` +
        'group; write a concrete model as <provider>:<model>',
    );
  }

  const separator = entry.indexOf(':');
  const providerText = entry.slice(0, separator).trim();
  const modelText = entry.slice(separator + 1).trim();
  if (!providerText || !modelText) {
    throw new Error(
      `${source} "models" entry "${entry}" is not a valid model selector; write it as ` +
        '<provider>:<model>',
    );
  }

  // `*:<model>` covers a model id that several providers publish under the same
  // name, which is the usual reason to reach for a cross-provider entry.
  const providerId = providerText === '*' ? undefined : resolveProviderId(providerText);
  if (providerText !== '*') {
    if (!providerId) {
      throw new Error(`${source} "models" entry "${entry}" names an unknown provider`);
    }
    assertProviderEnabled(providerId, entry, enabledProviderIds, source);
  }

  return {
    kind: 'model',
    ...(providerId ? { providerId } : {}),
    ...(modelText === '*' ? {} : { modelPattern: globToRegExp(modelText) }),
    text: entry,
  };
}

function assertProviderEnabled(
  providerId: SupportedProviderId,
  entry: string,
  enabledProviderIds: ReadonlySet<string>,
  source: string,
): void {
  if (enabledProviderIds.size && !enabledProviderIds.has(providerId)) {
    throw new Error(
      `${source} "models" entry "${entry}" names the provider "${providerId}", which is ` +
        'not enabled; add it to --providers or drop the entry',
    );
  }
}

/** Turns a model glob into an anchored, case-insensitive regex. */
function globToRegExp(glob: string): RegExp {
  const escaped = glob.replaceAll(/[.*+?^${}()|[\]\\]/g, (char) =>
    char === '*' ? '\u0000' : `\\${char}`,
  );
  return new RegExp(`^${escaped.replaceAll('\u0000', '.*')}$`, 'i');
}

function selectorMatches(selector: ModelSelector, target: RedactionTarget): boolean {
  if (selector.kind === 'group') {
    return normalizeName(target.requestedModel) === selector.name;
  }
  if (selector.providerId && selector.providerId !== target.providerId) return false;
  if (selector.modelPattern && !selector.modelPattern.test(target.modelId)) return false;
  return true;
}

function normalizeName(value: string | undefined): string {
  return (value ?? '').trim().toLowerCase().replaceAll('_', '-');
}

/**
 * Masks every readable outbound field of a context.
 *
 * Encrypted reasoning, thinking signatures, tool-call ids, and image bytes are
 * deliberately left alone: they are opaque to a credential pattern, and
 * rewriting a signed or correlated value breaks the request instead of
 * protecting it.
 */
export function redactContext(context: Context, rules: readonly RedactionRule[]): RedactionResult {
  if (!rules.length) return { context, summary: EMPTY_SUMMARY };

  const counts = new Map<string, number>();
  const mask = (value: string): string => applyRules(value, rules, counts);

  const redacted: Context = {
    ...context,
    ...(context.systemPrompt !== undefined ? { systemPrompt: mask(context.systemPrompt) } : {}),
    messages: context.messages.map((message) => redactMessage(message, mask)),
    ...(context.tools ? { tools: context.tools.map((tool) => redactTool(tool, mask)) } : {}),
  };

  const byRule = Object.fromEntries(counts);
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  return { context: redacted, summary: { total, byRule } };
}

function redactMessage(message: Message, mask: (value: string) => string): Message {
  switch (message.role) {
    case 'user':
      return {
        ...message,
        content:
          typeof message.content === 'string'
            ? mask(message.content)
            : message.content.map((block) =>
                block.type === 'text' ? { ...block, text: mask(block.text) } : block,
              ),
      };

    case 'assistant':
      return {
        ...message,
        content: message.content.map((block) => {
          // Thinking is skipped whole: its readable half is paired with an
          // encrypted signature that must keep matching it.
          if (block.type === 'text') return { ...block, text: mask(block.text) };
          if (block.type === 'toolCall') {
            return { ...block, arguments: maskJsonValue(block.arguments, mask) };
          }
          return block;
        }),
      };

    case 'toolResult':
      return {
        ...message,
        content: message.content.map((block) =>
          block.type === 'text' ? { ...block, text: mask(block.text) } : block,
        ),
      };
  }
}

/**
 * Tool descriptions reach the provider verbatim, so an example key written into
 * one leaks like any other text. The parameter schema is left alone because
 * providers validate it and constrained sampling is built from it.
 */
function redactTool(tool: Tool, mask: (value: string) => string): Tool {
  return { ...tool, description: mask(tool.description) };
}

/** Walks tool-call arguments, masking string leaves and keeping the shape. */
function maskJsonValue(value: unknown, mask: (value: string) => string): any {
  if (typeof value === 'string') return mask(value);
  if (Array.isArray(value)) return value.map((entry) => maskJsonValue(entry, mask));
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, entry]) => [
        key,
        maskJsonValue(entry, mask),
      ]),
    );
  }
  return value;
}

function applyRules(
  value: string,
  rules: readonly RedactionRule[],
  counts: Map<string, number>,
): string {
  let result = value;
  for (const rule of rules) {
    if (!result) break;
    let hits = 0;
    rule.pattern.lastIndex = 0;
    result =
      rule.captureGroup === undefined
        ? result.replaceAll(rule.pattern, () => {
            hits += 1;
            return rule.replacement;
          })
        : replaceCaptureGroup(result, rule, rule.captureGroup, () => {
            hits += 1;
          });
    if (hits) counts.set(rule.name, (counts.get(rule.name) ?? 0) + hits);
  }
  return result;
}

/**
 * Rewrites one capture group per match and keeps everything around it.
 *
 * The group's position comes from the `d` flag rather than from searching the
 * match for the group's text, which would pick the wrong occurrence whenever a
 * secret happens to repeat its own context.
 */
function replaceCaptureGroup(
  value: string,
  rule: RedactionRule,
  group: number,
  onHit: () => void,
): string {
  let result = '';
  let cursor = 0;
  let matched = false;

  for (const match of value.matchAll(rule.pattern)) {
    const span = match.indices?.[group];
    // An optional group can match nothing at all; there is no secret to mask.
    if (!span || span[0] === span[1]) continue;

    result += value.slice(cursor, span[0]) + rule.replacement;
    cursor = span[1];
    matched = true;
    onHit();
  }

  return matched ? result + value.slice(cursor) : value;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
