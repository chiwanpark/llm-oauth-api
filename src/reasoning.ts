import type { Model, ThinkingContent } from '@earendil-works/pi-ai';

/**
 * Providers attach continuation data to reasoning: an OpenAI reasoning item
 * holding `encrypted_content`, an Anthropic thinking signature, a Gemini
 * thought signature. Dropping it makes the reasoning unusable on the next turn,
 * so it is handed to the client verbatim and taken back unchanged.
 *
 * pi-ai normalizes all of it onto `ThinkingContent.thinkingSignature`, but the
 * shape differs per API: the Responses family stores the whole reasoning item
 * as JSON, everyone else stores a bare opaque string. This module translates
 * between that internal form and the wire formats clients already speak.
 */

const RESPONSES_APIS = new Set([
  'openai-responses',
  'azure-openai-responses',
  'openai-codex-responses',
]);

/** `include` value that asks for reasoning continuation data on the response. */
export const ENCRYPTED_REASONING_INCLUDE = 'reasoning.encrypted_content';

type ModelIdentity = Pick<Model<any>, 'api'>;

export type ReasoningItem = {
  id: string;
  type: 'reasoning';
  status: 'completed';
  summary: Array<{ type: 'summary_text'; text: string }>;
  encrypted_content?: string;
};

/**
 * An entry of the `reasoning_details` array that Chat Completions clients use
 * to carry reasoning across turns. `format` names the provider dialect so a
 * payload is never replayed at a provider that cannot read it.
 */
export type ReasoningDetail = {
  type: 'reasoning.summary' | 'reasoning.text' | 'reasoning.encrypted';
  index: number;
  format: string;
  id?: string;
  summary?: string;
  text?: string;
  signature?: string | null;
  data?: string;
};

function isResponsesApi(model: ModelIdentity): boolean {
  return RESPONSES_APIS.has(model.api);
}

/** The dialect names used by the `reasoning_details` convention. */
export function reasoningFormat(model: ModelIdentity): string {
  switch (model.api) {
    case 'openai-responses':
    case 'openai-codex-responses':
      return 'openai-responses-v1';
    case 'azure-openai-responses':
      return 'azure-openai-responses-v1';
    case 'anthropic-messages':
      return 'anthropic-claude-v1';
    case 'google-generative-ai':
    case 'google-vertex':
      return 'google-gemini-v1';
    default:
      return 'unknown';
  }
}

/** Unpacks the provider payload pi-ai stored on a thinking block. */
function providerPayload(
  model: ModelIdentity,
  block: ThinkingContent,
): { id?: string; encryptedContent?: string } {
  const signature = block.thinkingSignature;
  if (!signature) return {};

  if (!isResponsesApi(model)) return { encryptedContent: signature };

  try {
    const item = JSON.parse(signature);
    return {
      ...(typeof item?.id === 'string' ? { id: item.id } : {}),
      ...(typeof item?.encrypted_content === 'string'
        ? { encryptedContent: item.encrypted_content }
        : {}),
    };
  } catch {
    return {};
  }
}

/** Rebuilds the payload in the form pi-ai hands back to the provider. */
function restorePayload(
  model: ModelIdentity,
  payload: { id?: string; encryptedContent?: string; summary: string },
): string | undefined {
  if (!payload.encryptedContent) return undefined;
  if (!isResponsesApi(model)) return payload.encryptedContent;

  // The Responses API replays whole reasoning items, so the id has to travel
  // with the ciphertext; an item without one is rejected upstream.
  if (!payload.id) return undefined;
  return JSON.stringify({
    id: payload.id,
    type: 'reasoning',
    summary: payload.summary ? [{ type: 'summary_text', text: payload.summary }] : [],
    encrypted_content: payload.encryptedContent,
  });
}

/**
 * Renders one thinking block as a Responses API reasoning item. The upstream
 * item id is preserved when it is known, because the provider pairs the
 * ciphertext with it on replay.
 */
export function reasoningItemFromThinking(
  model: ModelIdentity,
  block: ThinkingContent,
  fallbackId: string,
  includeEncrypted: boolean,
): ReasoningItem | undefined {
  const payload = providerPayload(model, block);
  const encrypted = includeEncrypted ? payload.encryptedContent : undefined;
  // Redacted reasoning is a placeholder string with the real content sealed by
  // the provider, so it only carries meaning as encrypted content.
  const summary = block.redacted ? '' : block.thinking;
  if (!summary && !encrypted) return undefined;

  return {
    id: payload.id ?? fallbackId,
    type: 'reasoning',
    status: 'completed',
    summary: summary ? [{ type: 'summary_text', text: summary }] : [],
    ...(encrypted ? { encrypted_content: encrypted } : {}),
  };
}

/** Restores a thinking block from a Responses API `reasoning` input item. */
export function thinkingFromReasoningItem(
  model: ModelIdentity,
  item: any,
): ThinkingContent | undefined {
  const summary = reasoningItemText(item);
  const encryptedContent =
    typeof item?.encrypted_content === 'string' ? item.encrypted_content : undefined;
  const signature = restorePayload(model, {
    ...(typeof item?.id === 'string' ? { id: item.id } : {}),
    ...(encryptedContent !== undefined ? { encryptedContent } : {}),
    summary,
  });

  if (!summary && !signature) return undefined;

  // Anthropic redacts reasoning by replacing it with an opaque blob, which
  // arrives here as encrypted content with nothing readable beside it.
  const redacted = !summary && model.api === 'anthropic-messages';

  return {
    type: 'thinking',
    thinking: summary,
    ...(signature !== undefined ? { thinkingSignature: signature } : {}),
    ...(redacted ? { redacted: true } : {}),
  };
}

function reasoningItemText(item: any): string {
  const parts = [
    ...(Array.isArray(item?.summary) ? item.summary : []),
    ...(Array.isArray(item?.content) ? item.content : []),
  ];

  const texts: string[] = [];
  for (const part of parts) {
    const text = typeof part?.text === 'string' ? part.text : '';
    if (text) texts.push(text);
  }
  return texts.join('\n\n');
}

/**
 * Renders thinking blocks as `reasoning_details`, the array Chat Completions
 * clients round-trip. The Responses family reports a summary plus separate
 * ciphertext; the other providers sign the reasoning text itself.
 */
export function reasoningDetailsFromThinking(
  model: ModelIdentity,
  blocks: readonly ThinkingContent[],
): ReasoningDetail[] | undefined {
  const format = reasoningFormat(model);
  const details: ReasoningDetail[] = [];

  blocks.forEach((block, index) => {
    const payload = providerPayload(model, block);
    const id = payload.id !== undefined ? { id: payload.id } : {};

    if (isResponsesApi(model)) {
      if (block.thinking) {
        details.push({
          type: 'reasoning.summary',
          index,
          format,
          summary: block.thinking,
          ...id,
        });
      }
      if (payload.encryptedContent) {
        details.push({
          type: 'reasoning.encrypted',
          index,
          format,
          data: payload.encryptedContent,
          ...id,
        });
      }
      return;
    }

    if (block.redacted) {
      if (payload.encryptedContent) {
        details.push({
          type: 'reasoning.encrypted',
          index,
          format,
          data: payload.encryptedContent,
        });
      }
      return;
    }

    if (!block.thinking && !payload.encryptedContent) return;
    details.push({
      type: 'reasoning.text',
      index,
      format,
      text: block.thinking,
      signature: payload.encryptedContent ?? null,
    });
  });

  return details.length ? details : undefined;
}

/**
 * Restores thinking blocks from `reasoning_details`. Entries whose `format`
 * belongs to another provider keep their text and lose their payload: a
 * signature only means something to the provider that issued it, and replaying
 * a foreign one is a request error.
 */
export function thinkingFromReasoningDetails(
  model: ModelIdentity,
  value: unknown,
): ThinkingContent[] {
  if (!Array.isArray(value)) return [];

  const format = reasoningFormat(model);
  const groups = new Map<number, ReasoningDetail[]>();
  for (const [position, entry] of value.entries()) {
    if (!entry || typeof entry !== 'object') continue;
    const detail = entry as ReasoningDetail;
    const index = typeof detail.index === 'number' ? detail.index : position;
    const group = groups.get(index);
    if (group) group.push(detail);
    else groups.set(index, [detail]);
  }

  const blocks: ThinkingContent[] = [];
  for (const index of [...groups.keys()].sort((a, b) => a - b)) {
    const group = groups.get(index)!;
    // An entry without a format predates the convention or comes from a client
    // that only echoes text, so it is treated as portable plain reasoning.
    const portable = group.every((detail) => !detail.format || detail.format === format);

    const text = group
      .map((detail) => (typeof detail.text === 'string' ? detail.text : detail.summary))
      .find((value) => typeof value === 'string' && value);
    const encrypted = group.find((detail) => detail.type === 'reasoning.encrypted')?.data;
    const signature = group.find((detail) => typeof detail.signature === 'string')?.signature;
    const id = group.find((detail) => typeof detail.id === 'string')?.id;

    if (!portable) {
      if (text) blocks.push({ type: 'thinking', thinking: text });
      continue;
    }

    const restored = restorePayload(model, {
      ...(id !== undefined ? { id } : {}),
      ...(encrypted !== undefined ? { encryptedContent: encrypted } : {}),
      summary: text ?? '',
    });
    const thinkingSignature = restored ?? signature ?? undefined;

    if (!text && !thinkingSignature) continue;
    blocks.push({
      type: 'thinking',
      thinking: text ?? '',
      ...(thinkingSignature !== undefined ? { thinkingSignature } : {}),
      ...(!text && encrypted && model.api === 'anthropic-messages' ? { redacted: true } : {}),
    });
  }

  return blocks;
}
