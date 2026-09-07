import { createHash } from 'node:crypto';

import type { Context, Model } from '@earendil-works/pi-ai';
import type { FastifyRequest } from 'fastify';

export const SESSION_ID_HEADER = 'x-session-id';

const MAX_SESSION_ID_LENGTH = 256;
const MAX_SEED_LENGTH = 8192;
const SESSION_AFFINITY_PROVIDERS = new Set(['openrouter']);
const HEADER_SAFE = /^[\x20-\x7e]+$/;

type SessionRequest = Pick<FastifyRequest, 'headers'> & { body?: unknown };

export function requestSessionId(request: SessionRequest): string | undefined {
  const header = request.headers[SESSION_ID_HEADER];
  const claimed = Array.isArray(header) ? header[0] : header;
  const body = request.body as any;
  return normalizeSessionId(claimed) ?? normalizeSessionId(body?.session_id);
}

export function conversationSessionId(context: Context): string | undefined {
  const parts: string[] = [];
  if (context.systemPrompt) parts.push(context.systemPrompt);

  const opening = context.messages?.find((message) => message.role === 'user');
  if (opening) parts.push(messageText(opening.content));

  const seed = parts.join('\n\n').slice(0, MAX_SEED_LENGTH).trim();
  return seed ? digest(seed) : undefined;
}

function resolveSessionId(request: SessionRequest, context: Context): string | undefined {
  return requestSessionId(request) ?? conversationSessionId(context);
}

export function withSessionAffinity<T extends { headers?: Record<string, string> }>(
  options: T,
  model: Pick<Model<any>, 'provider'>,
  request: SessionRequest,
  context: Context,
): T {
  if (!SESSION_AFFINITY_PROVIDERS.has(model.provider)) return options;

  const sessionId = resolveSessionId(request, context);
  if (!sessionId) return options;

  return { ...options, headers: { [SESSION_ID_HEADER]: sessionId, ...options.headers } };
}

function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((part: any) => part?.type === 'text' && typeof part.text === 'string')
    .map((part: any) => part.text)
    .join('\n');
}

function normalizeSessionId(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const sendable = trimmed.length <= MAX_SESSION_ID_LENGTH && HEADER_SAFE.test(trimmed);
  return sendable ? trimmed : digest(trimmed);
}

function digest(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 32);
}
