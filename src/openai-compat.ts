import { randomUUID } from 'node:crypto';

import type {
  AssistantMessage,
  Context,
  Message,
  Model,
  MutableModels,
  TextContent,
  ThinkingContent,
  Tool,
  ToolCall,
  ToolResultMessage,
  UserMessage,
} from '@earendil-works/pi-ai';

import {
  ENCRYPTED_REASONING_INCLUDE,
  reasoningDetailsFromThinking,
  reasoningItemFromThinking,
  thinkingFromReasoningDetails,
  thinkingFromReasoningItem,
} from './reasoning.js';

export type OpenAIModelInfo = {
  id: string;
  object: 'model';
  created: number;
  owned_by: string;
};

export function createOpenAIModelsResponse(models: readonly Model<any>[]): {
  object: 'list';
  data: OpenAIModelInfo[];
} {
  return {
    object: 'list',
    data: models.map((model) => ({
      id: exposedModelId(model),
      object: 'model',
      created: 0,
      owned_by: model.provider,
    })),
  };
}

export function exposedModelId(model: Pick<Model<any>, 'provider' | 'id'>): string {
  return `${model.provider}:${model.id}`;
}

export function resolveModelByName(
  models: MutableModels,
  requested: string,
): Model<any> | undefined {
  for (const model of models.getModels()) {
    if (exposedModelId(model) === requested) return model;
  }

  const byRawId = models.getModels().filter((model) => model.id === requested);
  return byRawId.length === 1 ? byRawId[0] : undefined;
}

export function createOpenAIError(message: string, type = 'invalid_request_error', code?: string) {
  return {
    error: {
      message,
      type,
      param: null,
      code: code ?? null,
    },
  };
}

export function mapFinishReason(
  stopReason: AssistantMessage['stopReason'],
): 'stop' | 'length' | 'tool_calls' | 'content_filter' {
  switch (stopReason) {
    case 'length':
      return 'length';
    case 'toolUse':
      return 'tool_calls';
    default:
      return 'stop';
  }
}

export function assistantText(message: AssistantMessage): string {
  return message.content
    .filter((block): block is TextContent => block.type === 'text')
    .map((block) => block.text)
    .join('');
}

export function assistantThinking(message: AssistantMessage): ThinkingContent[] {
  return message.content.filter((block): block is ThinkingContent => block.type === 'thinking');
}

export type RenderOptions = {
  /**
   * Whether the client asked for the provider's reasoning continuation data.
   * Mirrors the Responses API, which omits `encrypted_content` unless it is
   * named in `include`.
   */
  includeEncryptedReasoning?: boolean;
  /** Chat Completions reports stream usage only under `stream_options`. */
  includeUsage?: boolean;
  /** The request, whose parameters the Responses API echoes on its response. */
  request?: any;
};

export function buildRenderOptions(body: any): RenderOptions {
  const include = Array.isArray(body?.include) ? body.include : [];
  return {
    includeEncryptedReasoning: include.includes(ENCRYPTED_REASONING_INCLUDE),
    includeUsage: body?.stream_options?.include_usage === true,
    request: body,
  };
}

/**
 * Chat Completions has no `include`, so reasoning continuation data always
 * rides along as `reasoning_details` the way the wider ecosystem does it.
 */
export function chatReasoningDetails(model: Model<any>, message: AssistantMessage) {
  return reasoningDetailsFromThinking(model, assistantThinking(message));
}

export function assistantReasoning(message: AssistantMessage): string {
  return assistantThinking(message)
    .map((block) => block.thinking)
    .join('');
}

export function assistantToolCalls(message: AssistantMessage): ToolCall[] {
  return message.content.filter((block): block is ToolCall => block.type === 'toolCall');
}

export function assistantUsage(message: AssistantMessage) {
  return {
    prompt_tokens: message.usage.input,
    completion_tokens: message.usage.output,
    total_tokens: message.usage.totalTokens,
    prompt_tokens_details: {
      cached_tokens: message.usage.cacheRead,
      audio_tokens: 0,
    },
    completion_tokens_details: {
      reasoning_tokens: message.usage.reasoning ?? 0,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0,
    },
  };
}

/**
 * Fields OpenAI reports on every completion. Nothing here is negotiated with
 * the provider, so they are constant: no tier is selected, and there is no
 * backend build to fingerprint.
 */
const SERVICE_TIER = 'default';
const SYSTEM_FINGERPRINT = null;

export async function buildChatContext(model: Model<any>, body: any): Promise<Context> {
  const systemPrompts: string[] = [];
  const messages: Message[] = [];
  const now = Date.now();

  for (
    let index = 0;
    index < (Array.isArray(body.messages) ? body.messages.length : 0);
    index += 1
  ) {
    const message = body.messages[index];
    const timestamp = now + index;
    switch (message?.role) {
      case 'system':
      case 'developer': {
        const text = await contentToPlainText(message.content);
        if (text) systemPrompts.push(text);
        break;
      }
      case 'user':
        messages.push({
          role: 'user',
          content: await normalizeRichContent(message.content),
          timestamp,
        } satisfies UserMessage);
        break;
      case 'assistant':
        messages.push(await normalizeAssistantHistoryMessage(model, message, timestamp));
        break;
      case 'tool':
        messages.push({
          role: 'toolResult',
          toolCallId: String(message.tool_call_id ?? message.toolCallId ?? randomUUID()),
          toolName: String(message.name ?? 'tool'),
          content: await normalizeContentBlocks(message.content),
          isError: false,
          timestamp,
        } satisfies ToolResultMessage);
        break;
      default:
        throw new Error(`Unsupported chat message role: ${String(message?.role)}`);
    }
  }

  const context: Context = { messages };
  const systemPrompt = systemPrompts.length ? systemPrompts.join('\n\n') : undefined;
  const tools = buildTools(body.tools, body.tool_choice);
  if (systemPrompt) context.systemPrompt = systemPrompt;
  if (tools) context.tools = tools;
  return context;
}

export async function buildResponsesContext(model: Model<any>, body: any): Promise<Context> {
  const systemPrompts: string[] = [];
  const messages: Message[] = [];
  const now = Date.now();

  if (typeof body.instructions === 'string' && body.instructions.trim()) {
    systemPrompts.push(body.instructions.trim());
  }

  // Reasoning items precede the assistant turn they belong to, but pi-ai keeps
  // thinking inside that assistant message, so they are held until the turn
  // they introduce shows up.
  let pendingThinking: ThinkingContent[] = [];
  const takeThinking = (): ThinkingContent[] => {
    const taken = pendingThinking;
    pendingThinking = [];
    return taken;
  };

  const input = body.input;
  if (typeof input === 'string') {
    messages.push({ role: 'user', content: input, timestamp: now });
  } else if (Array.isArray(input)) {
    for (let index = 0; index < input.length; index += 1) {
      const item = input[index];
      const timestamp = now + index;

      if (item?.type === 'reasoning') {
        const thinking = thinkingFromReasoningItem(model, item);
        if (thinking) pendingThinking.push(thinking);
        continue;
      }

      if (item?.type === 'function_call_output') {
        messages.push({
          role: 'toolResult',
          toolCallId: String(item.call_id ?? item.tool_call_id ?? randomUUID()),
          toolName: String(item.name ?? 'tool'),
          content: await normalizeContentBlocks(item.output ?? item.content ?? ''),
          isError: Boolean(item.is_error),
          timestamp,
        } satisfies ToolResultMessage);
        continue;
      }

      if (item?.type === 'function_call') {
        const call = normalizeToolCall(item, item.call_id ?? randomUUID());
        messages.push(createAssistantHistoryMessage(model, [...takeThinking(), call], timestamp));
        continue;
      }

      if (item?.type === 'message' || item?.role) {
        const role = item.role;
        if (role === 'system' || role === 'developer') {
          const text = await contentToPlainText(item.content);
          if (text) systemPrompts.push(text);
          continue;
        }
        if (role === 'user') {
          messages.push({
            role: 'user',
            content: await normalizeResponsesContent(item.content),
            timestamp,
          } satisfies UserMessage);
          continue;
        }
        if (role === 'assistant') {
          messages.push(
            await normalizeAssistantHistoryMessage(model, item, timestamp, takeThinking()),
          );
          continue;
        }
      }

      throw new Error(`Unsupported responses input item: ${JSON.stringify(item)}`);
    }
  } else if (input != null) {
    throw new Error('Responses input must be a string or an array');
  }

  // Reasoning that trails the last assistant turn has nothing to attach to: a
  // thinking-only message at the end of a context is rejected by some
  // providers, so `pendingThinking` is left to fall away here.

  const context: Context = { messages };
  const systemPrompt = systemPrompts.length ? systemPrompts.join('\n\n') : undefined;
  const tools = buildTools(body.tools, body.tool_choice);
  if (systemPrompt) context.systemPrompt = systemPrompt;
  if (tools) context.tools = tools;
  return context;
}

export const REASONING_EFFORTS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] as const;

export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];

export function isReasoningEffort(value: unknown): value is ReasoningEffort {
  return REASONING_EFFORTS.some((effort) => effort === value);
}

/**
 * pi-ai's OpenAI-compatible APIs (chat completions, responses) read the
 * `reasoningEffort` option, while its Anthropic Messages API reads a
 * separate `reasoning` (ThinkingLevel) option instead. Both fields must be
 * set so reasoning/thinking is enabled regardless of which provider ends up
 * handling the request. `"none"` has no ThinkingLevel equivalent, so it maps
 * to `undefined` (thinking disabled).
 */
function thinkingLevelFor(effort: ReasoningEffort): Exclude<ReasoningEffort, 'none'> | undefined {
  return effort === 'none' ? undefined : effort;
}

export function buildPiOptions(body: any, signal?: AbortSignal) {
  const maxTokens =
    typeof body.max_output_tokens === 'number'
      ? body.max_output_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : typeof body.max_tokens === 'number'
          ? body.max_tokens
          : undefined;

  return {
    ...(typeof body.temperature === 'number' ? { temperature: body.temperature } : {}),
    ...(maxTokens !== undefined ? { maxTokens } : {}),
    ...(signal !== undefined ? { signal } : {}),
    ...(typeof body.user === 'string' ? { metadata: { user_id: body.user } } : {}),
  };
}

export function buildChatPiOptions(body: any, signal?: AbortSignal) {
  const options = buildPiOptions(body, signal);
  if (!isReasoningEffort(body.reasoning_effort)) return options;
  const reasoning = thinkingLevelFor(body.reasoning_effort);
  return {
    ...options,
    reasoningEffort: body.reasoning_effort,
    ...(reasoning !== undefined ? { reasoning } : {}),
  };
}

export function buildResponsesPiOptions(body: any, signal?: AbortSignal) {
  const options = buildPiOptions(body, signal);
  if (!isReasoningEffort(body.reasoning?.effort)) return options;
  const reasoning = thinkingLevelFor(body.reasoning.effort);
  return {
    ...options,
    reasoningEffort: body.reasoning.effort,
    ...(reasoning !== undefined ? { reasoning } : {}),
  };
}

function buildTools(rawTools: any, toolChoice: any): Tool[] | undefined {
  if (!Array.isArray(rawTools) || rawTools.length === 0 || toolChoice === 'none') {
    return undefined;
  }

  const tools = rawTools
    .filter((tool) => tool?.type === 'function' && tool.function?.name)
    .map((tool) => ({
      name: String(tool.function.name),
      description: String(tool.function.description ?? ''),
      parameters: tool.function.parameters ?? { type: 'object', properties: {} },
    })) satisfies Tool[];

  if (!tools.length) return undefined;

  const forcedName = toolChoice?.type === 'function' ? toolChoice.function?.name : undefined;
  if (typeof forcedName === 'string' && forcedName) {
    return tools.filter((tool) => tool.name === forcedName);
  }

  return tools;
}

async function normalizeAssistantHistoryMessage(
  model: Model<any>,
  message: any,
  timestamp: number,
  leadingThinking: ThinkingContent[] = [],
): Promise<AssistantMessage> {
  const content: AssistantMessage['content'] = [];

  // Thinking has to come first: Anthropic rejects assistant turns whose
  // thinking does not lead the block list.
  content.push(...leadingThinking, ...thinkingFromChatMessage(model, message));

  const textBlocks = await normalizeAssistantTextBlocks(message.content);
  content.push(...textBlocks);

  if (Array.isArray(message.tool_calls)) {
    for (const toolCall of message.tool_calls) {
      content.push(normalizeToolCall(toolCall, toolCall?.id ?? randomUUID()));
    }
  } else if (message.function_call?.name) {
    content.push(
      normalizeToolCall(message.function_call, message.function_call.id ?? randomUUID()),
    );
  }

  return createAssistantHistoryMessage(model, content, timestamp);
}

function createAssistantHistoryMessage(
  model: Model<any>,
  content: AssistantMessage['content'],
  timestamp: number,
): AssistantMessage {
  return {
    role: 'assistant',
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp,
  };
}

/**
 * Rebuilds thinking from a Chat Completions assistant message. `reasoning_details`
 * carries the provider's continuation data; a bare `reasoning_content` string
 * only restores the text, which pi-ai downgrades to plain text for providers
 * that require a signature.
 */
function thinkingFromChatMessage(model: Model<any>, message: any): ThinkingContent[] {
  const restored = thinkingFromReasoningDetails(model, message?.reasoning_details);
  if (restored.length) return restored;

  const reasoning = message?.reasoning_content;
  if (typeof reasoning !== 'string' || !reasoning) return [];
  return [{ type: 'thinking', thinking: reasoning }];
}

function normalizeToolCall(toolCall: any, id: string): ToolCall {
  const source = toolCall?.function ?? toolCall;
  const rawArguments = source?.arguments;
  return {
    type: 'toolCall',
    id: String(toolCall?.id ?? toolCall?.call_id ?? id),
    name: String(source?.name ?? 'tool'),
    arguments: parseToolArguments(rawArguments),
  };
}

function parseToolArguments(raw: unknown): Record<string, any> {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) return raw as Record<string, any>;
  if (typeof raw !== 'string' || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return { _raw: raw };
  }
}

async function normalizeAssistantTextBlocks(content: any): Promise<TextContent[]> {
  if (typeof content === 'string') {
    return content ? [{ type: 'text', text: content }] : [];
  }
  if (!Array.isArray(content)) return [];

  const blocks: TextContent[] = [];
  for (const part of content) {
    const text =
      part?.type === 'text' || part?.type === 'output_text' || part?.type === 'input_text'
        ? String(part.text ?? '')
        : undefined;
    if (text) {
      blocks.push({ type: 'text', text });
    }
  }
  return blocks;
}

async function normalizeResponsesContent(
  content: any,
): Promise<
  string | Array<{ type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }>
> {
  if (typeof content === 'string') return content;
  return normalizeRichContent(content);
}

async function normalizeRichContent(
  content: any,
): Promise<
  string | Array<{ type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }>
> {
  if (typeof content === 'string') return content;
  const blocks = await normalizeContentBlocks(content);
  if (blocks.length === 1 && blocks[0]?.type === 'text') return blocks[0].text;
  return blocks;
}

async function normalizeContentBlocks(
  content: any,
): Promise<
  Array<{ type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }>
> {
  if (typeof content === 'string') {
    return content ? [{ type: 'text', text: content }] : [];
  }
  if (!Array.isArray(content)) {
    return [{ type: 'text', text: stringifyContentValue(content) }];
  }

  const blocks: Array<
    { type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }
  > = [];
  for (const part of content) {
    const text = extractTextPart(part);
    if (text != null) {
      if (text) blocks.push({ type: 'text', text });
      continue;
    }

    const image = await extractImagePart(part);
    if (image) blocks.push(image);
  }
  return blocks;
}

async function contentToPlainText(content: any): Promise<string> {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return stringifyContentValue(content);

  const texts: string[] = [];
  for (const part of content) {
    const text = extractTextPart(part);
    if (text) texts.push(text);
  }
  return texts.join('\n');
}

function stringifyContentValue(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function extractTextPart(part: any): string | undefined {
  if (!part || typeof part !== 'object') return undefined;
  switch (part.type) {
    case 'text':
    case 'input_text':
    case 'output_text':
      return String(part.text ?? '');
    default:
      return undefined;
  }
}

async function extractImagePart(
  part: any,
): Promise<{ type: 'image'; data: string; mimeType: string } | undefined> {
  if (!part || typeof part !== 'object') return undefined;

  let imageUrl: string | undefined;
  if (part.type === 'image_url') {
    imageUrl = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
  } else if (part.type === 'input_image') {
    imageUrl = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
    imageUrl ??= typeof part.url === 'string' ? part.url : undefined;
  }

  if (!imageUrl) return undefined;
  return imageFromUrl(imageUrl);
}

async function imageFromUrl(
  url: string,
): Promise<{ type: 'image'; data: string; mimeType: string }> {
  if (url.startsWith('data:')) {
    const match = /^data:([^;,]+);base64,(.+)$/i.exec(url);
    if (!match) throw new Error('Unsupported data URL image format');
    return { type: 'image', mimeType: match[1]!, data: match[2]! };
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch image URL: ${response.status} ${response.statusText}`);
  }

  const mimeType = response.headers.get('content-type')?.split(';')[0]?.trim() || 'image/png';
  const bytes = Buffer.from(await response.arrayBuffer());
  return {
    type: 'image',
    mimeType,
    data: bytes.toString('base64'),
  };
}

export function chatCompletionId(message?: AssistantMessage): string {
  return message?.responseId ?? `chatcmpl-${randomUUID()}`;
}

export function createChatCompletionResponse(model: Model<any>, message: AssistantMessage) {
  const reasoning = assistantReasoning(message);
  const reasoningDetails = chatReasoningDetails(model, message);
  const toolCalls = assistantToolCalls(message).map((toolCall) => ({
    id: toolCall.id,
    type: 'function',
    function: {
      name: toolCall.name,
      arguments: JSON.stringify(toolCall.arguments),
    },
  }));

  return {
    id: chatCompletionId(message),
    object: 'chat.completion',
    created: Math.floor(message.timestamp / 1000),
    model: exposedModelId(model),
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: assistantText(message) || null,
          refusal: null,
          annotations: [],
          ...(reasoning ? { reasoning_content: reasoning } : {}),
          ...(reasoningDetails ? { reasoning_details: reasoningDetails } : {}),
          // OpenAI leaves the key out entirely when the model called no tools.
          ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        },
        logprobs: null,
        finish_reason: mapFinishReason(message.stopReason),
      },
    ],
    usage: assistantUsage(message),
    service_tier: SERVICE_TIER,
    system_fingerprint: SYSTEM_FINGERPRINT,
  };
}

/**
 * One streamed chunk. With `stream_options.include_usage` every chunk carries a
 * `usage` key — null until the extra final chunk — and without it the key is
 * absent altogether, which is how OpenAI streams.
 */
export function createChatCompletionChunk(
  id: string,
  created: number,
  modelId: string,
  choices: unknown[],
  options: RenderOptions,
  usage?: unknown,
) {
  return {
    id,
    object: 'chat.completion.chunk',
    created,
    model: modelId,
    service_tier: SERVICE_TIER,
    system_fingerprint: SYSTEM_FINGERPRINT,
    choices,
    ...(options.includeUsage ? { usage: usage ?? null } : {}),
  };
}

export function createResponsesResponse(
  model: Model<any>,
  message: AssistantMessage,
  options: RenderOptions = {},
) {
  const text = assistantText(message);
  const toolCalls = assistantToolCalls(message);
  const output: any[] = [];

  for (const block of assistantThinking(message)) {
    // Each block becomes its own reasoning item, keeping the provider's own item
    // id and ciphertext so the client can hand it straight back.
    const item = reasoningItemFromThinking(
      model,
      block,
      `rs_${randomUUID()}`,
      options.includeEncryptedReasoning === true,
    );
    if (item) output.push(item);
  }

  if (text) {
    output.push({
      id: `msg_${randomUUID()}`,
      type: 'message',
      status: 'completed',
      role: 'assistant',
      content: [{ type: 'output_text', text, annotations: [] }],
    });
  }

  for (const toolCall of toolCalls) {
    output.push({
      id: `fc_${toolCall.id}`,
      type: 'function_call',
      call_id: toolCall.id,
      name: toolCall.name,
      arguments: JSON.stringify(toolCall.arguments),
      status: 'completed',
    });
  }

  return createResponsesEnvelope(model, options, {
    id: message.responseId ?? `resp_${randomUUID()}`,
    createdAt: Math.floor(message.timestamp / 1000),
    status:
      message.stopReason === 'error' || message.stopReason === 'aborted' ? 'failed' : 'completed',
    error: message.errorMessage ? { message: message.errorMessage } : null,
    output,
    usage: {
      input_tokens: message.usage.input,
      input_tokens_details: { cached_tokens: message.usage.cacheRead },
      output_tokens: message.usage.output,
      output_tokens_details: { reasoning_tokens: message.usage.reasoning ?? 0 },
      total_tokens: message.usage.totalTokens,
    },
  });
}

/**
 * The Responses API reports a request's own parameters back on every response,
 * including the ones the client left at their defaults, and repeats the whole
 * object on `response.created`, `response.in_progress`, `response.completed`
 * and `response.failed`.
 */
export function createResponsesEnvelope(
  model: Model<any>,
  options: RenderOptions,
  params: {
    id: string;
    createdAt: number;
    status: 'in_progress' | 'completed' | 'failed';
    error?: { message: string } | null;
    output?: unknown[];
    usage?: unknown;
  },
) {
  const body = options.request ?? {};
  const reasoningEffort = isReasoningEffort(body.reasoning?.effort) ? body.reasoning.effort : null;

  return {
    id: params.id,
    object: 'response',
    created_at: params.createdAt,
    status: params.status,
    error: params.error ?? null,
    incomplete_details: null,
    instructions: typeof body.instructions === 'string' ? body.instructions : null,
    max_output_tokens: typeof body.max_output_tokens === 'number' ? body.max_output_tokens : null,
    model: exposedModelId(model),
    output: params.output ?? [],
    parallel_tool_calls: body.parallel_tool_calls !== false,
    previous_response_id:
      typeof body.previous_response_id === 'string' ? body.previous_response_id : null,
    reasoning: {
      effort: reasoningEffort,
      summary: typeof body.reasoning?.summary === 'string' ? body.reasoning.summary : null,
    },
    service_tier: SERVICE_TIER,
    store: body.store !== false,
    temperature: typeof body.temperature === 'number' ? body.temperature : 1,
    text: body.text ?? { format: { type: 'text' } },
    tool_choice: body.tool_choice ?? 'auto',
    tools: Array.isArray(body.tools) ? body.tools : [],
    top_p: typeof body.top_p === 'number' ? body.top_p : 1,
    truncation: typeof body.truncation === 'string' ? body.truncation : 'disabled',
    usage: params.usage ?? null,
    user: typeof body.user === 'string' ? body.user : null,
    metadata: body.metadata ?? {},
  };
}
