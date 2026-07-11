import { randomUUID } from 'node:crypto';

import type {
  AssistantMessage,
  Context,
  Message,
  Model,
  MutableModels,
  TextContent,
  Tool,
  ToolCall,
  ToolResultMessage,
  UserMessage,
} from '@earendil-works/pi-ai';

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
    },
    completion_tokens_details: {
      reasoning_tokens: message.usage.reasoning ?? 0,
    },
  };
}

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

  const input = body.input;
  if (typeof input === 'string') {
    messages.push({ role: 'user', content: input, timestamp: now });
  } else if (Array.isArray(input)) {
    for (let index = 0; index < input.length; index += 1) {
      const item = input[index];
      const timestamp = now + index;

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
        messages.push(createAssistantHistoryMessage(model, [call], timestamp));
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
          messages.push(await normalizeAssistantHistoryMessage(model, item, timestamp));
          continue;
        }
      }

      throw new Error(`Unsupported responses input item: ${JSON.stringify(item)}`);
    }
  } else if (input != null) {
    throw new Error('Responses input must be a string or an array');
  }

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

export function buildPiOptions(body: any, signal?: AbortSignal) {
  return {
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    maxTokens:
      typeof body.max_output_tokens === 'number'
        ? body.max_output_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : typeof body.max_tokens === 'number'
            ? body.max_tokens
            : undefined,
    signal,
    metadata: typeof body.user === 'string' ? { user_id: body.user } : undefined,
  };
}

export function buildChatPiOptions(body: any, signal?: AbortSignal) {
  const options = buildPiOptions(body, signal);
  return isReasoningEffort(body.reasoning_effort)
    ? { ...options, reasoningEffort: body.reasoning_effort }
    : options;
}

export function buildResponsesPiOptions(body: any, signal?: AbortSignal) {
  const options = buildPiOptions(body, signal);
  return isReasoningEffort(body.reasoning?.effort)
    ? { ...options, reasoningEffort: body.reasoning.effort }
    : options;
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
): Promise<AssistantMessage> {
  const content: AssistantMessage['content'] = [];

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

export function createChatCompletionResponse(model: Model<any>, message: AssistantMessage) {
  return {
    id: message.responseId ?? `chatcmpl_${randomUUID()}`,
    object: 'chat.completion',
    created: Math.floor(message.timestamp / 1000),
    model: exposedModelId(model),
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: assistantText(message) || null,
          tool_calls: assistantToolCalls(message).map((toolCall) => ({
            id: toolCall.id,
            type: 'function',
            function: {
              name: toolCall.name,
              arguments: JSON.stringify(toolCall.arguments),
            },
          })),
        },
        finish_reason: mapFinishReason(message.stopReason),
      },
    ],
    usage: assistantUsage(message),
  };
}

export function createResponsesResponse(model: Model<any>, message: AssistantMessage) {
  const text = assistantText(message);
  const toolCalls = assistantToolCalls(message);
  const output: any[] = [];

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

  return {
    id: message.responseId ?? `resp_${randomUUID()}`,
    object: 'response',
    created_at: Math.floor(message.timestamp / 1000),
    status:
      message.stopReason === 'error' || message.stopReason === 'aborted' ? 'failed' : 'completed',
    error: message.errorMessage ? { message: message.errorMessage } : null,
    incomplete_details: null,
    model: exposedModelId(model),
    output,
    output_text: text,
    parallel_tool_calls: true,
    tools: [],
    usage: {
      input_tokens: message.usage.input,
      output_tokens: message.usage.output,
      total_tokens: message.usage.totalTokens,
      input_tokens_details: { cached_tokens: message.usage.cacheRead },
      output_tokens_details: { reasoning_tokens: message.usage.reasoning ?? 0 },
    },
  };
}
