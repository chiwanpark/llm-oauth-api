import { randomUUID } from 'node:crypto';

import { createModels, type MutableModels } from '@earendil-works/pi-ai';
import Fastify, { type FastifyReply, type FastifyRequest } from 'fastify';

import { JsonCredentialStore } from './credential-store.js';
import {
  assistantUsage,
  buildChatContext,
  buildChatPiOptions,
  buildPiOptions,
  buildResponsesContext,
  createChatCompletionResponse,
  createOpenAIError,
  createOpenAIModelsResponse,
  createResponsesResponse,
  exposedModelId,
  isReasoningEffort,
  mapFinishReason,
  REASONING_EFFORTS,
  resolveModelByName,
} from './openai-compat.js';
import { createSupportedProviders } from './providers.js';

export type ServerOptions = {
  authFile: string;
  providerIds?: string[];
  apiKey: string;
  port: number;
  host: string;
};

export async function startServer(options: ServerOptions): Promise<void> {
  const app = Fastify({ logger: true });
  const models = createModels({ credentials: new JsonCredentialStore(options.authFile) });
  for (const provider of createSupportedProviders(options.providerIds)) {
    models.setProvider(provider);
  }

  app.addHook('onRequest', async (request, reply) => {
    if (!isAuthorized(request, options.apiKey)) {
      reply
        .code(401)
        .send(createOpenAIError('Invalid API key', 'authentication_error', 'invalid_api_key'));
    }
  });

  app.get('/v1/models', async (_request, reply) => {
    const available = await getAvailableModels(models);
    reply.send(createOpenAIModelsResponse(available));
  });

  app.post('/v1/chat/completions', async (request, reply) => {
    await handleChatCompletions(models, request, reply);
  });

  app.post('/v1/responses', async (request, reply) => {
    await handleResponses(models, request, reply);
  });

  await app.listen({ host: options.host, port: options.port });
}

function isAuthorized(request: FastifyRequest, expectedApiKey: string): boolean {
  const authHeader = request.headers.authorization;
  if (typeof authHeader === 'string' && authHeader.startsWith('Bearer ')) {
    return authHeader.slice('Bearer '.length) === expectedApiKey;
  }

  const apiKeyHeader = request.headers['x-api-key'];
  return typeof apiKeyHeader === 'string' && apiKeyHeader === expectedApiKey;
}

async function getAvailableModels(models: MutableModels) {
  const available = [] as ReturnType<MutableModels['getModels']> extends readonly (infer T)[]
    ? T[]
    : never[];
  for (const provider of models.getProviders()) {
    const providerModels = models.getModels(provider.id);
    if (!providerModels.length) continue;

    try {
      const auth = await models.getAuth(providerModels[0]!);
      if (auth) {
        available.push(...providerModels);
      }
    } catch {
      // hide broken/unconfigured providers from /v1/models
    }
  }
  return available;
}

async function handleChatCompletions(
  models: MutableModels,
  request: FastifyRequest,
  reply: FastifyReply,
) {
  const body = request.body as any;
  const requestedModel = typeof body?.model === 'string' ? body.model : undefined;
  if (!requestedModel) {
    reply.code(400).send(createOpenAIError('`model` is required'));
    return;
  }

  if (body.reasoning_effort !== undefined && !isReasoningEffort(body.reasoning_effort)) {
    reply
      .code(400)
      .send(
        createOpenAIError(
          `\`reasoning_effort\` must be one of: ${REASONING_EFFORTS.join(', ')}`,
          'invalid_request_error',
          'invalid_value',
        ),
      );
    return;
  }

  const model = resolveModelByName(models, requestedModel);
  if (!model) {
    reply
      .code(404)
      .send(
        createOpenAIError(
          `Unknown model: ${requestedModel}`,
          'invalid_request_error',
          'model_not_found',
        ),
      );
    return;
  }

  try {
    const auth = await models.getAuth(model);
    if (!auth) {
      reply
        .code(400)
        .send(
          createOpenAIError(
            `Model is not configured: ${requestedModel}`,
            'invalid_request_error',
            'model_not_configured',
          ),
        );
      return;
    }

    const context = await buildChatContext(model, body);
    const signal = createRequestSignal(request, reply);
    const options = buildChatPiOptions(body, signal);

    if (body.stream) {
      await streamChatCompletions(models, model, context, options, reply);
      return;
    }

    const message = await models.complete(model, context, options);
    if (message.stopReason === 'error' || message.stopReason === 'aborted') {
      reply
        .code(502)
        .send(createOpenAIError(message.errorMessage ?? 'Upstream model error', 'api_error'));
      return;
    }

    reply.send(createChatCompletionResponse(model, message));
  } catch (error) {
    request.log.error({ err: error }, 'chat completions failed');
    reply.code(500).send(createOpenAIError(errorMessage(error), 'api_error'));
  }
}

async function handleResponses(
  models: MutableModels,
  request: FastifyRequest,
  reply: FastifyReply,
) {
  const body = request.body as any;
  const requestedModel = typeof body?.model === 'string' ? body.model : undefined;
  if (!requestedModel) {
    reply.code(400).send(createOpenAIError('`model` is required'));
    return;
  }

  const model = resolveModelByName(models, requestedModel);
  if (!model) {
    reply
      .code(404)
      .send(
        createOpenAIError(
          `Unknown model: ${requestedModel}`,
          'invalid_request_error',
          'model_not_found',
        ),
      );
    return;
  }

  try {
    const auth = await models.getAuth(model);
    if (!auth) {
      reply
        .code(400)
        .send(
          createOpenAIError(
            `Model is not configured: ${requestedModel}`,
            'invalid_request_error',
            'model_not_configured',
          ),
        );
      return;
    }

    const context = await buildResponsesContext(model, body);
    const signal = createRequestSignal(request, reply);
    const options = buildPiOptions(body, signal);

    if (body.stream) {
      await streamResponses(models, model, context, options, reply);
      return;
    }

    const message = await models.complete(model, context, options);
    if (message.stopReason === 'error' || message.stopReason === 'aborted') {
      reply
        .code(502)
        .send(createOpenAIError(message.errorMessage ?? 'Upstream model error', 'api_error'));
      return;
    }

    reply.send(createResponsesResponse(model, message));
  } catch (error) {
    request.log.error({ err: error }, 'responses failed');
    reply.code(500).send(createOpenAIError(errorMessage(error), 'api_error'));
  }
}

function createRequestSignal(request: FastifyRequest, reply: FastifyReply): AbortSignal {
  const controller = new AbortController();
  const abort = () => controller.abort();
  request.raw.once('aborted', abort);
  request.raw.once('close', abort);
  reply.raw.once('close', abort);
  return controller.signal;
}

async function streamChatCompletions(
  models: MutableModels,
  model: any,
  context: any,
  options: any,
  reply: FastifyReply,
) {
  prepareSse(reply);

  const id = `chatcmpl_${randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  const modelId = exposedModelId(model);
  const toolCallIndexes = new Map<number, number>();
  let nextToolCallIndex = 0;

  const stream = models.stream(model, context, options);

  writeSseData(reply, {
    id,
    object: 'chat.completion.chunk',
    created,
    model: modelId,
    choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
  });

  for await (const event of stream) {
    if (event.type === 'text_delta') {
      writeSseData(reply, {
        id,
        object: 'chat.completion.chunk',
        created,
        model: modelId,
        choices: [{ index: 0, delta: { content: event.delta }, finish_reason: null }],
      });
      continue;
    }

    if (event.type === 'toolcall_start') {
      const partial = event.partial.content[event.contentIndex];
      if (partial?.type !== 'toolCall') continue;
      const toolIndex = nextToolCallIndex++;
      toolCallIndexes.set(event.contentIndex, toolIndex);
      writeSseData(reply, {
        id,
        object: 'chat.completion.chunk',
        created,
        model: modelId,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: toolIndex,
                  id: partial.id,
                  type: 'function',
                  function: { name: partial.name, arguments: '' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      });
      continue;
    }

    if (event.type === 'toolcall_delta') {
      const toolIndex = toolCallIndexes.get(event.contentIndex) ?? nextToolCallIndex++;
      toolCallIndexes.set(event.contentIndex, toolIndex);
      writeSseData(reply, {
        id,
        object: 'chat.completion.chunk',
        created,
        model: modelId,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [{ index: toolIndex, function: { arguments: event.delta } }],
            },
            finish_reason: null,
          },
        ],
      });
      continue;
    }

    if (event.type === 'done') {
      writeSseData(reply, {
        id,
        object: 'chat.completion.chunk',
        created,
        model: modelId,
        choices: [
          { index: 0, delta: {}, finish_reason: mapFinishReason(event.message.stopReason) },
        ],
        usage: assistantUsage(event.message),
      });
      writeSseDone(reply);
      return;
    }

    if (event.type === 'error') {
      writeSseData(
        reply,
        createOpenAIError(event.error.errorMessage ?? 'Upstream model error', 'api_error'),
      );
      writeSseDone(reply);
      return;
    }
  }

  writeSseDone(reply);
}

async function streamResponses(
  models: MutableModels,
  model: any,
  context: any,
  options: any,
  reply: FastifyReply,
) {
  prepareSse(reply);

  const responseId = `resp_${randomUUID()}`;
  const modelId = exposedModelId(model);
  const createdAt = Math.floor(Date.now() / 1000);
  const stream = models.stream(model, context, options);

  let assistantOutputIndex: number | undefined;
  let assistantItemId: string | undefined;
  let assistantText = '';
  let assistantContentIndex = 0;
  const toolOutputIndexes = new Map<number, number>();
  const toolItemIds = new Map<number, string>();
  let nextOutputIndex = 0;

  writeSseEvent(reply, 'response.created', {
    type: 'response.created',
    response: {
      id: responseId,
      object: 'response',
      created_at: createdAt,
      model: modelId,
      status: 'in_progress',
    },
  });
  writeSseEvent(reply, 'response.in_progress', {
    type: 'response.in_progress',
    response: {
      id: responseId,
      object: 'response',
      created_at: createdAt,
      model: modelId,
      status: 'in_progress',
    },
  });

  for await (const event of stream) {
    if (event.type === 'text_start') {
      assistantItemId ??= `msg_${randomUUID()}`;
      assistantOutputIndex ??= nextOutputIndex++;
      writeSseEvent(reply, 'response.output_item.added', {
        type: 'response.output_item.added',
        response_id: responseId,
        output_index: assistantOutputIndex,
        item: {
          id: assistantItemId,
          type: 'message',
          role: 'assistant',
          status: 'in_progress',
          content: [],
        },
      });
      writeSseEvent(reply, 'response.content_part.added', {
        type: 'response.content_part.added',
        response_id: responseId,
        output_index: assistantOutputIndex,
        item_id: assistantItemId,
        content_index: assistantContentIndex,
        part: { type: 'output_text', text: '' },
      });
      continue;
    }

    if (event.type === 'text_delta' && assistantItemId != null && assistantOutputIndex != null) {
      assistantText += event.delta;
      writeSseEvent(reply, 'response.output_text.delta', {
        type: 'response.output_text.delta',
        response_id: responseId,
        output_index: assistantOutputIndex,
        item_id: assistantItemId,
        content_index: assistantContentIndex,
        delta: event.delta,
      });
      continue;
    }

    if (event.type === 'text_end' && assistantItemId != null && assistantOutputIndex != null) {
      writeSseEvent(reply, 'response.output_text.done', {
        type: 'response.output_text.done',
        response_id: responseId,
        output_index: assistantOutputIndex,
        item_id: assistantItemId,
        content_index: assistantContentIndex,
        text: event.content,
      });
      writeSseEvent(reply, 'response.content_part.done', {
        type: 'response.content_part.done',
        response_id: responseId,
        output_index: assistantOutputIndex,
        item_id: assistantItemId,
        content_index: assistantContentIndex,
        part: { type: 'output_text', text: event.content },
      });
      assistantContentIndex += 1;
      continue;
    }

    if (event.type === 'toolcall_start') {
      const partial = event.partial.content[event.contentIndex];
      if (partial?.type !== 'toolCall') continue;
      const outputIndex = nextOutputIndex++;
      const itemId = `fc_${partial.id}`;
      toolOutputIndexes.set(event.contentIndex, outputIndex);
      toolItemIds.set(event.contentIndex, itemId);
      writeSseEvent(reply, 'response.output_item.added', {
        type: 'response.output_item.added',
        response_id: responseId,
        output_index: outputIndex,
        item: {
          id: itemId,
          type: 'function_call',
          call_id: partial.id,
          name: partial.name,
          arguments: '',
          status: 'in_progress',
        },
      });
      continue;
    }

    if (event.type === 'toolcall_delta') {
      const outputIndex = toolOutputIndexes.get(event.contentIndex);
      const itemId = toolItemIds.get(event.contentIndex);
      if (outputIndex == null || itemId == null) continue;
      writeSseEvent(reply, 'response.function_call_arguments.delta', {
        type: 'response.function_call_arguments.delta',
        response_id: responseId,
        output_index: outputIndex,
        item_id: itemId,
        delta: event.delta,
      });
      continue;
    }

    if (event.type === 'toolcall_end') {
      const outputIndex = toolOutputIndexes.get(event.contentIndex);
      const itemId = toolItemIds.get(event.contentIndex);
      if (outputIndex == null || itemId == null) continue;
      writeSseEvent(reply, 'response.function_call_arguments.done', {
        type: 'response.function_call_arguments.done',
        response_id: responseId,
        output_index: outputIndex,
        item_id: itemId,
        arguments: JSON.stringify(event.toolCall.arguments),
      });
      writeSseEvent(reply, 'response.output_item.done', {
        type: 'response.output_item.done',
        response_id: responseId,
        output_index: outputIndex,
        item: {
          id: itemId,
          type: 'function_call',
          call_id: event.toolCall.id,
          name: event.toolCall.name,
          arguments: JSON.stringify(event.toolCall.arguments),
          status: 'completed',
        },
      });
      continue;
    }

    if (event.type === 'done') {
      if (assistantItemId != null && assistantOutputIndex != null) {
        writeSseEvent(reply, 'response.output_item.done', {
          type: 'response.output_item.done',
          response_id: responseId,
          output_index: assistantOutputIndex,
          item: {
            id: assistantItemId,
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: assistantText
              ? [{ type: 'output_text', text: assistantText, annotations: [] }]
              : [],
          },
        });
      }
      const response = createResponsesResponse(model, { ...event.message, responseId });
      writeSseEvent(reply, 'response.completed', {
        type: 'response.completed',
        response,
      });
      reply.raw.end();
      return;
    }

    if (event.type === 'error') {
      writeSseEvent(reply, 'response.failed', {
        type: 'response.failed',
        response: {
          id: responseId,
          object: 'response',
          created_at: createdAt,
          model: modelId,
          status: 'failed',
          error: { message: event.error.errorMessage ?? 'Upstream model error' },
        },
      });
      reply.raw.end();
      return;
    }
  }

  reply.raw.end();
}

function prepareSse(reply: FastifyReply) {
  reply.raw.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
}

function writeSseData(reply: FastifyReply, payload: unknown) {
  reply.raw.write(`data: ${JSON.stringify(payload)}\n\n`);
}

function writeSseEvent(reply: FastifyReply, event: string, payload: unknown) {
  reply.raw.write(`event: ${event}\n`);
  reply.raw.write(`data: ${JSON.stringify(payload)}\n\n`);
}

function writeSseDone(reply: FastifyReply) {
  reply.raw.write('data: [DONE]\n\n');
  reply.raw.end();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
