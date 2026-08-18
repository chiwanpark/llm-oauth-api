import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import fastifyStatic from '@fastify/static';
import {
  createModels,
  type AssistantMessage,
  type Model,
  type MutableModels,
  type OAuthCredential,
  type Provider,
} from '@earendil-works/pi-ai';
import Fastify, {
  type FastifyBaseLogger,
  type FastifyInstance,
  type FastifyReply,
  type FastifyRequest,
} from 'fastify';

import { JsonCredentialStore } from './credential-store.js';
import {
  createModelCooldown,
  DEFAULT_MODEL_COOLDOWN_MS,
  describeSkip,
  type ModelCooldown,
} from './model-cooldown.js';
import {
  DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS,
  DEFAULT_OAUTH_REFRESH_INTERVAL_MS,
  startOAuthRefreshScheduler,
} from './oauth-refresh.js';
import {
  assistantUsage,
  buildChatContext,
  buildChatPiOptions,
  buildRenderOptions,
  buildResponsesContext,
  buildResponsesPiOptions,
  chatCompletionId,
  chatReasoningDetails,
  createChatCompletionChunk,
  createChatCompletionResponse,
  createOpenAIError,
  createOpenAIModelsResponse,
  createResponsesEnvelope,
  createResponsesResponse,
  exposedModelId,
  isReasoningEffort,
  mapFinishReason,
  REASONING_EFFORTS,
  type RenderOptions,
} from './openai-compat.js';
import { reasoningItemFromThinking } from './reasoning.js';
import {
  describeGroup,
  findGroup,
  groupModelEntries,
  resolveModelCandidates,
  type ModelGroup,
} from './groups.js';
import { createSupportedProviders } from './providers.js';

export type ServerOptions = {
  authFile: string;
  providerIds?: string[];
  groups?: ModelGroup[];
  apiKey: string;
  port: number;
  host: string;
  /** How long a model that failed is passed over; 0 disables skipping. */
  modelCooldownMs?: number;
  oauthAutoRefresh?: boolean;
  oauthRefreshIntervalMs?: number;
  oauthRefreshBeforeExpiryMs?: number;
};

export async function startServer(options: ServerOptions): Promise<void> {
  const app = Fastify({ logger: true });
  const credentials = new JsonCredentialStore(options.authFile);
  const providers = createSupportedProviders(options.providerIds).map((provider) =>
    withTokenRefreshLogging(provider, app.log),
  );
  const models = createModels({ credentials });
  for (const provider of providers) {
    models.setProvider(provider);
  }

  const oauthRefreshScheduler =
    options.oauthAutoRefresh === false
      ? undefined
      : startOAuthRefreshScheduler({
          credentials,
          providers,
          logger: app.log,
          intervalMs: options.oauthRefreshIntervalMs ?? DEFAULT_OAUTH_REFRESH_INTERVAL_MS,
          refreshBeforeExpiryMs:
            options.oauthRefreshBeforeExpiryMs ?? DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS,
        });

  app.addHook('onClose', async () => {
    await oauthRefreshScheduler?.stop();
  });

  app.addHook('onRequest', async (request, reply) => {
    if (
      (request.url === '/v1' || request.url.startsWith('/v1/')) &&
      !isAuthorized(request, options.apiKey)
    ) {
      reply
        .code(401)
        .send(createOpenAIError('Invalid API key', 'authentication_error', 'invalid_api_key'));
    }
  });

  const groups = options.groups ?? [];
  const cooldown = createModelCooldown({
    cooldownMs: options.modelCooldownMs ?? DEFAULT_MODEL_COOLDOWN_MS,
  });
  for (const group of groups) {
    app.log.info(
      { group: group.name, members: describeGroup(group), cooldownMs: cooldown.cooldownMs },
      'Model group registered',
    );
  }

  app.get('/v1/models', async (_request, reply) => {
    const available = await getAvailableModels(models);
    const response = createOpenAIModelsResponse(available);
    response.data.push(...groupModelEntries(groups, available));
    reply.send(response);
  });

  app.post('/v1/chat/completions', async (request, reply) => {
    await handleChatCompletions(models, groups, request, reply, cooldown);
  });

  app.post('/v1/responses', async (request, reply) => {
    await handleResponses(models, groups, request, reply, cooldown);
  });

  await registerClient(app);
  try {
    await app.listen({ host: options.host, port: options.port });
  } catch (error) {
    await oauthRefreshScheduler?.stop();
    throw error;
  }
}

async function registerClient(app: FastifyInstance): Promise<void> {
  const clientRoot = findClientRoot();
  if (!clientRoot) {
    app.get('/', async (_request, reply) => {
      reply
        .code(503)
        .type('text/plain')
        .send('API playground assets are missing. Run `pnpm build` before starting the server.');
    });
    return;
  }

  await app.register(fastifyStatic, {
    root: clientRoot,
    maxAge: '30d',
    immutable: true,
  });

  app.get('/', async (_request, reply) => {
    return reply.sendFile('index.html', { maxAge: 0, immutable: false });
  });
}

function findClientRoot(): string | undefined {
  const moduleDirectory = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    resolve(moduleDirectory, 'client'),
    resolve(moduleDirectory, '../dist/client'),
  ];
  return candidates.find((candidate) => existsSync(resolve(candidate, 'index.html')));
}

function withTokenRefreshLogging(provider: Provider, logger: FastifyBaseLogger): Provider {
  const oauth = provider.auth.oauth;
  if (!oauth) return provider;

  const refresh = oauth.refresh.bind(oauth);
  oauth.refresh = async (credential: OAuthCredential, signal: AbortSignal) => {
    const now = Date.now();
    logger.warn(
      {
        providerId: provider.id,
        expiresAt: formatTimestamp(credential.expires),
        expiresInMs: credential.expires - now,
      },
      'OAuth access token refresh started',
    );

    try {
      const refreshed = await refresh(credential, signal);
      logger.info(
        {
          providerId: provider.id,
          expiresAt: formatTimestamp(refreshed.expires),
        },
        'OAuth access token refreshed',
      );
      return refreshed;
    } catch (error) {
      logger.error({ err: error, providerId: provider.id }, 'OAuth access token refresh failed');
      throw error;
    }
  };

  return provider;
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? String(timestamp) : date.toISOString();
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

/**
 * Signals that an attempt failed before anything was written to the client, so
 * the caller is still free to try the next provider in a group.
 */
class UpstreamAttemptError extends Error {
  constructor(
    message: string,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = 'UpstreamAttemptError';
  }
}

type AttemptFailure = {
  kind: 'unconfigured' | 'upstream' | 'exception' | 'cooling';
  model: Model<any>;
  message: string;
};

/** Used when no store is supplied, so every candidate is always attempted. */
const NO_COOLDOWN = createModelCooldown({ cooldownMs: 0 });

/**
 * How a committed stream ended.
 *
 * A stream that has already reached the client reports its own failure inline
 * instead of throwing, so the outcome has to be handed back explicitly for the
 * caller to be able to tell a working model from a broken one.
 */
export type StreamOutcome = {
  /** The failure the provider reported, absent when the stream completed. */
  error?: AssistantMessage;
};

/** Per-endpoint wiring; the fallback logic itself is shared. */
type EndpointAdapter = {
  label: string;
  buildContext(model: Model<any>, body: any): Promise<any>;
  buildOptions(body: any, signal: AbortSignal): any;
  render(model: Model<any>, message: AssistantMessage, render: RenderOptions): unknown;
  stream(
    models: MutableModels,
    model: Model<any>,
    context: any,
    options: any,
    reply: FastifyReply,
    logger: FastifyBaseLogger,
    allowFailover: boolean,
    render: RenderOptions,
  ): Promise<StreamOutcome>;
};

export const chatAdapter: EndpointAdapter = {
  label: 'chat completions',
  buildContext: buildChatContext,
  buildOptions: buildChatPiOptions,
  render: createChatCompletionResponse,
  stream: streamChatCompletions,
};

export const responsesAdapter: EndpointAdapter = {
  label: 'responses',
  buildContext: buildResponsesContext,
  buildOptions: buildResponsesPiOptions,
  render: createResponsesResponse,
  stream: streamResponses,
};

async function handleChatCompletions(
  models: MutableModels,
  groups: readonly ModelGroup[],
  request: FastifyRequest,
  reply: FastifyReply,
  cooldown?: ModelCooldown,
) {
  const body = request.body as any;

  if (body?.reasoning_effort !== undefined && !isReasoningEffort(body.reasoning_effort)) {
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

  if (body?.stream_options !== undefined && body?.stream !== true) {
    reply
      .code(400)
      .send(
        createOpenAIError(
          '`stream_options` can only be used when `stream` is true',
          'invalid_request_error',
          'invalid_value',
        ),
      );
    return;
  }

  if (!validateInclude(body, reply)) return;

  await runCompletion(models, groups, request, reply, chatAdapter, cooldown);
}

async function handleResponses(
  models: MutableModels,
  groups: readonly ModelGroup[],
  request: FastifyRequest,
  reply: FastifyReply,
  cooldown?: ModelCooldown,
) {
  const body = request.body as any;

  if (body?.reasoning?.effort !== undefined && !isReasoningEffort(body.reasoning.effort)) {
    reply
      .code(400)
      .send(
        createOpenAIError(
          `\`reasoning.effort\` must be one of: ${REASONING_EFFORTS.join(', ')}`,
          'invalid_request_error',
          'invalid_value',
        ),
      );
    return;
  }

  if (!validateInclude(body, reply)) return;

  await runCompletion(models, groups, request, reply, responsesAdapter, cooldown);
}

/**
 * Only the shape of `include` is checked. Unknown entries are ignored rather
 * than rejected, so clients can keep asking for fields this proxy does not
 * produce yet.
 */
function validateInclude(body: any, reply: FastifyReply): boolean {
  const include = body?.include;
  if (include === undefined || include === null) return true;

  if (!Array.isArray(include) || include.some((entry) => typeof entry !== 'string')) {
    reply
      .code(400)
      .send(
        createOpenAIError(
          '`include` must be an array of strings',
          'invalid_request_error',
          'invalid_value',
        ),
      );
    return false;
  }

  return true;
}

/**
 * Runs a request against its candidate models, falling back through a group
 * until one succeeds.
 *
 * Fallback only happens while nothing has been sent to the client. Once a
 * streaming response is committed to a provider the client already holds
 * partial output, so a later failure is reported on that stream instead of
 * being retried elsewhere.
 *
 * A model that failed is remembered for the length of the cooldown window, so
 * later requests start at the next member instead of paying for the same
 * failure again. The failure is still reported, because a skipped member is
 * part of the explanation when the whole group comes up empty.
 */
export async function runCompletion(
  models: MutableModels,
  groups: readonly ModelGroup[],
  request: FastifyRequest,
  reply: FastifyReply,
  adapter: EndpointAdapter,
  cooldown: ModelCooldown = NO_COOLDOWN,
): Promise<void> {
  const body = request.body as any;
  const requestedModel = typeof body?.model === 'string' ? body.model : undefined;
  if (!requestedModel) {
    reply.code(400).send(createOpenAIError('`model` is required'));
    return;
  }

  const group = findGroup(groups, requestedModel);
  const candidates = resolveModelCandidates(models, groups, requestedModel);
  if (!candidates.length) {
    // A group whose members are all missing from their catalogs is a different
    // problem from a typo, so say which models were looked for.
    const message = group
      ? `No model in group "${group.name}" is available (${describeGroup(group)})`
      : `Unknown model: ${requestedModel}`;
    reply.code(404).send(createOpenAIError(message, 'invalid_request_error', 'model_not_found'));
    return;
  }

  const { attempts, skipped } = cooldown.select(candidates);
  if (skipped.length) {
    request.log.info(
      {
        requestedModel,
        skipped: skipped.map((skip) => ({
          model: exposedModelId(skip.model),
          reason: skip.reason,
          remainingMs: skip.remainingMs,
        })),
      },
      `${adapter.label} skipping models that recently failed`,
    );
  }

  const allowFailover = attempts.length > 1;
  const signal = createRequestSignal(request, reply);
  // Skipped members are failures the group already knows about, so they belong
  // in the report if nothing else works out either.
  const failures: AttemptFailure[] = skipped.map((skip) => ({
    kind: 'cooling',
    model: skip.model,
    message: describeSkip(skip),
  }));

  for (const [index, model] of attempts.entries()) {
    if (signal.aborted) return;

    const modelId = exposedModelId(model);
    try {
      const auth = await models.getAuth(model);
      if (!auth) {
        // Missing credentials are a configuration gap rather than a model that
        // stopped responding, and cost nothing to detect, so they never start a
        // cooldown.
        failures.push({ kind: 'unconfigured', model, message: 'not configured' });
        logFallback(
          request.log,
          adapter,
          requestedModel,
          modelId,
          attempts,
          index,
          'not configured',
        );
        continue;
      }

      const context = await adapter.buildContext(model, body);
      const options = adapter.buildOptions(body, signal);
      const render = buildRenderOptions(body);

      if (body.stream) {
        const outcome = await adapter.stream(
          models,
          model,
          context,
          options,
          reply,
          request.log,
          allowFailover,
          render,
        );
        // A committed stream reports its failure on the stream itself, so the
        // model is scored from the outcome rather than from how this call
        // returned. Without failover — a plain model request, or the last
        // member of a group — this is the only place that failure is visible.
        recordStreamOutcome(cooldown, model, outcome);
        return;
      }

      const message = await models.completeSimple(model, context, options);
      logModelResponse(request.log, model, message);

      // A client disconnect is not an upstream fault; another provider would
      // fail the same way and nobody is listening for the result.
      if (message.stopReason === 'aborted') {
        reply
          .code(502)
          .send(createOpenAIError(message.errorMessage ?? 'Upstream model error', 'api_error'));
        return;
      }

      if (message.stopReason === 'error') {
        const detail = message.errorMessage ?? 'Upstream model error';
        cooldown.record(model, detail);
        failures.push({ kind: 'upstream', model, message: detail });
        logFallback(request.log, adapter, requestedModel, modelId, attempts, index, detail);
        continue;
      }

      cooldown.clear(model);
      reply.send(adapter.render(model, message, render));
      return;
    } catch (error) {
      const detail = error instanceof UpstreamAttemptError ? error.message : errorMessage(error);
      // A client that disappeared mid-request is not the model's failure, and
      // every provider would look broken by the end of a cancelled stream.
      if (!signal.aborted) cooldown.record(model, detail);

      if (reply.raw.headersSent) {
        // The stream is already committed to this provider; failing over now
        // would splice two providers' output into one response. The failure is
        // still the model's, so it counts against it for later requests.
        request.log.error(
          { err: error, model: modelId },
          `${adapter.label} failed after the response was committed`,
        );
        if (!reply.raw.writableEnded) reply.raw.end();
        return;
      }

      failures.push({ kind: 'exception', model, message: detail });
      request.log.error({ err: error, model: modelId }, `${adapter.label} failed`);
      logFallback(request.log, adapter, requestedModel, modelId, attempts, index, detail);
    }
  }

  sendExhausted(reply, requestedModel, group !== undefined, failures);
}

/**
 * Scores a model from how its stream ended.
 *
 * Streaming and non-streaming failures are treated the same way: an upstream
 * error counts against the model even when part of the answer already reached
 * the client, because the next request has no reason to expect better.
 */
function recordStreamOutcome(
  cooldown: ModelCooldown,
  model: Model<any>,
  outcome: StreamOutcome,
): void {
  const failure = outcome.error;
  if (!failure) {
    cooldown.clear(model);
    return;
  }

  // The provider reports a client disconnect the same way it reports its own
  // faults; only the latter says anything about the model's health.
  if (failure.stopReason === 'aborted') return;

  cooldown.record(model, failure.errorMessage ?? 'Upstream model error');
}

function logFallback(
  logger: FastifyBaseLogger,
  adapter: EndpointAdapter,
  requestedModel: string,
  modelId: string,
  candidates: readonly Model<any>[],
  index: number,
  reason: string,
): void {
  const next = candidates[index + 1];
  if (!next) return;
  logger.warn(
    {
      requestedModel,
      failedModel: modelId,
      nextModel: exposedModelId(next),
      reason,
    },
    `${adapter.label} falling back to the next provider in the group`,
  );
}

/** Reports the outcome once every candidate has failed. */
function sendExhausted(
  reply: FastifyReply,
  requestedModel: string,
  isGroup: boolean,
  failures: readonly AttemptFailure[],
): void {
  // Plain model requests keep their original, non-aggregated responses. They
  // resolve to a single candidate, which is never skipped, so the first failure
  // is always a real attempt.
  if (!isGroup) {
    const failure = failures[0]!;
    if (failure.kind === 'unconfigured') {
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
    reply
      .code(failure.kind === 'upstream' ? 502 : 500)
      .send(createOpenAIError(failure.message, 'api_error'));
    return;
  }

  const detail = failures
    .map((failure) => `${exposedModelId(failure.model)}: ${failure.message}`)
    .join('; ');

  // Skipped members carry no information about configuration, so the shape of
  // the error is decided by the members that were actually tried.
  const attempted = failures.filter((failure) => failure.kind !== 'cooling');
  if (attempted.length && attempted.every((failure) => failure.kind === 'unconfigured')) {
    reply
      .code(400)
      .send(
        createOpenAIError(
          `No model in group "${requestedModel}" is configured (${detail})`,
          'invalid_request_error',
          'model_not_configured',
        ),
      );
    return;
  }

  reply
    .code(502)
    .send(
      createOpenAIError(`All models in group "${requestedModel}" failed (${detail})`, 'api_error'),
    );
}

/**
 * Aborts only when the client actually goes away.
 *
 * Both streams emit `close` on the happy path too: the request closes once its
 * body has been read, and the reply closes once the response has been written.
 * Treating those as disconnects cancels in-flight upstream calls, so each one
 * is qualified by whether the transfer actually finished.
 */
export function createRequestSignal(request: FastifyRequest, reply: FastifyReply): AbortSignal {
  const controller = new AbortController();
  const abort = () => controller.abort();

  request.raw.once('aborted', abort);
  request.raw.once('close', () => {
    if (!request.raw.complete) abort();
  });
  reply.raw.once('close', () => {
    if (!reply.raw.writableEnded) abort();
  });

  return controller.signal;
}

/**
 * Starts an upstream stream, optionally holding back the response until the
 * first event proves the provider is actually answering.
 *
 * With `allowFailover`, a failure reported as the very first event is raised as
 * an `UpstreamAttemptError` while the response is still uncommitted, which lets
 * the caller try the next provider in the group. Without it the stream is
 * returned untouched and errors are reported inline on the SSE stream.
 */
async function beginStream(
  models: MutableModels,
  model: any,
  context: any,
  options: any,
  allowFailover: boolean,
): Promise<AsyncIterable<any>> {
  const stream = models.streamSimple(model, context, options);
  if (!allowFailover) return stream;

  const iterator = stream[Symbol.asyncIterator]();
  let first;
  try {
    first = await iterator.next();
  } catch (error) {
    throw new UpstreamAttemptError(errorMessage(error), error);
  }

  if (!first.done && first.value?.type === 'error') {
    throw new UpstreamAttemptError(
      first.value.error?.errorMessage ?? 'Upstream model error',
      first.value.error,
    );
  }

  return {
    async *[Symbol.asyncIterator]() {
      if (first.done) return;
      yield first.value;
      while (true) {
        const next = await iterator.next();
        if (next.done) return;
        yield next.value;
      }
    },
  };
}

export async function streamChatCompletions(
  models: MutableModels,
  model: any,
  context: any,
  options: any,
  reply: FastifyReply,
  logger: FastifyBaseLogger,
  allowFailover = false,
  render: RenderOptions = {},
): Promise<StreamOutcome> {
  const stream = await beginStream(models, model, context, options, allowFailover);

  prepareSse(reply);

  const id = chatCompletionId();
  const created = Math.floor(Date.now() / 1000);
  const modelId = exposedModelId(model);
  const toolCallIndexes = new Map<number, number>();
  let nextToolCallIndex = 0;

  const writeChunk = (delta: unknown, finishReason: string | null = null) => {
    writeSseData(
      reply,
      createChatCompletionChunk(
        id,
        created,
        modelId,
        [{ index: 0, delta, logprobs: null, finish_reason: finishReason }],
        render,
      ),
    );
  };

  writeChunk({ role: 'assistant', content: '', refusal: null });

  for await (const event of stream) {
    if (event.type === 'thinking_delta') {
      writeChunk({ reasoning_content: event.delta });
      continue;
    }

    if (event.type === 'text_delta') {
      writeChunk({ content: event.delta });
      continue;
    }

    if (event.type === 'toolcall_start') {
      const partial = event.partial.content[event.contentIndex];
      if (partial?.type !== 'toolCall') continue;
      const toolIndex = nextToolCallIndex++;
      toolCallIndexes.set(event.contentIndex, toolIndex);
      writeChunk({
        tool_calls: [
          {
            index: toolIndex,
            id: partial.id,
            type: 'function',
            function: { name: partial.name, arguments: '' },
          },
        ],
      });
      continue;
    }

    if (event.type === 'toolcall_delta') {
      const toolIndex = toolCallIndexes.get(event.contentIndex) ?? nextToolCallIndex++;
      toolCallIndexes.set(event.contentIndex, toolIndex);
      writeChunk({
        tool_calls: [{ index: toolIndex, function: { arguments: event.delta } }],
      });
      continue;
    }

    if (event.type === 'done') {
      logModelResponse(logger, model, event.message);
      // Providers attach continuation data as each reasoning block closes, so
      // the complete set is only known once the message is finished.
      const reasoningDetails = chatReasoningDetails(model, event.message);
      writeChunk(
        reasoningDetails ? { reasoning_details: reasoningDetails } : {},
        mapFinishReason(event.message.stopReason),
      );
      // Usage rides on an extra chunk that carries no choices, and only when the
      // client asked for it through `stream_options`.
      if (render.includeUsage) {
        writeSseData(
          reply,
          createChatCompletionChunk(
            id,
            created,
            modelId,
            [],
            render,
            assistantUsage(event.message),
          ),
        );
      }
      writeSseDone(reply);
      return {};
    }

    if (event.type === 'error') {
      logModelResponse(logger, model, event.error);
      writeSseData(
        reply,
        createOpenAIError(event.error.errorMessage ?? 'Upstream model error', 'api_error'),
      );
      writeSseDone(reply);
      return { error: event.error };
    }
  }

  writeSseDone(reply);
  return {};
}

export async function streamResponses(
  models: MutableModels,
  model: any,
  context: any,
  options: any,
  reply: FastifyReply,
  logger: FastifyBaseLogger,
  allowFailover = false,
  render: RenderOptions = {},
): Promise<StreamOutcome> {
  const stream = await beginStream(models, model, context, options, allowFailover);

  prepareSse(reply);

  const responseId = `resp_${randomUUID()}`;
  const createdAt = Math.floor(Date.now() / 1000);

  const reasoningItems = new Map<number, { outputIndex: number; itemId: string }>();
  let assistantOutputIndex: number | undefined;
  let assistantItemId: string | undefined;
  let assistantText = '';
  let assistantContentIndex = 0;
  const toolOutputIndexes = new Map<number, number>();
  const toolItemIds = new Map<number, string>();
  let nextOutputIndex = 0;

  const emit = sseEmitter(reply);
  const inProgress = () =>
    createResponsesEnvelope(model, render, {
      id: responseId,
      createdAt,
      status: 'in_progress',
    });

  emit('response.created', { response: inProgress() });
  emit('response.in_progress', { response: inProgress() });

  for await (const event of stream) {
    if (event.type === 'thinking_start') {
      const item = {
        itemId: `rs_${randomUUID()}`,
        outputIndex: nextOutputIndex++,
      };
      reasoningItems.set(event.contentIndex, item);
      emit('response.output_item.added', {
        output_index: item.outputIndex,
        item: {
          id: item.itemId,
          type: 'reasoning',
          status: 'in_progress',
          summary: [],
        },
      });
      emit('response.reasoning_summary_part.added', {
        item_id: item.itemId,
        output_index: item.outputIndex,
        summary_index: 0,
        part: { type: 'summary_text', text: '' },
      });
      continue;
    }

    if (event.type === 'thinking_delta') {
      const item = reasoningItems.get(event.contentIndex);
      if (!item) continue;
      emit('response.reasoning_summary_text.delta', {
        item_id: item.itemId,
        output_index: item.outputIndex,
        summary_index: 0,
        delta: event.delta,
      });
      continue;
    }

    if (event.type === 'thinking_end') {
      const item = reasoningItems.get(event.contentIndex);
      if (!item) continue;
      // The provider only attaches its reasoning item — id and ciphertext
      // included — as the block closes, so the finished item is the first place
      // it can be reported. Its id replaces the placeholder used by the summary
      // events above, which keeps the item identical to the one repeated in
      // `response.completed`.
      const block = event.partial?.content?.[event.contentIndex];
      const finished =
        block?.type === 'thinking'
          ? reasoningItemFromThinking(
              model,
              block,
              item.itemId,
              render.includeEncryptedReasoning === true,
            )
          : undefined;
      emit('response.reasoning_summary_text.done', {
        item_id: item.itemId,
        output_index: item.outputIndex,
        summary_index: 0,
        text: event.content,
      });
      emit('response.reasoning_summary_part.done', {
        item_id: item.itemId,
        output_index: item.outputIndex,
        summary_index: 0,
        part: { type: 'summary_text', text: event.content },
      });
      emit('response.output_item.done', {
        output_index: item.outputIndex,
        item: finished ?? {
          id: item.itemId,
          type: 'reasoning',
          status: 'completed',
          summary: event.content ? [{ type: 'summary_text', text: event.content }] : [],
        },
      });
      reasoningItems.delete(event.contentIndex);
      continue;
    }

    if (event.type === 'text_start') {
      assistantItemId ??= `msg_${randomUUID()}`;
      assistantOutputIndex ??= nextOutputIndex++;
      emit('response.output_item.added', {
        output_index: assistantOutputIndex,
        item: {
          id: assistantItemId,
          type: 'message',
          role: 'assistant',
          status: 'in_progress',
          content: [],
        },
      });
      emit('response.content_part.added', {
        item_id: assistantItemId,
        output_index: assistantOutputIndex,
        content_index: assistantContentIndex,
        part: { type: 'output_text', text: '', annotations: [] },
      });
      continue;
    }

    if (event.type === 'text_delta' && assistantItemId != null && assistantOutputIndex != null) {
      assistantText += event.delta;
      emit('response.output_text.delta', {
        item_id: assistantItemId,
        output_index: assistantOutputIndex,
        content_index: assistantContentIndex,
        delta: event.delta,
      });
      continue;
    }

    if (event.type === 'text_end' && assistantItemId != null && assistantOutputIndex != null) {
      emit('response.output_text.done', {
        item_id: assistantItemId,
        output_index: assistantOutputIndex,
        content_index: assistantContentIndex,
        text: event.content,
      });
      emit('response.content_part.done', {
        item_id: assistantItemId,
        output_index: assistantOutputIndex,
        content_index: assistantContentIndex,
        part: { type: 'output_text', text: event.content, annotations: [] },
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
      emit('response.output_item.added', {
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
      emit('response.function_call_arguments.delta', {
        item_id: itemId,
        output_index: outputIndex,
        delta: event.delta,
      });
      continue;
    }

    if (event.type === 'toolcall_end') {
      const outputIndex = toolOutputIndexes.get(event.contentIndex);
      const itemId = toolItemIds.get(event.contentIndex);
      if (outputIndex == null || itemId == null) continue;
      emit('response.function_call_arguments.done', {
        item_id: itemId,
        output_index: outputIndex,
        arguments: JSON.stringify(event.toolCall.arguments),
      });
      emit('response.output_item.done', {
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
      logModelResponse(logger, model, event.message);
      if (assistantItemId != null && assistantOutputIndex != null) {
        emit('response.output_item.done', {
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
      emit('response.completed', {
        response: createResponsesResponse(model, { ...event.message, responseId }, render),
      });
      reply.raw.end();
      return {};
    }

    if (event.type === 'error') {
      logModelResponse(logger, model, event.error);
      emit('response.failed', {
        response: createResponsesEnvelope(model, render, {
          id: responseId,
          createdAt,
          status: 'failed',
          error: { message: event.error.errorMessage ?? 'Upstream model error' },
        }),
      });
      reply.raw.end();
      return { error: event.error };
    }
  }

  reply.raw.end();
  return {};
}

function logModelResponse(
  logger: FastifyBaseLogger,
  model: Pick<Model<any>, 'provider' | 'id'>,
  message: AssistantMessage,
): void {
  logger.info(
    {
      model: exposedModelId(model),
      responseModel: message.responseModel,
      stopReason: message.stopReason,
      tokenUsage: {
        inputTokens: message.usage.input,
        outputTokens: message.usage.output,
        reasoningTokens: message.usage.reasoning ?? null,
        cacheReadTokens: message.usage.cacheRead,
        cacheWriteTokens: message.usage.cacheWrite,
        cacheWrite1hTokens: message.usage.cacheWrite1h ?? null,
        totalTokens: message.usage.totalTokens,
      },
    },
    'Upstream model response',
  );
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

/**
 * Responses events are numbered so a client can tell whether it missed one, and
 * each payload repeats its own event name.
 */
function sseEmitter(reply: FastifyReply) {
  let sequenceNumber = 0;
  return (event: string, payload: Record<string, unknown>) => {
    writeSseEvent(reply, event, { type: event, ...payload, sequence_number: sequenceNumber++ });
  };
}

function writeSseDone(reply: FastifyReply) {
  reply.raw.write('data: [DONE]\n\n');
  reply.raw.end();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
