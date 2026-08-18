#!/usr/bin/env node
import { Command, Option } from 'commander';

import { createCliAuthCallbacks, confirmOverwrite } from './auth-ui.js';
import { JsonCredentialStore } from './credential-store.js';
import {
  DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS,
  DEFAULT_OAUTH_REFRESH_INTERVAL_MS,
  parseRefreshSeconds,
} from './oauth-refresh.js';
import { loadModelGroups } from './groups.js';
import { DEFAULT_MODEL_COOLDOWN_MS, parseCooldownSeconds } from './model-cooldown.js';
import {
  createSupportedProvider,
  getSupportedProviderIds,
  resolveSupportedProviderIds,
} from './providers.js';
import { startServer } from './server.js';

const program = new Command();

program
  .name('llm-oauth-api')
  .description('OpenAI-compatible HTTP API backed by pi-ai OAuth/API-key providers');

program
  .command('serve')
  .requiredOption('--auth-file <path>', 'Path to auth JSON file')
  .option(
    '--providers <providers>',
    'Comma-separated providers to expose',
    getSupportedProviderIds().join(','),
  )
  .option('--groups-file <path>', 'Path to JSON file defining model groups')
  .option('--port <port>', 'Port to listen on', '3000')
  .option('--host <host>', 'Host to listen on', '0.0.0.0')
  .addOption(
    new Option('--oauth-refresh-interval <seconds>', 'How often to check OAuth credential expiry')
      .argParser((value) => parseRefreshSeconds(value, '--oauth-refresh-interval', false))
      .default(DEFAULT_OAUTH_REFRESH_INTERVAL_MS, String(DEFAULT_OAUTH_REFRESH_INTERVAL_MS / 1000)),
  )
  .addOption(
    new Option(
      '--oauth-refresh-before-expiry <seconds>',
      'Refresh OAuth credentials this long before expiry (0 means expired only)',
    )
      .argParser((value) => parseRefreshSeconds(value, '--oauth-refresh-before-expiry', true))
      .default(
        DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS,
        String(DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS / 1000),
      ),
  )
  .option('--no-oauth-auto-refresh', 'Disable automatic OAuth credential refresh')
  .addOption(
    new Option(
      '--model-cooldown <seconds>',
      'How long a group skips a model that failed to respond (0 disables)',
    )
      .argParser((value) => parseCooldownSeconds(value, '--model-cooldown'))
      .default(DEFAULT_MODEL_COOLDOWN_MS, String(DEFAULT_MODEL_COOLDOWN_MS / 1000)),
  )
  .addHelpText(
    'after',
    '\nModel groups:\n' +
      '  Pass --groups-file <path> to a JSON object mapping a group name to its members.\n' +
      '  e.g. {"free": ["github-copilot:gpt-5.4-mini", "openai-codex:gpt-5-mini"]}\n' +
      '  makes the model "free" try github-copilot first and fall back to openai-codex.\n' +
      '  An entry without a ":" names another group and is flattened into its parent.\n' +
      '  A member that fails is skipped for --model-cooldown seconds afterwards.',
  )
  .action(async (options) => {
    const apiKey = process.env.LLM_OAUTH_API_KEY;
    if (!apiKey) {
      throw new Error('LLM_OAUTH_API_KEY environment variable is required');
    }

    const providerIds = splitCsv(options.providers);
    const groups = options.groupsFile
      ? await loadModelGroups(options.groupsFile, resolveSupportedProviderIds(providerIds))
      : [];

    await startServer({
      authFile: options.authFile,
      providerIds,
      groups,
      apiKey,
      port: Number(options.port),
      host: options.host,
      modelCooldownMs: options.modelCooldown,
      oauthAutoRefresh: options.oauthAutoRefresh,
      oauthRefreshIntervalMs: options.oauthRefreshInterval,
      oauthRefreshBeforeExpiryMs: options.oauthRefreshBeforeExpiry,
    });
  });

program
  .command('login')
  .argument('<provider>', `Provider to authenticate (${getSupportedProviderIds().join(', ')})`)
  .requiredOption('--auth-file <path>', 'Path to auth JSON file')
  .option('--force', 'Overwrite existing credential without confirmation', false)
  .option('--api-key', 'Store an API key instead of running the OAuth flow', false)
  .action(async (providerName, options) => {
    const provider = createSupportedProvider(providerName);
    const store = new JsonCredentialStore(options.authFile);
    const existing = await store.read(provider.id);
    if (existing && !options.force) {
      const overwrite = await confirmOverwrite(provider.id);
      if (!overwrite) return;
    }

    // A few providers offer both flows; --api-key picks the key prompt so a
    // headless session never has to complete a browser round trip.
    const useApiKey = options.apiKey || !provider.auth.oauth;

    let credential;
    if (useApiKey && provider.auth.apiKey?.login) {
      credential = await provider.auth.apiKey.login(createCliAuthCallbacks());
    } else if (options.apiKey) {
      throw new Error(`Provider ${provider.id} does not support API-key login`);
    } else if (provider.auth.oauth) {
      credential = {
        ...(await provider.auth.oauth.login(createCliAuthCallbacks())),
        type: 'oauth' as const,
      };
    } else {
      throw new Error(`Provider ${provider.id} does not support interactive login`);
    }

    await store.modify(provider.id, async () => credential);
    console.log(`Saved credentials for ${provider.id} to ${options.authFile}`);
  });

program
  .command('logout')
  .argument('<provider>', `Provider credential to remove (${getSupportedProviderIds().join(', ')})`)
  .requiredOption('--auth-file <path>', 'Path to auth JSON file')
  .action(async (providerName, options) => {
    const provider = createSupportedProvider(providerName);
    const store = new JsonCredentialStore(options.authFile);
    await store.delete(provider.id);
    console.log(`Removed credentials for ${provider.id} from ${options.authFile}`);
  });

program
  .command('providers')
  .description('List supported providers')
  .action(() => {
    for (const providerId of getSupportedProviderIds()) {
      console.log(providerId);
    }
  });

program.parseAsync(process.argv).catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

function splitCsv(value: string): string[] {
  return value
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);
}
