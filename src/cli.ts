#!/usr/bin/env node
import { Command, Option } from 'commander';

import { createCliAuthCallbacks, confirmOverwrite } from './auth-ui.js';
import { JsonCredentialStore } from './credential-store.js';
import {
  DEFAULT_OAUTH_REFRESH_BEFORE_EXPIRY_MS,
  DEFAULT_OAUTH_REFRESH_INTERVAL_MS,
  parseRefreshSeconds,
} from './oauth-refresh.js';
import { createSupportedProvider, getSupportedProviderIds } from './providers.js';
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
  .action(async (options) => {
    const apiKey = process.env.LLM_OAUTH_API_KEY;
    if (!apiKey) {
      throw new Error('LLM_OAUTH_API_KEY environment variable is required');
    }

    await startServer({
      authFile: options.authFile,
      providerIds: splitCsv(options.providers),
      apiKey,
      port: Number(options.port),
      host: options.host,
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
  .action(async (providerName, options) => {
    const provider = createSupportedProvider(providerName);
    const store = new JsonCredentialStore(options.authFile);
    const existing = await store.read(provider.id);
    if (existing && !options.force) {
      const overwrite = await confirmOverwrite(provider.id);
      if (!overwrite) return;
    }

    let credential;
    if (provider.auth.oauth) {
      credential = {
        ...(await provider.auth.oauth.login(createCliAuthCallbacks())),
        type: 'oauth' as const,
      };
    } else if (provider.auth.apiKey?.login) {
      credential = await provider.auth.apiKey.login(createCliAuthCallbacks());
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
