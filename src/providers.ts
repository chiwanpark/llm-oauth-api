import type { Provider } from '@earendil-works/pi-ai';
import { anthropicProvider } from '@earendil-works/pi-ai/providers/anthropic';
import { cerebrasProvider } from '@earendil-works/pi-ai/providers/cerebras';
import { githubCopilotProvider } from '@earendil-works/pi-ai/providers/github-copilot';
import { googleProvider } from '@earendil-works/pi-ai/providers/google';
import { nvidiaProvider } from '@earendil-works/pi-ai/providers/nvidia';
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex';
import { opencodeGoProvider } from '@earendil-works/pi-ai/providers/opencode-go';

export type SupportedProviderId =
  | 'anthropic'
  | 'cerebras'
  | 'github-copilot'
  | 'google'
  | 'nvidia'
  | 'openai-codex'
  | 'opencode-go';

const providerFactories: Record<SupportedProviderId, () => Provider> = {
  anthropic: anthropicProvider,
  cerebras: cerebrasProvider,
  'github-copilot': githubCopilotProvider,
  google: googleProvider,
  nvidia: nvidiaProvider,
  'openai-codex': openaiCodexProvider,
  'opencode-go': opencodeGoProvider,
};

export function getSupportedProviderIds(): SupportedProviderId[] {
  return Object.keys(providerFactories) as SupportedProviderId[];
}

export function resolveProviderId(input: string): SupportedProviderId | undefined {
  const normalized = input.trim().toLowerCase();
  return Object.hasOwn(providerFactories, normalized)
    ? (normalized as SupportedProviderId)
    : undefined;
}

export function createSupportedProvider(input: string): Provider {
  const providerId = resolveProviderId(input);
  if (!providerId) {
    throw new Error(`Unsupported provider: ${input}`);
  }
  return providerFactories[providerId]();
}

/** Normalizes requested provider names into the deduped set the server exposes. */
export function resolveSupportedProviderIds(providerIds?: string[]): SupportedProviderId[] {
  const ids = providerIds?.length
    ? providerIds.map((id) => {
        const resolved = resolveProviderId(id);
        if (!resolved) throw new Error(`Unsupported provider: ${id}`);
        return resolved;
      })
    : getSupportedProviderIds();

  return Array.from(new Set(ids));
}

export function createSupportedProviders(providerIds?: string[]): Provider[] {
  return resolveSupportedProviderIds(providerIds).map((id) => providerFactories[id]());
}
