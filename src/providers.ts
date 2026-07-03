import type { Provider } from '@earendil-works/pi-ai';
import { anthropicProvider } from '@earendil-works/pi-ai/providers/anthropic';
import { githubCopilotProvider } from '@earendil-works/pi-ai/providers/github-copilot';
import { googleProvider } from '@earendil-works/pi-ai/providers/google';
import { nvidiaProvider } from '@earendil-works/pi-ai/providers/nvidia';
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex';
import { opencodeGoProvider } from '@earendil-works/pi-ai/providers/opencode-go';

export type SupportedProviderId =
  'anthropic' | 'github-copilot' | 'google' | 'nvidia' | 'openai-codex' | 'opencode-go';

const providerFactories: Record<SupportedProviderId, () => Provider> = {
  anthropic: anthropicProvider,
  'github-copilot': githubCopilotProvider,
  google: googleProvider,
  nvidia: nvidiaProvider,
  'openai-codex': openaiCodexProvider,
  'opencode-go': opencodeGoProvider,
};

const aliases: Record<string, SupportedProviderId> = {
  anthropic: 'anthropic',
  claude: 'anthropic',
  'claude-code': 'anthropic',
  copilot: 'github-copilot',
  github: 'github-copilot',
  'github-copilot': 'github-copilot',
  gemini: 'google',
  google: 'google',
  'google-ai': 'google',
  nim: 'nvidia',
  nvidia: 'nvidia',
  'nvidia-nim': 'nvidia',
  codex: 'openai-codex',
  'openai-codex': 'openai-codex',
  opencode: 'opencode-go',
  'opencode-go': 'opencode-go',
};

export function getSupportedProviderIds(): SupportedProviderId[] {
  return Object.keys(providerFactories) as SupportedProviderId[];
}

export function resolveProviderId(input: string): SupportedProviderId | undefined {
  return aliases[input.trim().toLowerCase()];
}

export function createSupportedProvider(input: string): Provider {
  const providerId = resolveProviderId(input);
  if (!providerId) {
    throw new Error(`Unsupported provider: ${input}`);
  }
  return providerFactories[providerId]();
}

export function createSupportedProviders(providerIds?: string[]): Provider[] {
  const ids = providerIds?.length
    ? providerIds.map((id) => {
        const resolved = resolveProviderId(id);
        if (!resolved) throw new Error(`Unsupported provider: ${id}`);
        return resolved;
      })
    : getSupportedProviderIds();

  return Array.from(new Set(ids)).map((id) => providerFactories[id]());
}
