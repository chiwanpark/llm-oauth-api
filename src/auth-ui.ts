import { confirm, input, password, select } from '@inquirer/prompts';

import type { AuthEvent, AuthPrompt, ProviderAuthInteraction } from '@earendil-works/pi-ai';

async function promptValue(prompt: AuthPrompt): Promise<string> {
  switch (prompt.type) {
    case 'text':
    case 'manual_code':
      return input({
        message: prompt.message,
        ...(prompt.placeholder ? { default: prompt.placeholder } : {}),
      });
    case 'secret':
      return password({ message: prompt.message });
    case 'select': {
      const value = await select({
        message: prompt.message,
        choices: prompt.options.map((option) => ({
          name: option.label + (option.description ? ` — ${option.description}` : ''),
          value: option.id,
        })),
      });
      return value;
    }
  }
}

/**
 * Provider `login()` implementations require a concrete abort signal (pi-ai
 * >=0.84). Interactive CLI logins have no in-process cancellation source --
 * the user cancels with Ctrl+C, which tears down the whole process -- so this
 * signal is intentionally never aborted.
 */
function neverAbortedSignal(): AbortSignal {
  return new AbortController().signal;
}

export function createCliAuthCallbacks(): ProviderAuthInteraction {
  return {
    signal: neverAbortedSignal(),
    async prompt(prompt) {
      return promptValue(prompt);
    },
    notify(event) {
      printAuthEvent(event);
    },
  };
}

function printAuthEvent(event: AuthEvent): void {
  switch (event.type) {
    case 'info':
      console.log(event.message);
      for (const link of event.links ?? []) {
        console.log(link.label ? `${link.label}: ${link.url}` : link.url);
      }
      break;
    case 'auth_url':
      console.log(`\nOpen this URL in your browser:\n${event.url}`);
      if (event.instructions) console.log(`\n${event.instructions}`);
      break;
    case 'device_code':
      console.log(`\nOpen ${event.verificationUri}`);
      console.log(`Enter code: ${event.userCode}`);
      break;
    case 'progress':
      console.log(event.message);
      break;
  }
}

export async function confirmOverwrite(providerId: string): Promise<boolean> {
  return confirm({ message: `Overwrite stored credentials for ${providerId}?`, default: true });
}
