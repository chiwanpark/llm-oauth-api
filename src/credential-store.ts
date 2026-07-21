import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';

import type { CredentialStore, Credential, CredentialInfo } from '@earendil-works/pi-ai';

async function readJsonFile(filePath: string): Promise<Record<string, Credential>> {
  try {
    const raw = await readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw) as Record<string, Credential>;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return {};
    }
    throw error;
  }
}

async function writeJsonFileAtomic(
  filePath: string,
  data: Record<string, Credential>,
): Promise<void> {
  const dir = path.dirname(filePath);
  await mkdir(dir, { recursive: true });
  const tempPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tempPath, JSON.stringify(data, null, 2) + '\n', { mode: 0o600 });
  await rename(tempPath, filePath);
}

export class JsonCredentialStore implements CredentialStore {
  private readonly filePath: string;
  private queue: Promise<void> = Promise.resolve();

  constructor(filePath: string) {
    this.filePath = path.resolve(filePath);
  }

  async read(providerId: string): Promise<Credential | undefined> {
    const data = await readJsonFile(this.filePath);
    return data[providerId];
  }

  async list(): Promise<readonly CredentialInfo[]> {
    const data = await readJsonFile(this.filePath);
    return Object.entries(data).map(([providerId, credential]) => ({
      providerId,
      type: credential.type,
    }));
  }

  async modify(
    providerId: string,
    fn: (current: Credential | undefined) => Promise<Credential | undefined>,
  ): Promise<Credential | undefined> {
    return this.serialized(async () => {
      const data = await readJsonFile(this.filePath);
      const next = await fn(data[providerId]);
      if (next !== undefined) {
        data[providerId] = next;
      }
      await writeJsonFileAtomic(this.filePath, data);
      return data[providerId];
    });
  }

  async delete(providerId: string): Promise<void> {
    await this.serialized(async () => {
      const data = await readJsonFile(this.filePath);
      delete data[providerId];
      if (Object.keys(data).length === 0) {
        await rm(this.filePath, { force: true });
        return;
      }
      await writeJsonFileAtomic(this.filePath, data);
    });
  }

  private async serialized<T>(task: () => Promise<T>): Promise<T> {
    const previous = this.queue;
    let release!: () => void;
    this.queue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await task();
    } finally {
      release();
    }
  }
}
