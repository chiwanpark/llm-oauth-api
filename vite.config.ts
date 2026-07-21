import { resolve } from 'node:path';

import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [svelte({ configFile: resolve(import.meta.dirname, 'svelte.config.js') })],
  build: {
    outDir: resolve(import.meta.dirname, 'dist/client'),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/v1': 'http://localhost:3000',
    },
  },
});
