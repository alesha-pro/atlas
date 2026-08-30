import { defineConfig } from 'vite';

// Относительный base: одна и та же сборка работает и в корне (Cloudflare Pages),
// и в подпапке (GitHub Pages отдаёт проект по /atlas/).
export default defineConfig({
  base: './',
  build: { assetsInlineLimit: 0 },
});
