import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const currentDir = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  root: path.resolve(currentDir, 'app'),
  plugins: [react()],
  build: {
    outDir: path.resolve(currentDir, 'dist'),
    emptyOutDir: true,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8337',
      '/agent': 'http://127.0.0.1:8337',
      '/audio': 'http://127.0.0.1:8337',
      '/summary': 'http://127.0.0.1:8337',
      '/notes': 'http://127.0.0.1:8337',
      '/health': 'http://127.0.0.1:8337',
    },
  },
})
