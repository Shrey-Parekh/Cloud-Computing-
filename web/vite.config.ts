import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, /api goes to the FastAPI server (uvicorn api.main:app --port 8000).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
