import path from "node:path"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

// During dev, the React app runs on :5173 and the FastAPI backend on :8000.
// Forward all /api/* requests to the backend so the browser stays on a
// single origin (no CORS preflight on every fetch). The proxy disables
// any intermediate buffering so SSE streams flush as they arrive.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: false,
        configure(proxy) {
          proxy.on("proxyRes", (proxyRes) => {
            proxyRes.headers["x-accel-buffering"] = "no"
          })
        },
      },
    },
  },
})
