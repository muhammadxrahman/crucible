import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build to web/dist, which the gateway serves at /. In dev, proxy the API to the
// running gateway so the SPA and API share one origin (no CORS), matching production.
const api = { target: "http://127.0.0.1:8000", changeOrigin: true };

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      "/v1": api,
      "/rag": api,
      "/admin": api,
      "/metrics": api,
      "/healthz": api,
    },
  },
});
