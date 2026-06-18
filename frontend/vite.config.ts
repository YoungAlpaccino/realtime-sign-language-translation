import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// onnxruntime-web ships .wasm/.mjs assets that must not be pre-bundled.
export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: {
    port: 5173,
    // The browser demo can either run fully local or proxy the backend WS.
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
