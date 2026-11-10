import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  // Keep ONNX Runtime's JS and WASM sidecars together in both dev and production.
  optimizeDeps: { exclude: ["onnxruntime-web/wasm"] },
  worker: { format: "es" },
});
