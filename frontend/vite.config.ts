import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendTarget = process.env.BACKEND_URL ?? "http://backend:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 3000,
    proxy: {
      "/api": backendTarget,
      "/faces": backendTarget,
      "/processes": backendTarget,
      "/media": backendTarget,
      "/health": backendTarget,
      "/openapi.json": backendTarget,
    },
  },
});
