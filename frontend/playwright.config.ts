import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  globalSetup: "./tests/global-setup.ts",
  workers: 1,
  timeout: 120_000,
  expect: {
    timeout: 60_000,
  },
  use: {
    baseURL: "http://localhost:3000",
    storageState: "/tmp/yuz-tanima-admin-state.json",
    browserName: "chromium",
    permissions: ["camera"],
    launchOptions: {
      executablePath: "/usr/bin/chromium-browser",
      args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
    },
    screenshot: "only-on-failure",
  },
  reporter: "line",
});
