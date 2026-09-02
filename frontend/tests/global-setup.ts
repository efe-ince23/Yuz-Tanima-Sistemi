import { request, type FullConfig } from "@playwright/test";

export default async function globalSetup(config: FullConfig) {
  const baseURL = config.projects[0]?.use.baseURL ?? "http://localhost:3000";
  const context = await request.newContext({ baseURL });
  const credentials = {
    identifier: process.env.E2E_ADMIN_USERNAME ?? "admin",
    password: process.env.E2E_ADMIN_PASSWORD,
  };
  const response = await context.post("/api/auth/login", { data: credentials });
  if (!response.ok()) {
    throw new Error(`E2E admin login failed: ${response.status()} ${await response.text()}`);
  }
  const backendResponse = await context.post("http://backend:8000/api/auth/login", {
    data: credentials,
  });
  if (!backendResponse.ok()) {
    throw new Error(`E2E backend login failed: ${backendResponse.status()} ${await backendResponse.text()}`);
  }
  await context.storageState({ path: "/tmp/yuz-tanima-admin-state.json" });
  await context.dispose();
}
