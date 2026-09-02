import { expect, test } from "@playwright/test";


test("returns a normal no-face result for a valid image without a face", async ({ page, request }) => {
  await page.setViewportSize({ width: 160, height: 160 });
  await page.setContent("<style>html,body{margin:0;width:100%;height:100%;background:#fff}</style>");
  const image = await page.screenshot({ type: "png" });
  const statisticsBefore = await (await request.get("/api/statistics")).json();

  const response = await request.post("/api/faces/identify", {
    multipart: {
      file: { name: "empty-scene.png", mimeType: "image/png", buffer: image },
    },
  });

  expect(response.status()).toBe(200);
  const body = await response.json();
  expect(body.status).toBe("no_face");
  expect(body.recognized).toBeFalsy();
  expect(body.detected_face_count).toBe(0);
  expect(body.person).toBeNull();

  const statisticsAfter = await (await request.get("/api/statistics")).json();
  expect(statisticsAfter.total_operations).toBe(statisticsBefore.total_operations);

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "empty-scene.png",
    mimeType: "image/png",
    buffer: image,
  });
  await page.getByRole("button", { name: "Kişiyi tanı" }).click();
  await expect(page.getByRole("heading", { name: "Fotoğrafta yüz bulunamadı" })).toBeVisible();
});


test("rejects empty, corrupt, unsupported and mismatched image files", async ({ request }) => {
  const emptyResponse = await request.post("/api/faces/identify", {
    multipart: {
      file: { name: "empty.jpg", mimeType: "image/jpeg", buffer: Buffer.alloc(0) },
    },
  });
  expect(emptyResponse.status()).toBe(400);
  await expect(emptyResponse.json()).resolves.toMatchObject({
    error: { code: "EMPTY_FILE", message: expect.stringContaining("dosya bos") },
    process_id: expect.any(String),
    timestamp: expect.any(String),
  });

  const corruptResponse = await request.post("/api/faces/identify", {
    multipart: {
      file: { name: "corrupt.jpg", mimeType: "image/jpeg", buffer: Buffer.from("not-a-jpeg") },
    },
  });
  expect(corruptResponse.status()).toBe(400);
  await expect(corruptResponse.json()).resolves.toMatchObject({
    error: { code: "CORRUPT_IMAGE", message: expect.stringContaining("bozuk veya gecersiz") },
    process_id: expect.any(String),
    timestamp: expect.any(String),
  });

  const unsupportedResponse = await request.post("/api/faces/identify", {
    multipart: {
      file: { name: "note.txt", mimeType: "text/plain", buffer: Buffer.from("plain text") },
    },
  });
  expect(unsupportedResponse.status()).toBe(415);
  await expect(unsupportedResponse.json()).resolves.toMatchObject({
    error: { code: "UNSUPPORTED_IMAGE_TYPE", message: expect.stringContaining("Desteklenmeyen") },
    process_id: expect.any(String),
    timestamp: expect.any(String),
  });

  const pngHeader = Buffer.from("89504e470d0a1a0a00000000", "hex");
  const mismatchedResponse = await request.post("/api/faces/identify", {
    multipart: {
      file: { name: "wrong.jpg", mimeType: "image/jpeg", buffer: pngHeader },
    },
  });
  expect(mismatchedResponse.status()).toBe(415);
  await expect(mismatchedResponse.json()).resolves.toMatchObject({
    error: { code: "IMAGE_CONTENT_TYPE_MISMATCH", message: expect.stringContaining("uyusmuyor") },
    process_id: expect.any(String),
    timestamp: expect.any(String),
  });
});
