import { expect, test } from "@playwright/test";


test("recognizes a registered person through the interface", async ({ page, request }) => {
  const personResponse = await request.get("/api/persons/4");
  expect(personResponse.ok()).toBeTruthy();
  const person = (await personResponse.json()) as { face_id: string };
  const identityBeforeResponse = await request.get(`/api/identities/${person.face_id}`);
  const identityBefore = (await identityBeforeResponse.json()) as {
    observation_count: number;
    last_seen_at: string | null;
  };

  const faceResponse = await request.get("/api/persons/4/face-images");
  expect(faceResponse.ok()).toBeTruthy();
  const faces = (await faceResponse.json()) as Array<{ image_url: string }>;
  expect(faces.length).toBeGreaterThan(0);

  const imageResponse = await request.get(faces[0].image_url);
  expect(imageResponse.ok()).toBeTruthy();
  const image = await imageResponse.body();

  await page.goto("/");
  await page.getByRole("button", { name: "Kayıtlı kişiler" }).click();
  await expect(page.getByText("Fatih Terim")).toBeVisible();
  await expect(page.getByText("Burak Yılmaz")).toBeVisible();
  await page.getByRole("button", { name: "Tanıma", exact: true }).click();

  await page.locator('input[type="file"]').setInputFiles({
    name: "registered-face.jpg",
    mimeType: "image/jpeg",
    buffer: image,
  });
  await page.getByRole("button", { name: "Kişiyi tanı" }).click();

  await expect(page.getByText("Kişi tanındı")).toBeVisible();
  await expect(page.locator(".recognized-result").getByText("Fatih Terim")).toBeVisible();
  const confidenceText = await page.locator(".confidence-block").textContent();
  const confidenceMatch = confidenceText?.match(/%(\d+(?:\.\d+)?)/);
  expect(confidenceMatch).toBeTruthy();
  expect(Number(confidenceMatch![1])).toBeGreaterThanOrEqual(99);
  const processId = await page.locator(".process-reference strong").textContent();
  expect(processId).toBeTruthy();
  const processResponse = await request.get(`/api/processes/${processId}`);
  const processRecord = await processResponse.json() as { operation_type: string; events: unknown[] };
  expect(processRecord.operation_type).toBe("identify");
  expect(processRecord.events).toHaveLength(1);

  const identityAfterResponse = await request.get(`/api/identities/${person.face_id}`);
  const identityAfter = (await identityAfterResponse.json()) as {
    observation_count: number;
    last_seen_at: string | null;
  };
  expect(identityAfter.observation_count).toBe(identityBefore.observation_count + 1);
  expect(identityAfter.last_seen_at).toBeTruthy();
  if (identityBefore.last_seen_at) {
    expect(Date.parse(identityAfter.last_seen_at!)).toBeGreaterThan(
      Date.parse(identityBefore.last_seen_at),
    );
  }
});


test("recognizes every registered face in one photo through the interface", async ({ page, request }) => {
  const referenceImages: Buffer[] = [];
  for (const personId of [4, 5]) {
    const listResponse = await request.get(`/api/persons/${personId}/face-images`);
    expect(listResponse.ok()).toBeTruthy();
    const images = (await listResponse.json()) as Array<{ image_url: string }>;
    expect(images.length).toBeGreaterThan(0);

    const imageResponse = await request.get(images[0].image_url);
    expect(imageResponse.ok()).toBeTruthy();
    referenceImages.push(await imageResponse.body());
  }

  const dataUrls = referenceImages.map(
    (image) => `data:image/jpeg;base64,${image.toString("base64")}`,
  );
  await page.setViewportSize({ width: 1000, height: 650 });
  await page.setContent(`
    <style>
      html, body { margin: 0; width: 100%; height: 100%; background: white; overflow: hidden; }
      main { display: grid; grid-template-columns: 1fr 1fr; width: 1000px; height: 650px; }
      img { width: 500px; height: 650px; object-fit: contain; }
    </style>
    <main><img src="${dataUrls[0]}" /><img src="${dataUrls[1]}" /></main>
  `);
  await Promise.all([
    page.locator("img").nth(0).evaluate((image: HTMLImageElement) => image.decode()),
    page.locator("img").nth(1).evaluate((image: HTMLImageElement) => image.decode()),
  ]);
  const multiFaceImage = await page.screenshot({ type: "png" });

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "fatih-ve-burak.png",
    mimeType: "image/png",
    buffer: multiFaceImage,
  });
  await page.locator(".primary-button").click();

  const results = page.locator(".face-result-list");
  await expect(results).toBeVisible();
  await expect(results).toContainText("Fatih Terim");
  await expect(results).toContainText("Burak");
  await expect(results.locator(".face-result-row.recognized")).toHaveCount(2);
  await expect(results.locator(".face-result-row.unrecognized")).toHaveCount(0);
  const processId = await page.locator(".process-reference strong").textContent();
  expect(processId).toBeTruthy();
  const processResponse = await request.get(`/api/processes/${processId}`);
  const processRecord = await processResponse.json() as {
    face_count: number;
    task_detail: { faces: Array<{ face_id: string; status: string }> };
    events: unknown[];
  };
  expect(processRecord.face_count).toBe(2);
  expect(processRecord.events).toHaveLength(2);
  expect(processRecord.task_detail.faces).toHaveLength(2);
  expect(processRecord.task_detail.faces.map((face) => face.status)).toEqual(["known", "known"]);
});


test("enrolls a new anonymous face without changing its persistent ID", async ({ page }) => {
  const faceId = "11111111-2222-4333-8444-555555555555";
  let enrollPayload: Record<string, unknown> | null = null;
  await page.route("**/api/faces/identify", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        process_id: "99999999-2222-4333-8444-555555555555",
        status: "unrecognized",
        recognized: false,
        similarity: null,
        threshold: 0.45,
        person: null,
        face_id: faceId,
        matched_image_url: null,
        execution_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        detected_face_count: 1,
        ignored_face_count: 0,
        faces: [{
          face_index: 0,
          face_id: faceId,
          status: "new_anonymous",
          recognized: false,
          similarity: null,
          person: null,
          matched_image_url: null,
          detection_confidence: 0.98,
          bounding_box: { x1: 10, y1: 10, x2: 100, y2: 100, width: 90, height: 90 },
        }],
      }),
    });
  });
  await page.route(`**/api/anonymous-identities/${faceId}/enroll`, async (route) => {
    enrollPayload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({
        id: 101,
        face_id: faceId,
        first_name: "AnonimTest",
        last_name: "Kisi",
        description: "Sonradan isimlendirildi",
        face_image_count: 0,
        created_at: "2026-08-19T06:30:00Z",
        updated_at: "2026-08-19T06:30:00Z",
      }),
    });
  });

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "anonymous.png",
    mimeType: "image/png",
    buffer: Buffer.from("89504e470d0a1a0a", "hex"),
  });
  await page.locator(".primary-button").click();

  await expect(page.getByRole("heading", { name: "Yeni anonim yüz" })).toBeVisible();
  await expect(page.locator(".anonymous-face-id")).toContainText(faceId);
  await page.getByRole("button", { name: "Kimlik bilgisi ekle" }).click();

  const dialog = page.getByRole("dialog", { name: "Anonim yüzü isimlendir" });
  await dialog.getByLabel("Ad", { exact: true }).fill("AnonimTest");
  await dialog.getByLabel("Soyad", { exact: true }).fill("Kisi");
  await dialog.getByLabel("Açıklama", { exact: true }).fill("Sonradan isimlendirildi");
  await dialog.getByRole("button", { name: "Kimliği kaydet" }).click();

  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".recognized-result")).toContainText("AnonimTest Kisi");
  expect(enrollPayload).toEqual({
    first_name: "AnonimTest",
    last_name: "Kisi",
    description: "Sonradan isimlendirildi",
  });
});


test("fits the recognition workflow on a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Yüz Tanıma", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Kişiyi tanı" })).toBeVisible();
  await page.getByTitle("Kayıtlı kişiler").click();
  await expect(page.getByRole("heading", { name: "Kayıtlı Kişiler" })).toBeVisible();

  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBeFalsy();
});


test("captures a test photo from the camera", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Kamerayı aç" }).click();

  const dialog = page.getByRole("dialog", { name: "Kameradan fotoğraf çek" });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator("video")).toBeVisible();

  const captureButton = dialog.getByRole("button", { name: "Çek ve kullan" });
  await expect(captureButton).toBeEnabled();
  await captureButton.click();

  await expect(dialog).toHaveCount(0);
  await expect(page.getByAltText("Seçilen test fotoğrafı")).toBeVisible();
  await expect(page.getByText("Fotoğraf hazır")).toBeVisible();
  await expect(page.getByRole("button", { name: "Kişiyi tanı" })).toBeEnabled();
});
