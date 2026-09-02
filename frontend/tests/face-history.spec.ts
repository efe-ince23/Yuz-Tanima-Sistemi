import { expect, test } from "@playwright/test";


test("queries a face ID's process history with timestamps and pagination", async ({ request }) => {
  const personResponse = await request.get("/api/persons/24");
  expect(personResponse.ok()).toBeTruthy();
  const person = await personResponse.json() as { face_id: string };

  const historyResponse = await request.get(
    `/api/identities/${person.face_id}/history?limit=2&offset=0`,
  );
  expect(historyResponse.ok()).toBeTruthy();
  const history = await historyResponse.json() as {
    face_id: string;
    total: number;
    limit: number;
    offset: number;
    first_seen_at: string;
    last_seen_at: string;
    appearances: Array<{
      process_id: string | null;
      timestamp: string;
      status: string;
    }>;
  };

  expect(history.face_id).toBe(person.face_id);
  expect(history.total).toBeGreaterThan(1);
  expect(history.limit).toBe(2);
  expect(history.offset).toBe(0);
  expect(history.appearances).toHaveLength(2);
  expect(history.appearances.every((item) => item.process_id)).toBeTruthy();
  expect(Date.parse(history.last_seen_at)).toBeGreaterThanOrEqual(
    Date.parse(history.first_seen_at),
  );
  expect(Date.parse(history.appearances[0].timestamp)).toBeGreaterThanOrEqual(
    Date.parse(history.appearances[1].timestamp),
  );

  const appearanceWithProcess = history.appearances.find((item) => item.process_id);
  expect(appearanceWithProcess?.process_id).toBeTruthy();
  const processResponse = await request.get(
    `/api/processes/${appearanceWithProcess!.process_id}`,
  );
  expect(processResponse.ok()).toBeTruthy();
  const processRecord = await processResponse.json() as {
    process_id: string;
    events: Array<{ face_id: string | null }>;
  };
  expect(processRecord.process_id).toBe(appearanceWithProcess!.process_id);
  expect(processRecord.events.some((event) => event.face_id === person.face_id)).toBeTruthy();

  const secondPageResponse = await request.get(
    `/api/identities/${person.face_id}/history?limit=2&offset=2`,
  );
  const secondPage = await secondPageResponse.json() as {
    appearances: Array<{ process_id: string | null; timestamp: string }>;
  };
  expect(secondPage.appearances.length).toBeGreaterThan(0);
  expect(secondPage.appearances[0]).not.toEqual(history.appearances[0]);
});


test("returns 404 for an unknown face ID history", async ({ request }) => {
  const response = await request.get(
    "/api/identities/00000000-0000-4000-8000-000000000000/history",
  );
  expect(response.status()).toBe(404);
});


test("opens a process detail from the face history in the interface", async ({ page }) => {
  const faceId = "11111111-1111-4111-8111-111111111111";
  const processId = "22222222-2222-4222-8222-222222222222";
  const timestamp = "2026-08-20T08:30:00Z";

  await page.route("**/api/identities", async (route) => {
    await route.fulfill({
      json: [{
        face_id: faceId,
        status: "known",
        person_id: 901,
        first_name: "Geçmiş",
        last_name: "Testi",
        description: "İşlem geçmişi ekran testi",
        sample_count: 1,
        reference_image_count: 1,
        observation_count: 1,
        sample_image_urls: [],
        created_at: timestamp,
        updated_at: timestamp,
        last_seen_at: timestamp,
      }],
    });
  });
  await page.route(`**/api/identities/${faceId}/history?*`, async (route) => {
    await route.fulfill({
      json: {
        face_id: faceId,
        total: 1,
        limit: 8,
        offset: 0,
        first_seen_at: timestamp,
        last_seen_at: timestamp,
        appearances: [{
          event_id: 77,
          process_id: processId,
          operation_type: "identify",
          timestamp,
          status: "known",
          recognized: true,
          similarity: 0.91,
          threshold: 0.45,
        }],
      },
    });
  });
  await page.route(`**/api/processes/${processId}`, async (route) => {
    await route.fulfill({
      json: {
        process_id: processId,
        operation_type: "identify",
        status: "recognized",
        http_status: 200,
        face_count: 1,
        task_detail: null,
        result: null,
        error_detail: null,
        created_at: timestamp,
        completed_at: "2026-08-20T08:30:01Z",
        events: [{
          id: 77,
          face_id: faceId,
          face_status: "known",
          recognized: true,
          person_id: 901,
          similarity: 0.91,
          threshold: 0.45,
          created_at: timestamp,
        }],
      },
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Kimlikler" }).first().click();
  await page.locator(`[data-face-id="${faceId}"]`).click();
  const history = page.getByRole("region", { name: "Görülme geçmişi" });
  await expect(history.getByText("1 kayıt")).toBeVisible();
  await expect(history.getByText("known", { exact: true })).toBeVisible();
  await expect(history.getByText(processId.slice(0, 8), { exact: true })).toBeVisible();

  await history.getByRole("button", { name: /known/ }).click();
  const dialog = page.getByRole("dialog", { name: "Process detayı" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(processId, { exact: true })).toBeVisible();
  await expect(dialog.getByText(faceId, { exact: true })).toBeVisible();
  await expect(dialog.getByText("%91", { exact: true })).toBeVisible();
});
