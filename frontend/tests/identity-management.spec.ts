import { expect, test } from "@playwright/test";


interface PersonRecord {
  id: number;
  face_id: string;
  first_name: string;
  last_name: string;
}


const TEST_FIRST_NAME = "KimlikApiTest";


test("queries, updates and deletes a known identity by face ID", async ({ request }) => {
  let personId: number | null = null;

  try {
    const createResponse = await request.post("/api/persons", {
      data: {
        first_name: TEST_FIRST_NAME,
        last_name: "Gecici",
        description: "Face ID yonetim testi",
      },
    });
    expect(createResponse.status()).toBe(201);
    const created = (await createResponse.json()) as PersonRecord;
    personId = created.id;

    const getResponse = await request.get(`/api/identities/${created.face_id}`);
    expect(getResponse.ok()).toBeTruthy();
    await expect(getResponse.json()).resolves.toMatchObject({
      face_id: created.face_id,
      status: "known",
      person_id: created.id,
      first_name: TEST_FIRST_NAME,
    });

    const updateResponse = await request.patch(`/api/identities/${created.face_id}`, {
      data: { last_name: "Guncel", description: "Guncellenmis metadata" },
    });
    expect(updateResponse.ok()).toBeTruthy();
    await expect(updateResponse.json()).resolves.toMatchObject({
      face_id: created.face_id,
      status: "known",
      last_name: "Guncel",
      description: "Guncellenmis metadata",
    });

    const deleteResponse = await request.delete(`/api/identities/${created.face_id}`);
    expect(deleteResponse.status()).toBe(204);
    personId = null;
    expect((await request.get(`/api/identities/${created.face_id}`)).status()).toBe(404);
  } finally {
    if (personId !== null) await request.delete(`/api/persons/${personId}`);
  }
});


test("lists and searches known and anonymous identities in the interface", async ({ page }) => {
  const knownFaceId = "11111111-1111-4111-8111-111111111111";
  const anonymousFaceId = "22222222-2222-4222-8222-222222222222";

  await page.route("**/media/anonymous/test/sample.jpg", async (route) => {
    await route.fulfill({
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64",
      ),
    });
  });

  await page.route("**/api/identities", async (route) => {
    await route.fulfill({
      json: [
        {
          face_id: knownFaceId,
          status: "known",
          person_id: 901,
          first_name: "Test",
          last_name: "Kisisi",
          description: "Kimlik ekran testi",
          sample_count: 2,
          reference_image_count: 2,
          observation_count: 0,
          sample_image_urls: [],
          created_at: "2026-08-19T10:00:00Z",
          updated_at: "2026-08-19T10:00:00Z",
          last_seen_at: null,
        },
        {
          face_id: anonymousFaceId,
          status: "anonymous",
          person_id: null,
          first_name: null,
          last_name: null,
          description: null,
          sample_count: 3,
          reference_image_count: 0,
          observation_count: 3,
          sample_image_urls: ["/media/anonymous/test/sample.jpg"],
          created_at: "2026-08-19T11:00:00Z",
          updated_at: "2026-08-19T11:10:00Z",
          last_seen_at: "2026-08-19T11:10:00Z",
        },
      ],
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Kimlikler" }).first().click();
  await expect(page.getByRole("heading", { name: "Kimlik Kayıtları" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Test Kisisi/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Anonim yüz/ })).toBeVisible();

  await page.getByLabel("Kimlik ara").fill(anonymousFaceId);
  await expect(page.getByRole("button", { name: /Anonim yüz/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Test Kisisi/ })).toHaveCount(0);
  await page.getByRole("button", { name: /Anonim yüz/ }).click();
  await expect(page.getByText(anonymousFaceId, { exact: true })).toBeVisible();
  await expect(page.getByText("anonymous", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: "Anonim yüz örneği 1" })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBeFalsy();
});
