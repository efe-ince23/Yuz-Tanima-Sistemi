import { expect, test } from "@playwright/test";

interface PersonRecord {
  id: number;
  first_name: string;
  last_name: string;
  face_image_count: number;
}

const TEST_FIRST_NAME = "ArayuzTest";
const DUPLICATE_TEST_FIRST_NAME = "TekrarTest";

test("edits and deletes a temporary person through the interface", async ({ page, request }) => {
  const cleanup = async () => {
    const response = await request.get("/api/persons");
    if (!response.ok()) return;
    const people = (await response.json()) as PersonRecord[];
    for (const person of people.filter((item) => item.first_name === TEST_FIRST_NAME)) {
      await request.delete(`/api/persons/${person.id}`);
    }
  };

  await cleanup();

  try {
    const createResponse = await request.post("/api/persons", {
      data: {
        first_name: TEST_FIRST_NAME,
        last_name: "Gecici",
        description: "Otomatik arayüz testi",
      },
    });
    expect(createResponse.ok()).toBeTruthy();
    const created = (await createResponse.json()) as PersonRecord;

    await page.goto("/");
    await page.getByRole("button", { name: "Kayıtlı kişiler" }).click();
    await page.locator(`[data-person-id="${created.id}"]`).click();
    await expect(page.getByRole("heading", { name: `${TEST_FIRST_NAME} Gecici` })).toBeVisible();
    await expect(page.getByText("Referans fotoğrafı yok")).toBeVisible();

    await page.getByTitle("Kişiyi düzenle").click();
    const editDialog = page.getByRole("dialog", { name: "Kişiyi düzenle" });
    await editDialog.getByLabel("Soyad", { exact: true }).fill("Guncel");
    await editDialog.getByRole("button", { name: "Kaydet" }).click();
    await expect(page.getByRole("heading", { name: `${TEST_FIRST_NAME} Guncel` })).toBeVisible();

    await page.getByTitle("Kişiyi sil").click();
    await page.getByRole("alertdialog").getByRole("button", { name: "Kalıcı olarak sil" }).click();
    await expect(page.getByRole("heading", { name: `${TEST_FIRST_NAME} Guncel` })).toHaveCount(0);
    await expect(page.getByText("Kişi ve referans fotoğrafları silindi.")).toBeVisible();
  } finally {
    await cleanup();
  }
});

test("blocks a face that is already registered to another person", async ({ page, request }) => {
  const cleanup = async () => {
    const response = await request.get("/api/persons");
    if (!response.ok()) return;
    const people = (await response.json()) as PersonRecord[];
    for (const person of people.filter((item) => item.first_name === DUPLICATE_TEST_FIRST_NAME)) {
      await request.delete(`/api/persons/${person.id}`);
    }
  };

  await cleanup();
  const peopleBefore = (await (await request.get("/api/persons")).json()) as PersonRecord[];
  const faceResponse = await request.get("/api/persons/4/face-images");
  expect(faceResponse.ok()).toBeTruthy();
  const faces = (await faceResponse.json()) as Array<{ image_url: string }>;
  const imageResponse = await request.get(faces[0].image_url);
  expect(imageResponse.ok()).toBeTruthy();
  const image = await imageResponse.body();

  try {
    await page.goto("/");
    await page.getByRole("button", { name: "Kayıtlı kişiler" }).click();
    await page.getByRole("button", { name: "Yeni kişi ekle" }).click();

    const dialog = page.getByRole("dialog", { name: "Yeni kişi ekle" });
    await dialog.getByLabel("Ad", { exact: true }).fill(DUPLICATE_TEST_FIRST_NAME);
    await dialog.getByLabel("Soyad", { exact: true }).fill("Kontrol");
    await dialog.locator('input[type="file"]').setInputFiles({
      name: "duplicate-face.jpg",
      mimeType: "image/jpeg",
      buffer: image,
    });
    await dialog.getByRole("button", { name: "Kaydet" }).click();

    await expect(dialog.getByText("Bu yüz zaten Fatih Terim", { exact: false })).toBeVisible();
    await expect(dialog.getByText("benzerlik:", { exact: false })).toBeVisible();

    const peopleAfter = (await (await request.get("/api/persons")).json()) as PersonRecord[];
    expect(peopleAfter.map((person) => person.id)).toEqual(peopleBefore.map((person) => person.id));
  } finally {
    await cleanup();
  }
});
