import { expect, test } from "@playwright/test";


test("shows the signed-in user's photo recognition history below recognition", async ({ page }) => {
  const processId = "11111111-2222-4333-8444-555555555555";
  await page.route("**/api/photos?**", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        total: 1,
        limit: 12,
        offset: 0,
        items: [{
          process_id: processId,
          status: "recognized",
          face_count: 1,
          original_filename: "test-fotografi.jpg",
          owner_username: "admin",
          owner_full_name: "Sistem Yöneticisi",
          image_url: `/api/photos/${processId}/content`,
          image_width: 640,
          image_height: 480,
          created_at: "2026-08-28T07:00:00Z",
          completed_at: "2026-08-28T07:00:01Z",
          result: {
            process_id: processId,
            status: "recognized",
            recognized: true,
            similarity: 0.92,
            threshold: 0.45,
            person: { id: 4, first_name: "Fatih", last_name: "Terim", description: null },
            face_id: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            matched_image_url: null,
            execution_providers: ["CUDAExecutionProvider"],
            detected_face_count: 1,
            ignored_face_count: 0,
            faces: [{
              face_index: 0,
              face_id: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
              status: "known",
              recognized: true,
              similarity: 0.92,
              person: { id: 4, first_name: "Fatih", last_name: "Terim", description: null },
              matched_image_url: null,
              detection_confidence: 0.99,
              bounding_box: { x1: 10, y1: 20, x2: 210, y2: 260, width: 200, height: 240 },
            }],
          },
        }],
      }),
    });
  });

  await page.goto("/");
  const history = page.getByRole("region", { name: "Fotoğraf geçmişi" });
  await expect(history).toBeVisible();
  await expect(history).toContainText("test-fotografi.jpg");
  await expect(history).toContainText("Fatih Terim");
  await expect(history).toContainText("1 bilinen, 0 anonim");
  await expect(history).toContainText(processId.slice(0, 8));
  await history.getByRole("button", { name: "test-fotografi.jpg sonucunu aç" }).click();
  await expect(page.locator(".result-panel")).toContainText("Fatih Terim");
  await expect(page.getByAltText("Seçilen test fotoğrafı")).toHaveAttribute(
    "src",
    `/api/photos/${processId}/content`,
  );
  const clearButton = page.getByTitle("Fotoğrafı kaldır");
  await expect(clearButton).toBeEnabled();
  await clearButton.click();
  await expect(page.getByAltText("Seçilen test fotoğrafı")).toHaveCount(0);
  await expect(page.locator(".result-panel")).toContainText("Sonuç bekleniyor");
});


test("hides administrative photo details from a normal user", async ({ page }) => {
  const processId = "22222222-3333-4444-8555-666666666666";
  await page.route("**/api/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        id: "77777777-8888-4999-8aaa-bbbbbbbbbbbb",
        username: "normaluser",
        email: "normal@example.com",
        full_name: "Normal Kullanıcı",
        role: "user",
        is_active: true,
        created_at: "2026-08-28T07:00:00Z",
        updated_at: "2026-08-28T07:00:00Z",
        last_login_at: "2026-08-28T07:00:00Z",
      }),
    });
  });
  await page.route("**/api/photos?**", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        total: 1,
        limit: 12,
        offset: 0,
        items: [{
          process_id: processId,
          status: "no_face",
          face_count: 0,
          original_filename: "kullanici-fotografi.jpg",
          owner_username: "normaluser",
          owner_full_name: "Normal Kullanıcı",
          image_url: `/api/photos/${processId}/content`,
          image_width: 640,
          image_height: 480,
          created_at: "2026-08-28T07:00:00Z",
          completed_at: "2026-08-28T07:00:01Z",
          result: {
            process_id: processId,
            status: "no_face",
            recognized: false,
            similarity: null,
            threshold: 0.45,
            person: null,
            face_id: null,
            matched_image_url: null,
            execution_providers: ["CUDAExecutionProvider"],
            detected_face_count: 0,
            ignored_face_count: 0,
            faces: [],
          },
        }],
      }),
    });
  });

  await page.goto("/");
  const history = page.getByRole("region", { name: "Fotoğraf geçmişi" });
  await expect(history).toContainText("kullanici-fotografi.jpg");
  await expect(history).not.toContainText("Process ID");
  await expect(history).not.toContainText("Gönderen:");
  await history.getByRole("button", { name: "kullanici-fotografi.jpg sonucunu aç" }).click();
  await expect(page.locator(".result-panel")).toContainText("Fotoğrafta yüz bulunamadı");
  await expect(page.locator(".process-reference")).toHaveCount(0);
});
