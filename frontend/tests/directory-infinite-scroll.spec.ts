import { expect, test } from "@playwright/test";


test("loads person and identity directories in batches of 200", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Kayıtlı kişiler" }).first().click();

  const personDirectory = page.getByTestId("person-directory");
  await expect(personDirectory.locator("[data-person-id]")).toHaveCount(200);
  await expect(personDirectory.locator(".directory-avatar img").first()).toBeVisible();
  await personDirectory.evaluate((directory) => {
    directory.scrollTop = directory.scrollHeight;
    directory.dispatchEvent(new Event("scroll", { bubbles: true }));
  });
  await expect(personDirectory.locator("[data-person-id]")).toHaveCount(400);
  await expect(page.getByTestId("person-directory-progress")).toContainText("400 /");

  await page.getByLabel("Kişi ara").fill("George W Bush");
  await expect(page.getByRole("button", { name: /George W Bush/ })).toBeVisible();

  await page.getByRole("button", { name: "Kimlikler" }).first().click();
  const identityDirectory = page.getByTestId("identity-directory");
  await expect(identityDirectory.locator("[data-face-id]")).toHaveCount(200);
  await identityDirectory.evaluate((directory) => {
    directory.scrollTop = directory.scrollHeight;
    directory.dispatchEvent(new Event("scroll", { bubbles: true }));
  });
  await expect(identityDirectory.locator("[data-face-id]")).toHaveCount(400);
  await expect(page.getByTestId("identity-directory-progress")).toContainText("400 /");
});
