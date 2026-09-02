import { expect, test } from "@playwright/test";

test("unauthenticated visitors see the login and registration screen", async ({ browser }) => {
  const context = await browser.newContext({ storageState: undefined });
  const page = await context.newPage();
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Yüz Tanıma Sistemi" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Giriş", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Kayıt ol" })).toBeVisible();
  await context.close();
});

test("the administrator sees management navigation", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Sistem Yöneticisi")).toBeVisible();
  await expect(page.getByRole("button", { name: /Kayıtlı kişiler/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Kullanıcılar" })).toBeVisible();
});
