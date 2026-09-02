import { expect, test } from "@playwright/test";


test("shows recognition totals and success rate", async ({ page }) => {
  await page.route("**/api/statistics", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        total_operations: 20,
        recognized_count: 15,
        unrecognized_count: 5,
        success_rate: 75,
        latest_event_at: "2026-08-18T07:18:54Z",
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "İstatistikler" }).click();

  await expect(page.getByRole("heading", { name: "İstatistikler" })).toBeVisible();
  await expect(page.locator(".metric-card.total")).toContainText("20");
  await expect(page.locator(".metric-card.recognized")).toContainText("15");
  await expect(page.locator(".metric-card.unrecognized")).toContainText("5");
  await expect(page.locator(".metric-card.rate")).toContainText("%75.00");
  await expect(page.locator(".distribution-track")).toBeVisible();
});


test("fits the statistics screen on a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByTitle("İstatistikler").click();
  await expect(page.getByRole("heading", { name: "İstatistikler" })).toBeVisible();

  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBeFalsy();
});
