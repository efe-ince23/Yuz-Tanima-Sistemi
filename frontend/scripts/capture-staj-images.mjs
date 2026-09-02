import { chromium } from "playwright";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve("..");
const output = path.join(root, "artifacts", "staj-defteri-gorselleri");
await mkdir(output, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  args: ["--disable-dev-shm-usage"],
});
const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.25 });
const page = await context.newPage();

const save = async (name, locator = null) => {
  const target = path.join(output, name);
  if (locator) await locator.screenshot({ path: target, animations: "disabled" });
  else await page.screenshot({ path: target, animations: "disabled" });
};

const diagramFile = pathToFileURL(path.join(output, "render.html")).href;
await page.goto(diagramFile);
for (const [id, filename] of [
  ["architecture", "01-sistem-mimarisi.png"],
  ["flow", "02-yuz-tanima-islem-akisi.png"],
  ["er", "03-postgresql-er-diyagrami.png"],
  ["timeline", "08-videoda-gorunme-zamanlari.png"],
  ["data-stack", "09-veri-katmanlari.png"],
  ["process", "10-process-id-ve-gecmis.png"],
  ["benchmark", "12-lfw-benchmark.png"],
  ["docker", "13-docker-servisleri.png"],
]) {
  await save(filename, page.locator(`#${id}`));
}

await page.goto("http://localhost:3000", { waitUntil: "networkidle" });
await page.locator(".auth-page").waitFor();
await save("14-giris-ekrani.png");

const login = await context.request.post("http://localhost:3000/api/auth/login", {
  data: { identifier: process.env.E2E_ADMIN_USERNAME ?? "admin", password: process.env.E2E_ADMIN_PASSWORD },
});
if (!login.ok()) throw new Error(`Admin login failed: ${login.status()} ${await login.text()}`);

await page.goto("http://localhost:3000", { waitUntil: "networkidle" });
await page.locator(".app-shell").waitFor();
await save("04-ana-uygulama-arayuzu.png");

await page.getByRole("button", { name: /Video tanıma/i }).first().click();
await page.locator(".video-workspace").waitFor();
await save("06-video-yukleme-ve-analiz.png", page.locator(".video-workspace"));

await page.locator(".video-history-row").first().waitFor({ timeout: 60000 });
const preferredVideo = page.locator(".video-history-row").filter({ hasText: "Beyaz Kıvanç" }).first();
const historyRows = page.locator(".video-history-row");
const historyRow = await preferredVideo.count() ? preferredVideo : historyRows.first();
if (await historyRow.count()) {
  await historyRow.locator(".video-history-open").click();
  await page.locator(".video-completed").waitFor({ timeout: 60000 });
  await page.locator(".video-workspace").scrollIntoViewIfNeeded();

  const video = page.locator(".video-drop-zone video");
  await video.waitFor();
  await video.evaluate((element) => {
    element.currentTime = 12;
    element.dispatchEvent(new Event("timeupdate"));
  });
  await page.waitForTimeout(500);
  const knownFaceCount = await page.locator(".video-face-box.known").count();
  if (knownFaceCount < 2) {
    throw new Error(`12. saniyede iki dogru bilinen yuz bekleniyordu, ${knownFaceCount} bulundu.`);
  }
  const stage = page.locator(".video-drop-zone");
  const sourceFrame = await readFile(path.join(output, ".source-video-frame-12s.jpg"));
  await stage.evaluate((element, dataUrl) => {
    element.style.backgroundImage = `url(${dataUrl})`;
    element.style.backgroundSize = "100% 100%";
    const videoElement = element.querySelector("video");
    if (videoElement) videoElement.style.visibility = "hidden";
  }, `data:image/jpeg;base64,${sourceFrame.toString("base64")}`);
  await save("05-coklu-yuz-tanima-sonucu.png", stage);

  const kivancRow = page.locator(".video-track-row").filter({ hasText: "Kıvanç Tatlıtuğ" }).first();
  if (await kivancRow.count()) {
    await kivancRow.scrollIntoViewIfNeeded();
  }

  const workspace = page.locator(".video-workspace");
  await workspace.evaluate((element) => {
    element.dataset.captureStyle = element.getAttribute("style") ?? "";
    element.style.height = "800px";
    element.style.overflow = "hidden";
  });
  await save("07-final-video-analiz-sonucu.png", workspace);
  await stage.evaluate((element) => {
    element.style.backgroundImage = "";
    element.style.backgroundSize = "";
    const videoElement = element.querySelector("video");
    if (videoElement) videoElement.style.visibility = "";
  });
  await workspace.evaluate((element) => {
    const previous = element.dataset.captureStyle ?? "";
    if (previous) element.setAttribute("style", previous);
    else element.removeAttribute("style");
    delete element.dataset.captureStyle;
  });

  const knownTrack = page.locator(".video-track-row").filter({ hasText: /Bilinen kişi/i }).first();
  if (await knownTrack.count()) await knownTrack.click();
}

await page.goto("http://localhost:8000/docs", { waitUntil: "networkidle" });
await page.locator(".swagger-ui").waitFor();
await page.evaluate(() => window.scrollTo(0, 0));
await save("11-swagger-api-endpointleri.png");

await browser.close();
