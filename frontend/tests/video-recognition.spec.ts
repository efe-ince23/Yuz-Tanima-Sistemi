import { expect, test } from "@playwright/test";


test("uploads and follows a video job through the interface", async ({ page }) => {
  const processId = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
  let statusReads = 0;
  let resultReads = 0;
  let deleted = false;
  const job = (status: "queued" | "processing" | "completed") => ({
    process_id: processId,
    status,
    original_filename: "test.mp4",
    object_path: `videos/${processId}/source.mp4`,
    content_type: "video/mp4",
    file_size_bytes: 1024,
    duration_seconds: 2,
    source_fps: 10,
    width: 800,
    height: 500,
    frame_count: 20,
    sampled_frame_count: status === "completed" ? 6 : 0,
    processed_frame_count: status === "completed" ? 6 : 0,
    progress_percent: status === "completed" ? 100 : 0,
    detected_face_count: status === "completed" ? 12 : 0,
    unique_face_count: status === "completed" ? 2 : 0,
    error_code: null,
    error_detail: null,
    created_at: "2026-08-24T10:00:00Z",
    updated_at: "2026-08-24T10:00:01Z",
    started_at: status === "queued" ? null : "2026-08-24T10:00:00Z",
    completed_at: status === "completed" ? "2026-08-24T10:00:01Z" : null,
  });

  await page.route("**/api/videos?*", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ total: 1, limit: 20, offset: 0, items: [job("completed")] }),
    });
  });
  await page.route("**/api/videos/faces/*/history?*", async (route) => {
    const faceId = route.request().url().split("/faces/")[1].split("/history")[0];
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        face_id: faceId,
        total: 1,
        limit: 20,
        offset: 0,
        items: [{
          process_id: processId,
          original_filename: "test.mp4",
          created_at: "2026-08-24T10:00:00Z",
          duration_seconds: 2,
          first_seen_ms: 0,
          last_seen_ms: 1700,
          observation_count: 6,
          appearances: [{
            start_ms: 0,
            end_ms: 1700,
            start_frame: 0,
            end_frame: 17,
            observation_count: 6,
            max_recognition_confidence: 0.95,
            average_recognition_confidence: 0.93,
          }],
        }],
      }),
    });
  });

  await page.route("**/api/videos", async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify(job("queued")) });
      return;
    }
    await route.continue();
  });
  await page.route(`**/api/videos/${processId}`, async (route) => {
    if (route.request().method() === "DELETE") {
      deleted = true;
      await route.fulfill({ status: 204 });
      return;
    }
    statusReads += 1;
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(job(statusReads > 1 ? "completed" : "processing")),
    });
  });
  await page.route(`**/api/videos/${processId}/result`, async (route) => {
    resultReads += 1;
    if (resultReads === 1) {
      await route.abort("connectionfailed");
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        process_id: processId,
        status: "completed",
        video_url: `/media/videos/${processId}/source.mp4`,
        duration_seconds: 2,
        sampled_frame_count: 6,
        detected_face_count: 12,
        unique_face_count: 2,
        tracks: [
          {
            track_id: 1,
            face_id: "11111111-1111-4111-8111-111111111111",
            status: "known",
            name: "Fatih Terim",
            metadata: { description: "Teknik direktör" },
            first_seen_ms: 0,
            last_seen_ms: 1700,
            observation_count: 6,
            best_detection_confidence: 0.9,
            best_recognition_confidence: 0.95,
            best_frame_number: 10,
            best_image_url: null,
            appearances: [{
              start_ms: 0,
              end_ms: 1700,
              start_frame: 0,
              end_frame: 17,
              observation_count: 2,
              max_recognition_confidence: 0.95,
              average_recognition_confidence: 0.95,
            }],
            observations: [
              { frame_number: 0, timestamp_ms: 0, bounding_box: { x1: 0.1, y1: 0.2, x2: 0.3, y2: 0.6 }, detection_confidence: 0.9, recognition_confidence: 0.95 },
              { frame_number: 5, timestamp_ms: 500, bounding_box: { x1: 0.15, y1: 0.2, x2: 0.35, y2: 0.6 }, detection_confidence: 0.9, recognition_confidence: 0.95 },
              { frame_number: 10, timestamp_ms: 1000, bounding_box: { x1: 0.2, y1: 0.2, x2: 0.4, y2: 0.6 }, detection_confidence: 0.9, recognition_confidence: 0.95 },
            ],
          },
          {
            track_id: 2,
            face_id: "22222222-2222-4222-8222-222222222222",
            status: "anonymous",
            name: null,
            metadata: null,
            first_seen_ms: 300,
            last_seen_ms: 1700,
            observation_count: 4,
            best_detection_confidence: 0.85,
            best_recognition_confidence: 0.72,
            best_frame_number: 7,
            best_image_url: null,
            appearances: [
              {
                start_ms: 300,
                end_ms: 700,
                start_frame: 3,
                end_frame: 7,
                observation_count: 2,
                max_recognition_confidence: 0.72,
                average_recognition_confidence: 0.72,
              },
              {
                start_ms: 1500,
                end_ms: 1700,
                start_frame: 15,
                end_frame: 17,
                observation_count: 2,
                max_recognition_confidence: 0.70,
                average_recognition_confidence: 0.70,
              },
            ],
            observations: [
              { frame_number: 3, timestamp_ms: 300, bounding_box: { x1: 0.65, y1: 0.15, x2: 0.8, y2: 0.55 }, detection_confidence: 0.85, recognition_confidence: 0.72 },
              { frame_number: 5, timestamp_ms: 500, bounding_box: { x1: 0.68, y1: 0.15, x2: 0.83, y2: 0.55 }, detection_confidence: 0.85, recognition_confidence: 0.72 },
              { frame_number: 7, timestamp_ms: 700, bounding_box: { x1: 0.7, y1: 0.15, x2: 0.85, y2: 0.55 }, detection_confidence: 0.85, recognition_confidence: 0.72 },
              { frame_number: 15, timestamp_ms: 1500, bounding_box: { x1: 0.68, y1: 0.15, x2: 0.83, y2: 0.55 }, detection_confidence: 0.84, recognition_confidence: 0.70 },
              { frame_number: 17, timestamp_ms: 1700, bounding_box: { x1: 0.7, y1: 0.15, x2: 0.85, y2: 0.55 }, detection_confidence: 0.84, recognition_confidence: 0.70 },
            ],
          },
        ],
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Video tanıma" }).first().click();
  await page.locator('.video-upload-panel input[type="file"]').setInputFiles({
    name: "test.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("test-video"),
  });
  await page.getByRole("button", { name: "Videoyu analiz et" }).click();

  await expect(page.getByText("Video analizi tamamlandı")).toBeVisible();
  expect(resultReads).toBeGreaterThan(1);
  await expect(page.locator(".video-track-list").getByText("Fatih Terim")).toBeVisible();
  await expect(page.locator(".video-track-list").getByText("Anonim · 22222222")).toHaveCount(2);
  await expect(page.getByText("1 bilinen, 1 anonim kişi")).toBeVisible();
  await expect(page.locator(".process-reference strong")).toHaveText(processId);
  await expect(page.locator(".video-timeline-heading strong")).toHaveText("Fatih Terim");
  await expect(page.locator(".video-face-history")).toContainText("test.mp4");
  await expect(page.locator(".video-face-history")).toContainText("1 video");
  await expect(page.locator(".video-track-row")).toHaveCount(3);
  await expect(page.locator(".video-track-row").nth(0)).toContainText("0:00.0 – 0:01.7");
  await expect(page.locator(".video-track-row").nth(1)).toContainText("0:00.3 – 0:00.7");
  await expect(page.locator(".video-track-row").nth(2)).toContainText("0:01.5 – 0:01.7");
  await page.locator(".video-track-row").nth(2).click();
  await expect(page.locator(".video-timeline-heading strong")).toHaveText("Anonim · 22222222");
  await expect.poll(
    () => page.locator(".video-drop-zone video").evaluate((video) => video.currentTime),
  ).toBeCloseTo(1.5, 1);
  await page.locator(".video-drop-zone video").evaluate((video) => {
    Object.defineProperty(video, "currentTime", { configurable: true, value: 0.5 });
    video.dispatchEvent(new Event("timeupdate"));
  });
  await expect(page.locator(".video-face-box")).toHaveCount(2);
  await expect(page.locator(".video-face-box.known")).toContainText("Fatih Terim");
  await expect(page.locator(".video-face-box.anonymous")).toContainText("Anonim · 22222222");
  await page.locator(".video-drop-zone video").evaluate((video) => {
    Object.defineProperty(video, "currentTime", { configurable: true, value: 1.85 });
    video.dispatchEvent(new Event("timeupdate"));
  });
  await expect(page.locator(".video-face-box")).toHaveCount(0);

  await expect(page.getByRole("heading", { name: "Video geçmişi" })).toBeVisible();
  await expect(page.locator(".video-history-row")).toHaveCount(1);
  await page.locator(".video-history-row").click();
  await expect(page.getByText("Video analizi tamamlandı")).toBeVisible();
  await expect(page.locator(".video-drop-zone video")).toHaveAttribute(
    "src",
    `/media/videos/${processId}/source.mp4`,
  );
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "test.mp4 video analizini sil" }).click();
  await expect(page.locator(".video-history-row")).toHaveCount(0);
  expect(deleted).toBe(true);
});


test("runs live camera recognition inside the video screen", async ({ page }) => {
  let recognitionRequests = 0;
  let liveRecordingUploads = 0;
  await page.route("**/api/videos/live-recordings", async (route) => {
    liveRecordingUploads += 1;
    await route.fulfill({
      contentType: "application/json",
      status: 202,
      body: JSON.stringify({
        process_id: "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        status: "completed",
        original_filename: "canli-kamera-test.mp4",
        object_path: "videos/dddddddd-dddd-4ddd-8ddd-dddddddddddd/source.mp4",
        content_type: "video/mp4",
        file_size_bytes: 2048,
        duration_seconds: 2,
        source_fps: 30,
        width: 640,
        height: 480,
        frame_count: 60,
        sampled_frame_count: 12,
        processed_frame_count: 12,
        progress_percent: 100,
        detected_face_count: 1,
        unique_face_count: 1,
        error_code: null,
        error_detail: null,
        created_at: "2026-08-27T08:00:00Z",
        updated_at: "2026-08-27T08:00:02Z",
        started_at: "2026-08-27T08:00:00Z",
        completed_at: "2026-08-27T08:00:02Z",
      }),
    });
  });
  await page.route("**/api/faces/identify", async (route) => {
    recognitionRequests += 1;
    expect(route.request().headers()["x-recognition-source"]).toBe("live_video_frame");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        process_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        status: "recognized",
        recognized: true,
        similarity: 0.91,
        threshold: 0.45,
        person: {
          id: 4,
          first_name: "Fatih",
          last_name: "Terim",
          description: "Teknik direktör",
        },
        face_id: "11111111-1111-4111-8111-111111111111",
        matched_image_url: null,
        execution_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        detected_face_count: 1,
        ignored_face_count: 0,
        faces: [{
          face_index: 0,
          face_id: "11111111-1111-4111-8111-111111111111",
          status: "known",
          recognized: true,
          similarity: 0.91,
          person: {
            id: 4,
            first_name: "Fatih",
            last_name: "Terim",
            description: "Teknik direktör",
          },
          matched_image_url: null,
          detection_confidence: 0.96,
          bounding_box: {
            x1: 70,
            y1: 35,
            x2: 210,
            y2: 190,
            width: 140,
            height: 155,
          },
        }],
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Video tanıma" }).first().click();
  await page.getByRole("button", { name: "Canlı kamerayı aç" }).click();

  const dialog = page.getByRole("dialog", { name: "Canlı kamera analizi" });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator("video")).toBeVisible();
  await expect(dialog.locator(".live-face-box.known")).toContainText("Fatih Terim");
  await expect(dialog.locator(".live-face-list")).toContainText("Bilinen kişi");
  await expect(dialog.locator(".live-process-id")).toContainText(
    "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
  );
  await expect.poll(() => recognitionRequests).toBeGreaterThan(0);

  await expect(dialog.locator(".live-recording-indicator")).toContainText("REC");
  await dialog.locator(".live-save-button").click();
  await expect(dialog).toHaveCount(0);
  expect(liveRecordingUploads).toBe(1);
  await page.waitForTimeout(500);
  const requestsAfterClose = recognitionRequests;
  await page.waitForTimeout(1000);
  expect(recognitionRequests).toBe(requestsAfterClose);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Canlı kamerayı aç" }).click();
  const mobileDialog = page.getByRole("dialog", { name: "Canlı kamera analizi" });
  await expect(mobileDialog).toBeVisible();
  await expect(mobileDialog.locator(".live-recording-indicator")).toContainText("REC");
  await page.waitForTimeout(1100);
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBeFalsy();
  const dialogBox = await mobileDialog.boundingBox();
  expect(dialogBox?.width ?? 999).toBeLessThanOrEqual(390);
  await mobileDialog.locator(".live-save-button").click();
  await expect(mobileDialog).toHaveCount(0);
  expect(liveRecordingUploads).toBe(2);
});
