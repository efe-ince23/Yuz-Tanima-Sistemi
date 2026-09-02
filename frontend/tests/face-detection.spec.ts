import { expect, test } from "@playwright/test";


type FaceImage = {
  image_url: string;
};


test("returns no_face and an empty face list when no face exists", async ({ page, request }) => {
  await page.setViewportSize({ width: 200, height: 200 });
  await page.setContent("<style>html,body{margin:0;width:100%;height:100%;background:#fff}</style>");
  const image = await page.screenshot({ type: "png" });

  const response = await request.post("/api/faces/detect", {
    multipart: {
      file: { name: "empty-scene.png", mimeType: "image/png", buffer: image },
    },
  });

  expect(response.status()).toBe(200);
  await expect(response.json()).resolves.toMatchObject({
    status: "no_face",
    face_found: false,
    image_width: 200,
    image_height: 200,
    face_count: 0,
    faces: [],
  });
});


test("detects and locates every face independently", async ({ page, request }) => {
  const referenceImages: Buffer[] = [];
  for (const personId of [4, 5]) {
    const listResponse = await request.get(`/api/persons/${personId}/face-images`);
    expect(listResponse.ok()).toBeTruthy();
    const images = await listResponse.json() as FaceImage[];
    expect(images.length).toBeGreaterThan(0);

    const imageResponse = await request.get(images[0].image_url);
    expect(imageResponse.ok()).toBeTruthy();
    referenceImages.push(await imageResponse.body());
  }

  const dataUrls = referenceImages.map(
    (image) => `data:image/jpeg;base64,${image.toString("base64")}`,
  );

  const singleFaceResponse = await request.post("/api/faces/detect", {
    multipart: {
      file: { name: "single-face.jpg", mimeType: "image/jpeg", buffer: referenceImages[0] },
    },
  });
  expect(singleFaceResponse.status()).toBe(200);
  const singleFaceBody = await singleFaceResponse.json();
  expect(singleFaceBody.status).toBe("faces_detected");
  expect(singleFaceBody.face_found).toBeTruthy();
  expect(singleFaceBody.face_count).toBe(1);
  expect(singleFaceBody.faces).toHaveLength(1);

  await page.setViewportSize({ width: 1000, height: 650 });
  await page.setContent(`
    <style>
      html, body { margin: 0; width: 100%; height: 100%; background: white; overflow: hidden; }
      main { display: grid; grid-template-columns: 1fr 1fr; width: 1000px; height: 650px; }
      img { width: 500px; height: 650px; object-fit: contain; }
    </style>
    <main>
      <img src="${dataUrls[0]}" />
      <img src="${dataUrls[1]}" />
    </main>
  `);
  await Promise.all([
    page.locator("img").nth(0).evaluate((image: HTMLImageElement) => image.decode()),
    page.locator("img").nth(1).evaluate((image: HTMLImageElement) => image.decode()),
  ]);
  const image = await page.screenshot({ type: "png" });

  const response = await request.post("/api/faces/detect", {
    multipart: {
      file: { name: "two-faces.png", mimeType: "image/png", buffer: image },
    },
  });

  expect(response.status()).toBe(200);
  const body = await response.json();
  expect(body.status).toBe("faces_detected");
  expect(body.face_count).toBeGreaterThanOrEqual(2);
  expect(body.faces).toHaveLength(body.face_count);
  expect(body.faces.map((face: { face_index: number }) => face.face_index)).toEqual(
    Array.from({ length: body.face_count }, (_, index) => index),
  );

  for (const face of body.faces) {
    const box = face.bounding_box;
    expect(box.x1).toBeGreaterThanOrEqual(0);
    expect(box.y1).toBeGreaterThanOrEqual(0);
    expect(box.x2).toBeLessThanOrEqual(body.image_width);
    expect(box.y2).toBeLessThanOrEqual(body.image_height);
    expect(box.width).toBe(box.x2 - box.x1);
    expect(box.height).toBe(box.y2 - box.y1);
    expect(box.width).toBeGreaterThan(0);
    expect(box.height).toBeGreaterThan(0);
  }

  const xPositions = body.faces.map((face: { bounding_box: { x1: number } }) => face.bounding_box.x1);
  expect(xPositions).toEqual([...xPositions].sort((left, right) => left - right));
});
