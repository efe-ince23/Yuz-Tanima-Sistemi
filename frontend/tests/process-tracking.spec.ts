import { expect, test } from "@playwright/test";


test("returns a unique and queryable process ID for every face request", async ({ request }) => {
  const emptyScene = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    "base64",
  );

  const processIds: string[] = [];
  for (let index = 0; index < 2; index += 1) {
    const response = await request.post("/api/faces/detect", {
      multipart: {
        file: { name: `empty-${index}.png`, mimeType: "image/png", buffer: emptyScene },
      },
    });
    expect(response.status()).toBe(200);
    const body = await response.json() as { process_id: string; status: string };
    expect(body.status).toBe("no_face");
    expect(response.headers()["x-process-id"]).toBe(body.process_id);
    processIds.push(body.process_id);

    const processResponse = await request.get(`/api/processes/${body.process_id}`);
    expect(processResponse.ok()).toBeTruthy();
    await expect(processResponse.json()).resolves.toMatchObject({
      process_id: body.process_id,
      operation_type: "detect",
      status: "no_face",
      http_status: 200,
      face_count: 0,
      task_detail: {
        operation_type: "detect",
        processed_face_count: 0,
        faces: [],
        status: "no_face",
      },
      events: [],
    });
  }

  expect(processIds[0]).not.toBe(processIds[1]);
});


test("tracks invalid requests and returns their process ID in the error", async ({ request }) => {
  const response = await request.post("/api/faces/identify");
  expect(response.status()).toBe(422);
  const body = await response.json() as { process_id: string };
  expect(body.process_id).toBeTruthy();
  expect(response.headers()["x-process-id"]).toBe(body.process_id);

  const processResponse = await request.get(`/api/processes/${body.process_id}`);
  await expect(processResponse.json()).resolves.toMatchObject({
    process_id: body.process_id,
    operation_type: "identify",
    status: "failed",
    http_status: 422,
    face_count: 0,
    task_detail: {
      operation_type: "identify",
      processed_face_count: 0,
      faces: [],
      status: "failed",
    },
  });
});


test("tracks face comparison as a separate process", async ({ request }) => {
  const faceList = await request.get("/api/persons/4/face-images");
  const faces = await faceList.json() as Array<{ image_url: string }>;
  const imageResponse = await request.get(faces[0].image_url);
  const image = await imageResponse.body();

  const response = await request.post("/api/faces/compare", {
    multipart: {
      image_a: { name: "face-a.jpg", mimeType: "image/jpeg", buffer: image },
      image_b: { name: "face-b.jpg", mimeType: "image/jpeg", buffer: image },
    },
  });
  expect(response.status()).toBe(200);
  const body = await response.json() as { process_id: string; same_person: boolean };
  expect(body.same_person).toBeTruthy();

  const processResponse = await request.get(`/api/processes/${body.process_id}`);
  await expect(processResponse.json()).resolves.toMatchObject({
    process_id: body.process_id,
    operation_type: "compare",
    status: "completed",
    http_status: 200,
    face_count: 2,
    task_detail: {
      operation_type: "compare",
      processed_face_count: 2,
      status: "completed",
    },
  });
});


test("returns 404 for an unknown process ID", async ({ request }) => {
  const response = await request.get("/api/processes/00000000-0000-4000-8000-000000000000");
  expect(response.status()).toBe(404);
});
