import { expect, test } from "@playwright/test";


interface ErrorBody {
  error: {
    code: string;
    message: string;
    details: unknown;
  };
  process_id: string | null;
  timestamp: string;
}


function expectStandardError(body: ErrorBody, code: string): void {
  expect(body.error.code).toBe(code);
  expect(body.error.message.length).toBeGreaterThan(0);
  expect(Number.isNaN(Date.parse(body.timestamp))).toBeFalsy();
  expect(body).toHaveProperty("process_id");
  expect(body.error).toHaveProperty("details");
}


test("documents input, output and standard error contracts for every endpoint", async ({ request }) => {
  const response = await request.get("/openapi.json");
  expect(response.ok()).toBeTruthy();
  const document = await response.json() as {
    paths: Record<string, Record<string, {
      requestBody?: unknown;
      responses: Record<string, {
        content?: { "application/json"?: { schema?: { $ref?: string } } };
      }>;
    }>>;
    components: { schemas: Record<string, { required?: string[] }> };
  };

  expect(document.components.schemas.ApiErrorResponse).toBeTruthy();
  const methods = new Set(["get", "post", "patch", "delete"]);
  const documentedOperations: string[] = [];
  for (const [path, pathItem] of Object.entries(document.paths)) {
    for (const [method, operation] of Object.entries(pathItem)) {
      if (!methods.has(method)) continue;
      documentedOperations.push(`${method.toUpperCase()} ${path}`);
      const successCode = Object.keys(operation.responses).find((code) => code.startsWith("2"));
      expect(successCode, `${method.toUpperCase()} ${path} success response`).toBeTruthy();
      if (successCode !== "204") {
        expect(
          operation.responses[successCode!].content?.["application/json"]?.schema,
          `${method.toUpperCase()} ${path} output schema`,
        ).toBeTruthy();
      }
      expect(
        operation.responses["422"].content?.["application/json"]?.schema?.$ref,
        `${method.toUpperCase()} ${path} error schema`,
      ).toBe("#/components/schemas/ApiErrorResponse");
      if (method === "post" || method === "patch") {
        expect(operation.requestBody, `${method.toUpperCase()} ${path} input schema`).toBeTruthy();
      }
    }
  }

  expect(documentedOperations.sort()).toEqual([
    "DELETE /faces/{faceId}",
    "GET /faces/{faceId}",
    "GET /faces/{faceId}/history",
    "GET /processes/{processId}",
    "POST /faces/enroll",
    "POST /faces/recognize",
  ]);

  const identifyRequired = document.components.schemas.PublicFaceRecognitionResponse.required ?? [];
  expect(identifyRequired).toEqual(expect.arrayContaining([
    "processId",
    "status",
    "detectedFaceCount",
    "faces",
  ]));
  const faceRequired = document.components.schemas.PublicRecognizedFaceResponse.required ?? [];
  expect(faceRequired).toEqual(expect.arrayContaining([
    "faceId",
    "status",
    "name",
    "metadata",
    "boundingBox",
    "confidence",
  ]));
});


test("returns distinguishable standard errors for validation and missing resources", async ({ request }) => {
  const validationResponse = await request.post("/faces/recognize");
  expect(validationResponse.status()).toBe(422);
  const validation = await validationResponse.json() as ErrorBody;
  expectStandardError(validation, "VALIDATION_ERROR");
  expect(validation.process_id).toBeTruthy();
  expect(Array.isArray(validation.error.details)).toBeTruthy();
  expect(validationResponse.headers()["x-process-id"]).toBe(validation.process_id);

  const missingResponse = await request.get(
    "/faces/00000000-0000-4000-8000-000000000000",
  );
  expect(missingResponse.status()).toBe(404);
  const missing = await missingResponse.json() as ErrorBody;
  expectStandardError(missing, "IDENTITY_NOT_FOUND");
  expect(missing.process_id).toBeNull();
});


test("reports oversized files with a stable code and process ID", async ({ request }) => {
  const response = await request.post("/faces/recognize", {
    multipart: {
      file: {
        name: "oversized.jpg",
        mimeType: "image/jpeg",
        buffer: Buffer.alloc(10 * 1024 * 1024 + 1, 1),
      },
    },
  });
  expect(response.status()).toBe(413);
  const body = await response.json() as ErrorBody;
  expectStandardError(body, "FILE_TOO_LARGE");
  expect(body.process_id).toBeTruthy();
  expect(body.error.details).toMatchObject({ max_bytes: 10 * 1024 * 1024 });
});


test("connects public recognition, face history and process detail contracts", async ({ request }) => {
  const peopleResponse = await request.get("/api/persons");
  const people = await peopleResponse.json() as Array<{
    id: number;
    face_id: string;
    first_name: string;
    last_name: string;
    description: string | null;
  }>;
  const person = people.find((item) => item.id === 4) ?? people[0];
  expect(person).toBeTruthy();

  const imagesResponse = await request.get(
    `/api/persons/${person.id}/face-images`,
  );
  const images = await imagesResponse.json() as Array<{ image_url: string }>;
  expect(images.length).toBeGreaterThan(0);
  const image = await (
    await request.get(images[0].image_url)
  ).body();

  const recognizeResponse = await request.post("/faces/recognize", {
    multipart: {
      file: { name: "known-face.jpg", mimeType: "image/jpeg", buffer: image },
    },
  });
  expect(recognizeResponse.ok()).toBeTruthy();
  const recognition = await recognizeResponse.json() as {
    processId: string;
    detectedFaceCount: number;
    faces: Array<{
      faceId: string;
      status: string;
      name: string | null;
      metadata: { description?: string } | null;
      boundingBox: Record<string, number>;
      confidence: number | null;
    }>;
  };
  expect(recognition.detectedFaceCount).toBe(recognition.faces.length);
  expect(recognition.faces).toEqual(expect.arrayContaining([
    expect.objectContaining({
      faceId: person.face_id,
      status: "known",
      name: `${person.first_name} ${person.last_name}`,
      metadata: person.description ? { description: person.description } : {},
      boundingBox: expect.objectContaining({
        x1: expect.any(Number),
        y1: expect.any(Number),
        x2: expect.any(Number),
        y2: expect.any(Number),
      }),
      confidence: expect.any(Number),
    }),
  ]));

  const faceResponse = await request.get(
    `/faces/${person.face_id}`,
  );
  await expect(faceResponse.json()).resolves.toMatchObject({
    face_id: person.face_id,
    status: "known",
  });

  const historyResponse = await request.get(
    `/faces/${person.face_id}/history`,
  );
  const history = await historyResponse.json() as {
    appearances: Array<{ process_id: string }>;
  };
  expect(history.appearances.map((item) => item.process_id)).toContain(recognition.processId);

  const processResponse = await request.get(
    `/processes/${recognition.processId}`,
  );
  await expect(processResponse.json()).resolves.toMatchObject({
    process_id: recognition.processId,
    operation_type: "identify",
    face_count: recognition.faces.length,
    result: recognition,
  });
});


test("updates and deletes a temporary identity through the public face API", async ({ request }) => {
  const createdResponse = await request.post("/api/persons", {
    data: {
      first_name: "PublicApiTest",
      last_name: "Temporary",
      description: "temporary contract record",
    },
  });
  expect(createdResponse.status()).toBe(201);
  const created = await createdResponse.json() as { face_id: string };

  try {
    const enrollResponse = await request.post("/faces/enroll", {
      multipart: {
        faceId: created.face_id,
        first_name: "PublicApiUpdated",
        last_name: "Temporary",
        description: "updated through public enroll",
      },
    });
    expect(enrollResponse.status()).toBe(201);
    await expect(enrollResponse.json()).resolves.toMatchObject({
      face_id: created.face_id,
      status: "known",
      first_name: "PublicApiUpdated",
      description: "updated through public enroll",
    });

    const deleteResponse = await request.delete(
      `/faces/${created.face_id}`,
    );
    expect(deleteResponse.status()).toBe(204);
    const missingResponse = await request.get(
      `/faces/${created.face_id}`,
    );
    expect(missingResponse.status()).toBe(404);
  } finally {
    const peopleResponse = await request.get("/api/persons");
    if (peopleResponse.ok()) {
      const people = await peopleResponse.json() as Array<{ id: number; first_name: string }>;
      for (const person of people.filter((item) => item.first_name.startsWith("PublicApi"))) {
        await request.delete(`/api/persons/${person.id}`);
      }
    }
  }
});
