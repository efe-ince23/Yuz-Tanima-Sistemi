import { expect, test } from "@playwright/test";


const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION
  ?? "face_embeddings_arcface_r50_v1";
const QDRANT_COLLECTION_URL = `http://qdrant:6333/collections/${QDRANT_COLLECTION}`;

interface QdrantCollectionResponse {
  result: {
    points_count: number;
    config: {
      params: {
        vectors: { size: number; distance: string };
      };
    };
  };
}


test("stores face embeddings with identity payloads in Qdrant", async ({ request }) => {
  const response = await request.get(QDRANT_COLLECTION_URL);
  expect(response.ok()).toBeTruthy();
  const collection = await response.json() as QdrantCollectionResponse;
  expect(collection.result.points_count).toBeGreaterThan(0);
  expect(collection.result.config.params.vectors).toMatchObject({
    size: 512,
    distance: "Cosine",
  });

  const scrollResponse = await request.post(
    `${QDRANT_COLLECTION_URL}/points/scroll`,
    {
      data: { limit: 1, with_payload: true, with_vector: true },
    },
  );
  expect(scrollResponse.ok()).toBeTruthy();
  const scroll = await scrollResponse.json() as {
    result: {
      points: Array<{
        vector: number[];
        payload: Record<string, unknown>;
      }>;
    };
  };
  expect(scroll.result.points).toHaveLength(1);
  expect(scroll.result.points[0].vector).toHaveLength(512);
  expect(scroll.result.points[0].payload).toMatchObject({
    faceId: expect.any(String),
    personId: expect.any(Number),
    status: "known",
    sampleType: expect.any(String),
    sampleId: expect.any(Number),
  });
});


test("synchronizes an added and deleted face sample with Qdrant", async ({ request }) => {
  const personResponse = await request.get("/api/persons/4");
  expect(personResponse.ok()).toBeTruthy();
  const person = await personResponse.json() as { id: number; face_id: string };

  const imagesResponse = await request.get(`/api/persons/${person.id}/face-images`);
  const images = await imagesResponse.json() as Array<{ id: number; image_url: string }>;
  expect(images.length).toBeGreaterThan(0);
  const sourceImage = await (await request.get(images[0].image_url)).body();

  let createdImageId: number | null = null;
  try {
    const uploadResponse = await request.post(`/api/persons/${person.id}/face-images`, {
      multipart: {
        file: {
          name: "qdrant-sync-test.jpg",
          mimeType: "image/jpeg",
          buffer: sourceImage,
        },
      },
    });
    expect(uploadResponse.status()).toBe(201);
    const created = await uploadResponse.json() as { id: number };
    createdImageId = created.id;

    const storedResponse = await request.post(
      `${QDRANT_COLLECTION_URL}/points/scroll`,
      {
        data: {
          filter: {
            must: [
              { key: "sampleType", match: { value: "face_image" } },
              { key: "sampleId", match: { value: createdImageId } },
            ],
          },
          limit: 10,
          with_payload: true,
          with_vector: false,
        },
      },
    );
    const stored = await storedResponse.json() as { result: { points: unknown[] } };
    expect(stored.result.points).toHaveLength(1);
  } finally {
    if (createdImageId !== null) {
      await request.delete(`/api/persons/${person.id}/face-images/${createdImageId}`);
    }
  }

  const deletedResponse = await request.post(
    `${QDRANT_COLLECTION_URL}/points/scroll`,
    {
      data: {
        filter: {
          must: [
            { key: "sampleType", match: { value: "face_image" } },
            { key: "sampleId", match: { value: createdImageId } },
          ],
        },
        limit: 10,
        with_payload: true,
        with_vector: false,
      },
    },
  );
  const deleted = await deletedResponse.json() as { result: { points: unknown[] } };
  expect(deleted.result.points).toHaveLength(0);
});
