import { expect, test } from "@playwright/test";


type HealthResponse = {
  object_storage: "minio" | "local";
  storage_objects: number | null;
};

type FaceImage = {
  id: number;
  image_url: string;
};


test("serves persisted face images from MinIO", async ({ request }) => {
  const healthResponse = await request.get("http://backend:8000/health");
  expect(healthResponse.ok()).toBeTruthy();
  const health = await healthResponse.json() as HealthResponse;
  expect(health.object_storage).toBe("minio");
  expect(health.storage_objects).toBeGreaterThan(0);

  const imagesResponse = await request.get("http://backend:8000/api/persons/4/face-images");
  expect(imagesResponse.ok()).toBeTruthy();
  const images = await imagesResponse.json() as FaceImage[];
  expect(images.length).toBeGreaterThan(0);

  const imageResponse = await request.get(`http://backend:8000${images[0].image_url}`);
  expect(imageResponse.ok()).toBeTruthy();
  expect(imageResponse.headers()["content-type"]).toContain("image/jpeg");
  expect((await imageResponse.body()).length).toBeGreaterThan(0);
});


test("synchronizes uploaded and deleted face images with MinIO", async ({ request }) => {
  const imagesResponse = await request.get("http://backend:8000/api/persons/4/face-images");
  const images = await imagesResponse.json() as FaceImage[];
  const sourceImage = await (
    await request.get(`http://backend:8000${images[0].image_url}`)
  ).body();

  let createdImage: FaceImage | null = null;
  let createdImageUrl: string | null = null;
  try {
    const uploadResponse = await request.post(
      "http://backend:8000/api/persons/4/face-images",
      {
        multipart: {
          file: {
            name: "minio-storage-test.jpg",
            mimeType: "image/jpeg",
            buffer: sourceImage,
          },
        },
      },
    );
    expect(uploadResponse.status()).toBe(201);
    createdImage = await uploadResponse.json() as FaceImage;
    createdImageUrl = createdImage.image_url;

    const storedImage = await request.get(
      `http://backend:8000${createdImageUrl}`,
    );
    expect(storedImage.ok()).toBeTruthy();
  } finally {
    if (createdImage !== null) {
      const deleteResponse = await request.delete(
        `http://backend:8000/api/persons/4/face-images/${createdImage.id}`,
      );
      expect(deleteResponse.status()).toBe(204);
    }
  }

  expect(createdImageUrl).not.toBeNull();
  const deletedImage = await request.get(
    `http://backend:8000${createdImageUrl}`,
  );
  expect(deletedImage.status()).toBe(404);
});
