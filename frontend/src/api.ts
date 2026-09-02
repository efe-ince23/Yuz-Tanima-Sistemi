import type {
  FaceImage,
  FaceHistory,
  HealthResponse,
  IdentifyResponse,
  Identity,
  Person,
  PersonInput,
  RecognitionStatistics,
  RecognitionProcess,
  VideoJob,
  VideoJobList,
  LiveVideoManifest,
  VideoFaceHistory,
  VideoResult,
  AppUser,
  AuthResponse,
  LoginInput,
  RegisterInput,
  PhotoHistoryList,
  AppearanceSearchParams,
  AppearanceSearchResponse,
} from "./types";

let refreshRequest: Promise<boolean> | null = null;

async function refreshSession(): Promise<boolean> {
  if (!refreshRequest) {
    refreshRequest = fetch("/api/auth/refresh", {
      method: "POST",
      credentials: "same-origin",
    })
      .then((response) => response.ok)
      .catch(() => false)
      .finally(() => {
        refreshRequest = null;
      });
  }
  return refreshRequest;
}

export async function authFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
  retry = true,
): Promise<Response> {
  const response = await fetch(input, { ...init, credentials: "same-origin" });
  if (response.status !== 401 || !retry) return response;
  if (!(await refreshSession())) return response;
  return fetch(input, { ...init, credentials: "same-origin" });
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?: string;
      error?: { code?: string; message?: string; details?: unknown };
    };
    const message = body.error?.message ?? body.detail;
    if (typeof message === "string") {
      if (message.includes("Desteklenmeyen dosya turu")) {
        return "Desteklenmeyen dosya türü. Yalnızca JPEG, PNG veya WebP kullanın.";
      }
      if (message.includes("bozuk veya gecersiz") || message.includes("okunamadi")) {
        return "Görüntü dosyası bozuk veya okunamıyor.";
      }
      if (message.includes("Yuklenen dosya bos")) {
        return "Yüklenen görüntü dosyası boş.";
      }
      if (message.includes("goruntu turuyle uyusmuyor")) {
        return "Dosya içeriği bildirilen görüntü türüyle uyuşmuyor.";
      }
      if (message.includes("Bulunan yuz sayisi: 0")) {
        return "Fotoğrafta yüz bulunamadı. Daha net bir fotoğraf deneyin.";
      }
      if (message.includes("Birden fazla belirgin yuz")) {
        return "Birden fazla belirgin yüz bulundu. Ana yüz güvenle seçilemedi.";
      }
      if (message.includes("Tam olarak bir yuz")) {
        return "Fotoğrafta birden fazla yüz bulundu. Tek kişilik bir fotoğraf seçin.";
      }
      return message;
    }
  } catch {
    // The generic message below also covers non-JSON proxy errors.
  }
  return "İşlem tamamlanamadı. Lütfen tekrar deneyin.";
}

async function getJson<T>(url: string): Promise<T> {
  const response = await authFetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/health");
}

export function getPersons(): Promise<Person[]> {
  return getJson<Person[]>(`/api/persons?refresh=${Date.now()}`);
}

export function getPersonFaceImages(personId: number): Promise<FaceImage[]> {
  return getJson<FaceImage[]>(`/api/persons/${personId}/face-images`);
}

export function getStatistics(): Promise<RecognitionStatistics> {
  return getJson<RecognitionStatistics>("/api/statistics");
}

export function getIdentities(): Promise<Identity[]> {
  return getJson<Identity[]>("/api/identities");
}

export function getIdentityHistory(
  faceId: string,
  limit = 8,
  offset = 0,
): Promise<FaceHistory> {
  return getJson<FaceHistory>(
    `/api/identities/${faceId}/history?limit=${limit}&offset=${offset}`,
  );
}

export function getProcess(processId: string): Promise<RecognitionProcess> {
  return getJson<RecognitionProcess>(`/api/processes/${processId}`);
}

async function sendJson<T>(url: string, method: "POST" | "PATCH", body: unknown): Promise<T> {
  const response = await authFetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<T>;
}

export function createPerson(input: PersonInput): Promise<Person> {
  return sendJson<Person>("/api/persons", "POST", input);
}

export function updatePerson(personId: number, input: PersonInput): Promise<Person> {
  return sendJson<Person>(`/api/persons/${personId}`, "PATCH", input);
}

export function enrollAnonymousIdentity(faceId: string, input: PersonInput): Promise<Person> {
  return sendJson<Person>(`/api/anonymous-identities/${faceId}/enroll`, "POST", input);
}

export function updateIdentity(faceId: string, input: PersonInput): Promise<Identity> {
  return sendJson<Identity>(`/api/identities/${faceId}`, "PATCH", input);
}

async function deleteResource(url: string): Promise<void> {
  const response = await authFetch(url, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}

export function deletePerson(personId: number): Promise<void> {
  return deleteResource(`/api/persons/${personId}`);
}

export function deleteFaceImage(personId: number, faceImageId: number): Promise<void> {
  return deleteResource(`/api/persons/${personId}/face-images/${faceImageId}`);
}

export function deleteIdentity(faceId: string): Promise<void> {
  return deleteResource(`/api/identities/${faceId}`);
}

export async function uploadPersonFaceImage(personId: number, file: File): Promise<FaceImage> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await authFetch(`/api/persons/${personId}/face-images`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<FaceImage>;
}

export async function identifyFace(
  file: File,
  signal?: AbortSignal,
  source: "photo_upload" | "live_video_frame" = "photo_upload",
): Promise<IdentifyResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await authFetch("/api/faces/identify", {
    method: "POST",
    body: formData,
    signal,
    headers: source === "live_video_frame"
      ? { "X-Recognition-Source": "live_video_frame" }
      : undefined,
  });
  if (!response.ok) {
    const processId = response.headers.get("X-Process-ID");
    const message = await readError(response);
    throw new Error(processId ? `${message} İşlem ID: ${processId}` : message);
  }
  return response.json() as Promise<IdentifyResponse>;
}

export function getPhotoHistory(limit = 12, offset = 0): Promise<PhotoHistoryList> {
  return getJson<PhotoHistoryList>(`/api/photos?limit=${limit}&offset=${offset}`);
}

export function searchAppearances(params: AppearanceSearchParams): Promise<AppearanceSearchResponse> {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  });
  return getJson<AppearanceSearchResponse>(`/api/search/appearances?${query.toString()}`);
}

export async function uploadVideo(file: File): Promise<VideoJob> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await authFetch("/api/videos", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<VideoJob>;
}

export async function uploadLiveVideo(
  file: File,
  manifest: LiveVideoManifest,
): Promise<VideoJob> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("manifest", JSON.stringify(manifest));
  const response = await authFetch("/api/videos/live-recordings", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<VideoJob>;
}

export function getVideoJob(processId: string): Promise<VideoJob> {
  return getJson<VideoJob>(`/api/videos/${processId}`);
}

export function getVideoJobs(limit = 20, offset = 0): Promise<VideoJobList> {
  return getJson<VideoJobList>(`/api/videos?limit=${limit}&offset=${offset}`);
}

export function getVideoResult(processId: string): Promise<VideoResult> {
  return getJson<VideoResult>(`/api/videos/${processId}/result`);
}

export function getVideoFaceHistory(faceId: string): Promise<VideoFaceHistory> {
  return getJson<VideoFaceHistory>(`/api/videos/faces/${faceId}/history?limit=20&offset=0`);
}

export function deleteVideo(processId: string): Promise<void> {
  return deleteResource(`/api/videos/${processId}`);
}

export async function getCurrentUser(): Promise<AppUser> {
  const response = await authFetch("/api/auth/me", { cache: "no-store" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<AppUser>;
}

export async function login(input: LoginInput): Promise<AuthResponse> {
  const response = await authFetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  }, false);
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<AuthResponse>;
}

export async function register(input: RegisterInput): Promise<AuthResponse> {
  const response = await authFetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  }, false);
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<AuthResponse>;
}

export async function logout(): Promise<void> {
  await authFetch("/api/auth/logout", { method: "POST" }, false);
}

export function getUsers(): Promise<AppUser[]> {
  return getJson<AppUser[]>("/api/admin/users");
}

export function setUserActive(userId: string, isActive: boolean): Promise<AppUser> {
  return sendJson<AppUser>(`/api/admin/users/${userId}`, "PATCH", { is_active: isActive });
}
