export interface Person {
  id: number;
  face_id: string;
  first_name: string;
  last_name: string;
  description: string | null;
  face_image_count: number;
  sample_image_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface PersonInput {
  first_name: string;
  last_name: string;
  description: string | null;
}

export interface FaceImage {
  id: number;
  person_id: number;
  image_url: string;
  detection_confidence: number;
  created_at: string;
}

export interface HealthResponse {
  status: string;
  database: string;
  execution_providers: string[];
}

export interface IdentifiedPerson {
  id: number;
  first_name: string;
  last_name: string;
  description: string | null;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface IdentifiedFace {
  face_index: number;
  face_id: string;
  status: "known" | "anonymous" | "new_anonymous";
  recognized: boolean;
  similarity: number | null;
  person: IdentifiedPerson | null;
  matched_image_url: string | null;
  detection_confidence: number;
  bounding_box: BoundingBox;
}

export interface IdentifyResponse {
  process_id: string;
  status: "recognized" | "unrecognized" | "no_face";
  recognized: boolean;
  similarity: number | null;
  threshold: number;
  person: IdentifiedPerson | null;
  face_id: string | null;
  matched_image_url: string | null;
  execution_providers: string[];
  detected_face_count: number;
  ignored_face_count: number;
  faces: IdentifiedFace[];
}

export interface PhotoHistoryItem {
  process_id: string;
  status: string;
  face_count: number;
  original_filename: string | null;
  owner_username: string | null;
  owner_full_name: string | null;
  image_url: string;
  image_width: number | null;
  image_height: number | null;
  created_at: string;
  completed_at: string | null;
  result: IdentifyResponse | null;
}

export interface PhotoHistoryList {
  total: number;
  limit: number;
  offset: number;
  items: PhotoHistoryItem[];
}

export interface AppearanceSearchInterval {
  start_ms: number;
  end_ms: number;
}

export interface AppearanceSearchItem {
  source_type: "photo" | "video";
  process_id: string;
  face_id: string;
  status: "known" | "anonymous" | "new_anonymous";
  person_id: number | null;
  first_name: string | null;
  last_name: string | null;
  metadata: { description?: string | null } | null;
  owner_user_id: string | null;
  owner_username: string | null;
  owner_full_name: string | null;
  occurred_at: string;
  confidence: number | null;
  original_filename: string | null;
  preview_url: string | null;
  content_url: string;
  observation_count: number;
  first_seen_ms: number | null;
  last_seen_ms: number | null;
  intervals: AppearanceSearchInterval[];
}

export interface AppearanceSearchResponse {
  total: number;
  limit: number;
  offset: number;
  items: AppearanceSearchItem[];
}

export interface AppearanceSearchParams {
  q?: string;
  face_id?: string;
  identity_status?: "known" | "anonymous";
  source_type?: "all" | "photo" | "video";
  date_from?: string;
  date_to?: string;
  min_confidence?: number;
  max_confidence?: number;
  owner_user_id?: string;
  sort?: "newest" | "oldest" | "confidence";
  limit?: number;
  offset?: number;
}

export interface Identity {
  face_id: string;
  status: "known" | "anonymous";
  person_id: number | null;
  first_name: string | null;
  last_name: string | null;
  description: string | null;
  sample_count: number;
  reference_image_count: number;
  observation_count: number;
  photo_observation_count: number;
  photo_last_seen_at: string | null;
  video_observation_count: number;
  video_last_seen_at: string | null;
  sample_image_urls: string[];
  created_at: string;
  updated_at: string;
  last_seen_at: string | null;
}

export interface FaceHistoryEntry {
  event_id: number;
  process_id: string | null;
  operation_type: "detect" | "compare" | "identify" | null;
  timestamp: string;
  status: string | null;
  recognized: boolean;
  similarity: number | null;
  threshold: number;
}

export interface FaceHistory {
  face_id: string;
  total: number;
  limit: number;
  offset: number;
  first_seen_at: string | null;
  last_seen_at: string | null;
  appearances: FaceHistoryEntry[];
}

export interface ProcessEvent {
  id: number;
  face_id: string | null;
  face_status: string | null;
  recognized: boolean;
  person_id: number | null;
  similarity: number | null;
  threshold: number;
  created_at: string;
}

export interface RecognitionProcess {
  process_id: string;
  operation_type: "detect" | "compare" | "identify";
  status: string;
  http_status: number | null;
  face_count: number;
  task_detail: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  error_detail: string | null;
  created_at: string;
  completed_at: string | null;
  events: ProcessEvent[];
}

export interface RecognitionStatistics {
  total_operations: number;
  recognized_count: number;
  unrecognized_count: number;
  success_rate: number;
  latest_event_at: string | null;
}

export type VideoJobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";

export interface VideoJob {
  process_id: string;
  status: VideoJobStatus;
  original_filename: string;
  object_path: string;
  content_type: string;
  file_size_bytes: number;
  duration_seconds: number | null;
  source_fps: number | null;
  width: number | null;
  height: number | null;
  frame_count: number | null;
  sampled_frame_count: number;
  processed_frame_count: number;
  progress_percent: number;
  detected_face_count: number;
  unique_face_count: number;
  error_code: string | null;
  error_detail: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface LiveVideoObservation {
  timestamp_ms: number;
  face_id: string;
  status: "known" | "anonymous" | "new_anonymous";
  name: string | null;
  metadata: { description?: string | null } | null;
  bounding_box: VideoBoundingBox;
  detection_confidence: number;
  recognition_confidence: number | null;
  matched_image_url: string | null;
}

export interface LiveVideoManifest {
  version: 1;
  duration_ms: number;
  analysis_count: number;
  first_analysis_ms: number | null;
  last_analysis_ms: number | null;
  observations: LiveVideoObservation[];
}

export interface VideoJobList {
  total: number;
  limit: number;
  offset: number;
  items: VideoJob[];
}

export interface VideoBoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface VideoObservation {
  frame_number: number;
  timestamp_ms: number;
  bounding_box: VideoBoundingBox;
  detection_confidence: number;
  recognition_confidence: number | null;
}

export interface VideoAppearance {
  start_ms: number;
  end_ms: number;
  start_frame: number;
  end_frame: number;
  observation_count: number;
  max_recognition_confidence: number | null;
  average_recognition_confidence: number | null;
}

export interface VideoTrackResult {
  track_id: number;
  face_id: string;
  status: "known" | "anonymous" | "new_anonymous";
  name: string | null;
  metadata: { description?: string | null } | null;
  first_seen_ms: number;
  last_seen_ms: number;
  observation_count: number;
  best_detection_confidence: number | null;
  best_recognition_confidence: number | null;
  best_frame_number: number | null;
  best_image_url: string | null;
  appearances: VideoAppearance[];
  observations: VideoObservation[];
}

export interface VideoResult {
  process_id: string;
  status: "completed";
  video_url: string;
  duration_seconds: number | null;
  sampled_frame_count: number;
  detected_face_count: number;
  unique_face_count: number;
  tracks: VideoTrackResult[];
}

export interface VideoFaceHistoryItem {
  process_id: string;
  original_filename: string;
  created_at: string;
  duration_seconds: number | null;
  first_seen_ms: number;
  last_seen_ms: number;
  observation_count: number;
  appearances: VideoAppearance[];
}

export interface VideoFaceHistory {
  face_id: string;
  total: number;
  limit: number;
  offset: number;
  items: VideoFaceHistoryItem[];
}

export type UserRole = "admin" | "user";

export interface AppUser {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_login_at: string | null;
}

export interface AuthResponse {
  user: AppUser;
  access_expires_at: string;
}

export interface LoginInput {
  identifier: string;
  password: string;
}

export interface RegisterInput {
  username: string;
  email: string;
  full_name: string;
  password: string;
}
