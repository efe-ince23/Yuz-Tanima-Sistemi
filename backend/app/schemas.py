from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: str
    full_name: str
    role: Literal["admin", "user"]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=150)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 2:
            raise ValueError("Ad soyad bos birakilamaz.")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not any(char.islower() for char in value):
            raise ValueError("Parola en az bir kucuk harf icermelidir.")
        if not any(char.isupper() for char in value):
            raise ValueError("Parola en az bir buyuk harf icermelidir.")
        if not any(char.isdigit() for char in value):
            raise ValueError("Parola en az bir rakam icermelidir.")
        return value


class LoginRequest(BaseModel):
    identifier: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    user: UserResponse
    access_expires_at: datetime


class AdminUserUpdate(BaseModel):
    is_active: bool


class PersonCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Bu alan bos birakilamaz.")
        return value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ApiErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail
    process_id: Optional[UUID] = None
    timestamp: datetime


class RootResponse(BaseModel):
    project: str
    status: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["connected"]
    vector_store: Literal["qdrant", "postgres_fallback"]
    qdrant_points: Optional[int]
    object_storage: Literal["minio", "local"]
    storage_objects: Optional[int]
    execution_providers: List[str]


class PersonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    face_id: UUID
    first_name: str
    last_name: str
    description: Optional[str]
    face_image_count: int
    sample_image_url: Optional[str]
    created_at: datetime
    updated_at: datetime


class PersonUpdate(BaseModel):
    first_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("Bu alan null olamaz.")
        value = value.strip()
        if not value:
            raise ValueError("Bu alan bos birakilamaz.")
        return value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def require_a_field(self):
        if not self.model_fields_set:
            raise ValueError("En az bir alan gonderilmelidir.")
        return self


class FaceImageResponse(BaseModel):
    id: int
    person_id: int
    image_url: str
    detection_confidence: float
    created_at: datetime


class FaceImageUploadResponse(FaceImageResponse):
    execution_providers: List[str]


class BoundingBoxResponse(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class DetectedFaceResponse(BaseModel):
    face_index: int
    bounding_box: BoundingBoxResponse
    confidence: float


class FaceDetectResponse(BaseModel):
    process_id: UUID
    status: Literal["faces_detected", "no_face"]
    face_found: bool
    image_width: int
    image_height: int
    face_count: int
    execution_providers: List[str]
    faces: List[DetectedFaceResponse]


class FaceCompareResponse(BaseModel):
    process_id: UUID
    same_person: bool
    similarity: float
    threshold: float
    execution_providers: List[str]


class IdentifiedPersonResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    description: Optional[str]


class IdentifiedFaceResponse(BaseModel):
    face_index: int
    face_id: UUID
    status: Literal["known", "anonymous", "new_anonymous"]
    recognized: bool
    similarity: Optional[float]
    person: Optional[IdentifiedPersonResponse]
    matched_image_url: Optional[str]
    detection_confidence: float
    bounding_box: BoundingBoxResponse


class FaceIdentifyResponse(BaseModel):
    process_id: UUID
    status: Literal["recognized", "unrecognized", "no_face"]
    recognized: bool
    similarity: Optional[float]
    threshold: float
    person: Optional[IdentifiedPersonResponse]
    face_id: Optional[UUID]
    matched_image_url: Optional[str]
    execution_providers: List[str]
    detected_face_count: int
    ignored_face_count: int
    faces: List[IdentifiedFaceResponse]


class PhotoHistoryItemResponse(BaseModel):
    process_id: UUID
    status: str
    face_count: int
    original_filename: Optional[str]
    owner_username: Optional[str]
    owner_full_name: Optional[str]
    image_url: str
    image_width: Optional[int]
    image_height: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]
    result: Optional[FaceIdentifyResponse]


class PhotoHistoryListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[PhotoHistoryItemResponse]


class AppearanceSearchIntervalResponse(BaseModel):
    start_ms: int
    end_ms: int


class AppearanceSearchItemResponse(BaseModel):
    source_type: Literal["photo", "video"]
    process_id: UUID
    face_id: UUID
    status: Literal["known", "anonymous", "new_anonymous"]
    person_id: Optional[int]
    first_name: Optional[str]
    last_name: Optional[str]
    metadata: Optional[Dict[str, Any]]
    owner_user_id: Optional[UUID]
    owner_username: Optional[str]
    owner_full_name: Optional[str]
    occurred_at: datetime
    confidence: Optional[float]
    original_filename: Optional[str]
    preview_url: Optional[str]
    content_url: str
    observation_count: int
    first_seen_ms: Optional[int]
    last_seen_ms: Optional[int]
    intervals: List[AppearanceSearchIntervalResponse]


class AppearanceSearchResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[AppearanceSearchItemResponse]


class PublicRecognizedFaceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    face_id: UUID = Field(alias="faceId")
    status: Literal["known", "anonymous", "new_anonymous"]
    name: Optional[str]
    metadata: Optional[Dict[str, Any]]
    bounding_box: BoundingBoxResponse = Field(alias="boundingBox")
    confidence: Optional[float]


class PublicFaceRecognitionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    process_id: UUID = Field(alias="processId")
    status: Literal["recognized", "unrecognized", "no_face"]
    detected_face_count: int = Field(alias="detectedFaceCount")
    faces: List[PublicRecognizedFaceResponse]


class IdentityResponse(BaseModel):
    face_id: UUID
    status: Literal["known", "anonymous"]
    person_id: Optional[int]
    first_name: Optional[str]
    last_name: Optional[str]
    description: Optional[str]
    sample_count: int
    reference_image_count: int
    observation_count: int
    photo_observation_count: int
    photo_last_seen_at: Optional[datetime]
    video_observation_count: int
    video_last_seen_at: Optional[datetime]
    sample_image_urls: List[str]
    created_at: datetime
    updated_at: datetime
    last_seen_at: Optional[datetime]


class FaceHistoryEntryResponse(BaseModel):
    event_id: int
    process_id: Optional[UUID]
    operation_type: Optional[Literal["detect", "compare", "identify"]]
    timestamp: datetime
    status: Optional[str]
    recognized: bool
    similarity: Optional[float]
    threshold: float


class FaceHistoryResponse(BaseModel):
    face_id: UUID
    total: int
    limit: int
    offset: int
    first_seen_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    appearances: List[FaceHistoryEntryResponse]


class RecognitionProcessEventResponse(BaseModel):
    id: int
    face_id: Optional[UUID]
    face_status: Optional[str]
    recognized: bool
    person_id: Optional[int]
    similarity: Optional[float]
    threshold: float
    created_at: datetime


class RecognitionProcessResponse(BaseModel):
    process_id: UUID
    operation_type: Literal["detect", "compare", "identify", "video_recognize"]
    status: str
    http_status: Optional[int]
    face_count: int
    task_detail: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error_detail: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    events: List[RecognitionProcessEventResponse]


class RecognitionStatisticsResponse(BaseModel):
    total_operations: int
    recognized_count: int
    unrecognized_count: int
    success_rate: float
    latest_event_at: Optional[datetime]


class LiveVideoBoundingBoxInput(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding box coordinates are out of order.")
        return self


class LiveVideoObservationInput(BaseModel):
    timestamp_ms: int = Field(ge=0)
    face_id: UUID
    status: Literal["known", "anonymous", "new_anonymous"]
    name: Optional[str] = Field(default=None, max_length=250)
    metadata: Optional[Dict[str, Any]] = None
    bounding_box: LiveVideoBoundingBoxInput
    detection_confidence: float = Field(ge=0.0, le=1.0)
    recognition_confidence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    matched_image_url: Optional[str] = Field(default=None, max_length=1000)


class LiveVideoManifestInput(BaseModel):
    version: Literal[1]
    duration_ms: int = Field(ge=1)
    analysis_count: int = Field(ge=0, le=10000)
    first_analysis_ms: Optional[int] = Field(default=None, ge=0)
    last_analysis_ms: Optional[int] = Field(default=None, ge=0)
    observations: List[LiveVideoObservationInput] = Field(max_length=5000)

    @model_validator(mode="after")
    def validate_timestamps(self):
        timestamps = [
            item.timestamp_ms
            for item in self.observations
        ] + [
            value
            for value in (self.first_analysis_ms, self.last_analysis_ms)
            if value is not None
        ]
        if any(timestamp > self.duration_ms + 2000 for timestamp in timestamps):
            raise ValueError("Observation timestamp exceeds recording duration.")
        if (
            self.first_analysis_ms is not None
            and self.last_analysis_ms is not None
            and self.first_analysis_ms > self.last_analysis_ms
        ):
            raise ValueError("Analysis timestamps are out of order.")
        return self


class VideoJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    process_id: UUID
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    original_filename: str
    object_path: str
    content_type: str
    file_size_bytes: int
    duration_seconds: Optional[float]
    source_fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    frame_count: Optional[int]
    sampled_frame_count: int
    processed_frame_count: int
    progress_percent: float
    detected_face_count: int
    unique_face_count: int
    error_code: Optional[str]
    error_detail: Optional[str]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class VideoJobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[VideoJobResponse]


class VideoBoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class VideoObservationResponse(BaseModel):
    frame_number: int
    timestamp_ms: int
    bounding_box: VideoBoundingBoxResponse
    detection_confidence: float
    recognition_confidence: Optional[float]


class VideoAppearanceSegmentResponse(BaseModel):
    start_ms: int
    end_ms: int
    start_frame: int
    end_frame: int
    observation_count: int
    max_recognition_confidence: Optional[float]
    average_recognition_confidence: Optional[float]


class VideoFaceHistoryItemResponse(BaseModel):
    process_id: UUID
    original_filename: str
    created_at: datetime
    duration_seconds: Optional[float]
    first_seen_ms: int
    last_seen_ms: int
    observation_count: int
    appearances: List[VideoAppearanceSegmentResponse]


class VideoFaceHistoryResponse(BaseModel):
    face_id: UUID
    total: int
    limit: int
    offset: int
    items: List[VideoFaceHistoryItemResponse]


class VideoTrackResultResponse(BaseModel):
    track_id: int
    face_id: UUID
    status: Literal["known", "anonymous", "new_anonymous"]
    name: Optional[str]
    metadata: Optional[Dict[str, Any]]
    first_seen_ms: int
    last_seen_ms: int
    observation_count: int
    best_detection_confidence: Optional[float]
    best_recognition_confidence: Optional[float]
    best_frame_number: Optional[int]
    best_image_url: Optional[str]
    appearances: List[VideoAppearanceSegmentResponse]
    observations: List[VideoObservationResponse]


class VideoResultResponse(BaseModel):
    process_id: UUID
    status: Literal["completed"]
    video_url: str
    duration_seconds: Optional[float]
    sampled_frame_count: int
    detected_face_count: int
    unique_face_count: int
    tracks: List[VideoTrackResultResponse]
