import logging
import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.face_detector import (
    active_execution_providers,
    ANONYMOUS_MATCH_THRESHOLD,
    available_execution_providers,
    compare_embeddings,
    detect_faces,
    extract_all_faces_data,
    MATCH_THRESHOLD,
)
from app.api_errors import ApiHTTPException, DEFAULT_ERROR_CODES, api_error
from app.database import SessionLocal, database_is_ready, get_database_session
from app.auth import current_user_id_from_request, ensure_initial_admin, get_current_user, require_admin
from app.face_identification import (
    create_anonymous_identity,
    find_closest_anonymous_face,
    find_closest_face,
    lock_anonymous_matching,
    record_anonymous_observation,
)
from app.face_storage import (
    delete_face_image,
    ensure_data_directory,
    ensure_object_storage,
    image_url,
    migrate_legacy_objects,
    read_face_image,
    save_anonymous_face,
    save_recognition_photo,
    storage_backend_name,
    storage_is_ready,
    storage_object_count,
)
from app.image_upload import (
    extract_face_data_or_error,
    read_uploaded_image,
)
from app.models import (
    AnonymousFaceEmbedding,
    AnonymousIdentity,
    FaceImage,
    Person,
    RecognitionProcess,
    User,
)
from app.routers.persons import router as persons_router
from app.routers.anonymous_identities import router as anonymous_identities_router
from app.routers.identities import router as identities_router
from app.routers.processes import router as processes_router
from app.routers.public_faces import router as public_faces_router
from app.routers.public_processes import router as public_processes_router
from app.routers.statistics import router as statistics_router
from app.routers.videos import router as videos_router
from app.routers.photos import router as photos_router
from app.routers.search import router as search_router
from app.routers.auth import admin_router as admin_users_router, router as auth_router
from app.process_tracking import (
    TRACKED_FACE_PATHS,
    begin_process,
    complete_process,
    complete_process_if_pending,
    fail_process,
)
from app.recognition_events import record_recognition_event
from app.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    FaceCompareResponse,
    FaceDetectResponse,
    FaceIdentifyResponse,
    IdentifiedFaceResponse,
    IdentifiedPersonResponse,
    HealthResponse,
    PublicFaceRecognitionResponse,
    PublicRecognizedFaceResponse,
    RootResponse,
)
from app.vector_store import (
    qdrant_is_ready,
    qdrant_point_count,
    synchronize_all_safely,
    synchronize_face_id_safely,
)
from app.video_config import get_video_settings
from app.video_worker import start_video_workers, stop_video_workers


ensure_data_directory()
logger = logging.getLogger(__name__)
COMMON_ERROR_RESPONSES = {
    400: {"model": ApiErrorResponse, "description": "Gecersiz istek"},
    404: {"model": ApiErrorResponse, "description": "Kaynak bulunamadi"},
    409: {"model": ApiErrorResponse, "description": "Kaynak cakismasi"},
    413: {"model": ApiErrorResponse, "description": "Dosya cok buyuk"},
    415: {"model": ApiErrorResponse, "description": "Desteklenmeyen medya turu"},
    422: {"model": ApiErrorResponse, "description": "Dogrulama hatasi"},
    500: {"model": ApiErrorResponse, "description": "Sunucu hatasi"},
    503: {"model": ApiErrorResponse, "description": "Servis kullanilamiyor"},
}
app = FastAPI(
    title="Yuz Tanima Sistemi",
    version="0.1.0",
    responses=COMMON_ERROR_RESPONSES,
)
app.include_router(auth_router)
app.include_router(admin_users_router)
app.include_router(public_faces_router, dependencies=[Depends(get_current_user)])
app.include_router(public_processes_router, dependencies=[Depends(get_current_user)])
app.include_router(anonymous_identities_router, include_in_schema=False, dependencies=[Depends(get_current_user)])
app.include_router(identities_router, include_in_schema=False, dependencies=[Depends(get_current_user)])
app.include_router(persons_router, include_in_schema=False, dependencies=[Depends(require_admin)])
app.include_router(processes_router, include_in_schema=False, dependencies=[Depends(get_current_user)])
app.include_router(statistics_router, include_in_schema=False, dependencies=[Depends(get_current_user)])
app.include_router(videos_router, dependencies=[Depends(get_current_user)])
app.include_router(photos_router, dependencies=[Depends(get_current_user)])
app.include_router(search_router, dependencies=[Depends(get_current_user)])
@app.on_event("startup")
def initialize_services() -> None:
    ensure_object_storage()
    providers = active_execution_providers()
    logger.info("Face engine initialized with providers: %s", providers)
    video_settings = get_video_settings()
    logger.info(
        "Video rules initialized: max_bytes=%s max_seconds=%s sample_fps=%s concurrency=%s",
        video_settings.max_size_bytes,
        video_settings.max_duration_seconds,
        video_settings.sample_fps,
        video_settings.processing_concurrency,
    )
    with SessionLocal() as session:
        admin = ensure_initial_admin(session)
        logger.info("Authentication initialized with admin user: %s", admin.username)
        migrate_legacy = os.getenv(
            "MIGRATE_LEGACY_OBJECTS_ON_STARTUP", "false"
        ).strip().lower() in {"1", "true", "yes"}
        if migrate_legacy:
            image_paths = list(session.scalars(select(FaceImage.image_path)).all())
            image_paths.extend(
                session.scalars(
                    select(AnonymousFaceEmbedding.image_path).where(
                        AnonymousFaceEmbedding.image_path.is_not(None)
                    )
                ).all()
            )
            migration = migrate_legacy_objects(image_paths)
            logger.info(
                "Object storage synchronization completed: migrated=%s present=%s missing=%s",
                migration.migrated,
                migration.already_present,
                migration.missing_local,
            )
        else:
            logger.info("Legacy object migration skipped on startup")
        synchronize_all_safely(session)
    start_video_workers()


@app.on_event("shutdown")
def shutdown_services() -> None:
    stop_video_workers()


@app.get("/media/{relative_path:path}", include_in_schema=False)
def get_media(
    relative_path: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> Response:
    parts = relative_path.strip("/").split("/")
    permitted = user.role == "admin"
    if not permitted and len(parts) >= 2 and parts[0] == "persons":
        try:
            person_id = int(parts[1])
        except ValueError:
            person_id = -1
        person = session.get(Person, person_id)
        permitted = person is not None and (
            person.is_global or person.owner_user_id == user.id
        )
    elif not permitted and len(parts) >= 2 and parts[0] == "anonymous":
        try:
            face_id = UUID(parts[1])
        except ValueError:
            face_id = None
        anonymous = (
            session.scalar(
                select(AnonymousIdentity).where(
                    AnonymousIdentity.face_id == face_id,
                    AnonymousIdentity.owner_user_id == user.id,
                )
            )
            if face_id is not None
            else None
        )
        permitted = anonymous is not None
    elif not permitted and len(parts) >= 2 and parts[0] == "videos":
        try:
            process_id = UUID(parts[1])
        except ValueError:
            process_id = None
        permitted = process_id is not None and session.scalar(
            select(RecognitionProcess.process_id).where(
                RecognitionProcess.process_id == process_id,
                RecognitionProcess.owner_user_id == user.id,
            )
        ) is not None
    if not permitted:
        raise api_error(404, "FACE_IMAGE_NOT_FOUND", "Yuz fotografi bulunamadi.")
    try:
        content, content_type = read_face_image(relative_path)
    except (FileNotFoundError, ValueError) as error:
        raise api_error(404, "FACE_IMAGE_NOT_FOUND", "Yuz fotografi bulunamadi.") from error
    except OSError as error:
        raise api_error(
            503,
            "OBJECT_STORAGE_UNAVAILABLE",
            "Fotograf depolama servisine erisilemiyor.",
        ) from error
    return Response(
        content=content,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.middleware("http")
async def track_face_process(request: Request, call_next):
    operation_type = TRACKED_FACE_PATHS.get(request.url.path)
    if operation_type is None:
        return await call_next(request)

    process_id = uuid4()
    request.state.process_id = process_id
    request.state.process_error = None
    request.state.process_persisted = begin_process(
        process_id,
        operation_type,
        owner_user_id=current_user_id_from_request(request),
    )

    try:
        response = await call_next(request)
    except Exception as error:
        fail_process(
            process_id,
            operation_type,
            500,
            str(error) or "Beklenmeyen sunucu hatasi.",
        )
        raise

    response.headers["X-Process-ID"] = str(process_id)
    if response.status_code >= 400:
        fail_process(
            process_id,
            operation_type,
            response.status_code,
            request.state.process_error or f"HTTP {response.status_code}",
        )
    else:
        complete_process_if_pending(process_id, operation_type, response.status_code)
    return response


@app.exception_handler(HTTPException)
async def tracked_http_exception(request: Request, error: HTTPException) -> JSONResponse:
    process_id = getattr(request.state, "process_id", None)
    if process_id is not None:
        request.state.process_error = str(error.detail)
    if isinstance(error, ApiHTTPException):
        code = error.code
        message = error.message
        details = error.details
    else:
        code = DEFAULT_ERROR_CODES.get(error.status_code, "HTTP_ERROR")
        message = error.detail if isinstance(error.detail, str) else "Istek tamamlanamadi."
        details = None if isinstance(error.detail, str) else error.detail
    content = ApiErrorResponse(
        error=ApiErrorDetail(code=code, message=message, details=details),
        process_id=process_id,
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")
    return JSONResponse(
        status_code=error.status_code,
        content=content,
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
async def tracked_validation_exception(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    process_id = getattr(request.state, "process_id", None)
    request.state.process_error = "Istek dogrulanamadi."
    content = ApiErrorResponse(
        error=ApiErrorDetail(
            code="VALIDATION_ERROR",
            message="Istek dogrulanamadi.",
            details=error.errors(),
        ),
        process_id=process_id,
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")
    return JSONResponse(status_code=422, content=jsonable_encoder(content))


@app.exception_handler(Exception)
async def tracked_unexpected_exception(request: Request, error: Exception) -> JSONResponse:
    process_id = getattr(request.state, "process_id", None)
    if process_id is not None:
        request.state.process_error = "Beklenmeyen sunucu hatasi."
    logger.exception("Unhandled API error")
    content = ApiErrorResponse(
        error=ApiErrorDetail(
            code="INTERNAL_ERROR",
            message="Beklenmeyen sunucu hatasi.",
        ),
        process_id=process_id,
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")
    return JSONResponse(status_code=500, content=content)


@app.get("/", response_model=RootResponse, include_in_schema=False)
def root() -> RootResponse:
    return RootResponse(project="Yuz Tanima Sistemi", status="calisiyor")


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    if not database_is_ready():
        raise api_error(
            503,
            "DATABASE_UNAVAILABLE",
            "Database baglantisi kurulamadi.",
        )

    if not storage_is_ready():
        raise api_error(
            503,
            "OBJECT_STORAGE_UNAVAILABLE",
            "Fotograf depolama servisine erisilemiyor.",
        )

    qdrant_ready = qdrant_is_ready()
    return HealthResponse(
        status="ok",
        database="connected",
        vector_store="qdrant" if qdrant_ready else "postgres_fallback",
        qdrant_points=qdrant_point_count() if qdrant_ready else None,
        object_storage=storage_backend_name(),
        storage_objects=storage_object_count(),
        execution_providers=available_execution_providers(),
    )


@app.post(
    "/api/faces/detect",
    response_model=FaceDetectResponse,
    tags=["faces"],
    include_in_schema=False,
)
async def detect_face(
    request: Request,
    file: UploadFile = File(
        ...,
        description="JPEG, PNG veya WebP goruntu; en fazla 10 MB.",
    ),
    _user: User = Depends(get_current_user),
) -> FaceDetectResponse:
    process_id: UUID = request.state.process_id
    image = await read_uploaded_image(file, "file")

    faces = detect_faces(image)
    image_height, image_width = image.shape[:2]
    response = FaceDetectResponse(
        process_id=process_id,
        status="faces_detected" if faces else "no_face",
        face_found=bool(faces),
        image_width=image_width,
        image_height=image_height,
        face_count=len(faces),
        execution_providers=active_execution_providers(),
        faces=faces,
    )
    complete_process(
        process_id,
        operation_type="detect",
        status=response.status,
        http_status=200,
        face_count=response.face_count,
        faces=[
            {"face_id": None, "status": "detected", "face_index": index}
            for index in range(response.face_count)
        ],
        result=response.model_dump(mode="json"),
    )
    return response


@app.post(
    "/api/faces/compare",
    response_model=FaceCompareResponse,
    tags=["faces"],
    include_in_schema=False,
)
async def compare_faces(
    request: Request,
    image_a: UploadFile = File(
        ...,
        description="Ilk JPEG, PNG veya WebP yuz goruntusu; en fazla 10 MB.",
    ),
    image_b: UploadFile = File(
        ...,
        description="Ikinci JPEG, PNG veya WebP yuz goruntusu; en fazla 10 MB.",
    ),
    _user: User = Depends(get_current_user),
) -> FaceCompareResponse:
    process_id: UUID = request.state.process_id
    first_image = await read_uploaded_image(image_a, "image_a")
    second_image = await read_uploaded_image(image_b, "image_b")

    first_embedding, _ = extract_face_data_or_error(first_image, "image_a")
    second_embedding, _ = extract_face_data_or_error(second_image, "image_b")
    similarity = compare_embeddings(first_embedding, second_embedding)

    response = FaceCompareResponse(
        process_id=process_id,
        same_person=similarity >= MATCH_THRESHOLD,
        similarity=round(similarity, 4),
        threshold=MATCH_THRESHOLD,
        execution_providers=active_execution_providers(),
    )
    complete_process(
        process_id,
        operation_type="compare",
        status="completed",
        http_status=200,
        face_count=2,
        faces=[
            {"face_id": None, "status": "compared", "face_index": 0},
            {"face_id": None, "status": "compared", "face_index": 1},
        ],
        result=response.model_dump(mode="json"),
    )
    return response


@app.post(
    "/faces/recognize",
    response_model=PublicFaceRecognitionResponse,
    tags=["faces"],
    summary="Recognize faces",
)
@app.post(
    "/api/faces/identify",
    response_model=FaceIdentifyResponse,
    tags=["faces"],
    include_in_schema=False,
)
async def identify_face(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Tum yuzleri tanimlamak icin JPEG, PNG veya WebP goruntu; en fazla 10 MB.",
    ),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> object:
    process_id: UUID = request.state.process_id
    image = await read_uploaded_image(file, "file")
    if request.headers.get("X-Recognition-Source") != "live_video_frame":
        stored_photo_path = None
        try:
            stored_photo_path, stored_photo_size = save_recognition_photo(user.id, process_id, image)
            process = session.get(RecognitionProcess, process_id)
            if process is None:
                raise RuntimeError("Fotograf process kaydi bulunamadi.")
            process.source_image_path = stored_photo_path
            process.source_filename = (file.filename or "fotograf.jpg")[:255]
            process.source_content_type = "image/jpeg"
            process.source_file_size_bytes = stored_photo_size
            process.source_image_width = int(image.shape[1])
            process.source_image_height = int(image.shape[0])
            session.commit()
        except (OSError, RuntimeError, ValueError, SQLAlchemyError):
            session.rollback()
            if stored_photo_path is not None:
                try:
                    delete_face_image(stored_photo_path)
                except OSError:
                    logger.exception("Kaydedilemeyen tanima fotografi temizlenemedi: %s", process_id)
            logger.exception("Tanima fotografi gecmise eklenemedi: %s", process_id)
    detected_faces = extract_all_faces_data(image)
    providers = active_execution_providers()

    if not detected_faces:
        response = FaceIdentifyResponse(
            process_id=process_id,
            status="no_face",
            recognized=False,
            similarity=None,
            threshold=MATCH_THRESHOLD,
            person=None,
            face_id=None,
            matched_image_url=None,
            execution_providers=providers,
            detected_face_count=0,
            ignored_face_count=0,
            faces=[],
        )
        public_response = _public_recognition_response(response)
        result = (
            public_response.model_dump(mode="json", by_alias=True)
            if request.url.path == "/faces/recognize"
            else response.model_dump(mode="json")
        )
        complete_process(
            process_id,
            operation_type="identify",
            status="no_face",
            http_status=200,
            face_count=0,
            faces=[],
            result=result,
        )
        return public_response if request.url.path == "/faces/recognize" else response

    face_results = []
    stored_anonymous_images = []
    try:
        lock_anonymous_matching(session)
        for face_index, detected_face in enumerate(detected_faces):
            if detected_face.embedding is None:
                raise api_error(
                    500,
                    "FACE_EMBEDDING_FAILED",
                    "Yuz icin kimlik verisi olusturulamadi.",
                )

            embedding = detected_face.embedding.tolist()
            known_match = find_closest_face(session, embedding, owner_user_id=user.id)
            person = None
            matched_image = None
            similarity = None
            recognized = False

            if known_match is not None and known_match[2] >= MATCH_THRESHOLD:
                matched_person, face_image, similarity = known_match
                recognized = True
                face_status = "known"
                face_id = matched_person.face_id
                person = IdentifiedPersonResponse(
                    id=matched_person.id,
                    first_name=matched_person.first_name,
                    last_name=matched_person.last_name,
                    description=matched_person.description,
                )
                matched_image = (
                    image_url(face_image.image_path) if face_image is not None else None
                )
            else:
                anonymous_match = find_closest_anonymous_face(
                    session, embedding, owner_user_id=user.id
                )
                if (
                    anonymous_match is not None
                    and anonymous_match[1] >= ANONYMOUS_MATCH_THRESHOLD
                ):
                    anonymous_identity, similarity = anonymous_match
                    face_status = "anonymous"
                    face_id = anonymous_identity.face_id
                    anonymous_sample = record_anonymous_observation(
                        session,
                        anonymous_identity,
                        embedding,
                        detected_face.confidence,
                        similarity,
                    )
                else:
                    anonymous_identity, anonymous_sample = create_anonymous_identity(
                        session,
                        embedding,
                        detected_face.confidence,
                        owner_user_id=user.id,
                    )
                    face_status = "new_anonymous"
                    face_id = anonymous_identity.face_id

                if anonymous_sample is not None:
                    anonymous_sample.image_path = save_anonymous_face(
                        face_id,
                        image,
                        detected_face.bounding_box,
                    )
                    stored_anonymous_images.append(anonymous_sample.image_path)

            x1, y1, x2, y2 = detected_face.bounding_box
            face_result = IdentifiedFaceResponse(
                face_index=face_index,
                face_id=face_id,
                status=face_status,
                recognized=recognized,
                similarity=round(similarity, 4) if similarity is not None else None,
                person=person,
                matched_image_url=matched_image,
                detection_confidence=round(detected_face.confidence, 4),
                bounding_box={
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
            )
            face_results.append(face_result)
            record_recognition_event(
                session,
                process_id=(
                    process_id if request.state.process_persisted else None
                ),
                recognized=recognized,
                person_id=person.id if person is not None else None,
                face_id=face_id,
                face_status=face_status,
                similarity=similarity,
                threshold=(
                    MATCH_THRESHOLD
                    if face_status == "known"
                    else ANONYMOUS_MATCH_THRESHOLD
                ),
                commit=False,
            )
            session.flush()
        session.commit()
        synchronized_face_ids = {
            face.face_id
            for face in face_results
            if face.status in ("anonymous", "new_anonymous")
        }
        for synchronized_face_id in synchronized_face_ids:
            synchronize_face_id_safely(session, synchronized_face_id)
    except HTTPException:
        session.rollback()
        for stored_path in stored_anonymous_images:
            delete_face_image(stored_path)
        raise
    except (OSError, RuntimeError, ValueError) as error:
        session.rollback()
        for stored_path in stored_anonymous_images:
            delete_face_image(stored_path)
        raise api_error(
            500,
            "ANONYMOUS_FACE_STORAGE_FAILED",
            "Anonim yuz goruntusu saklanamadi.",
        ) from error
    except SQLAlchemyError as error:
        session.rollback()
        for stored_path in stored_anonymous_images:
            delete_face_image(stored_path)
        raise api_error(
            500,
            "FACE_DATABASE_WRITE_FAILED",
            "Yuz kimlikleri veritabanina kaydedilemedi.",
        ) from error

    recognized_faces = [face for face in face_results if face.recognized]
    representative = max(
        recognized_faces or face_results,
        key=lambda face: face.similarity if face.similarity is not None else -1.0,
    )
    any_recognized = bool(recognized_faces)
    response = FaceIdentifyResponse(
        process_id=process_id,
        status="recognized" if any_recognized else "unrecognized",
        recognized=any_recognized,
        similarity=representative.similarity,
        threshold=MATCH_THRESHOLD,
        person=representative.person if any_recognized else None,
        face_id=representative.face_id,
        matched_image_url=(
            representative.matched_image_url if any_recognized else None
        ),
        execution_providers=providers,
        detected_face_count=len(face_results),
        ignored_face_count=0,
        faces=face_results,
    )
    public_response = _public_recognition_response(response)
    result = (
        public_response.model_dump(mode="json", by_alias=True)
        if request.url.path == "/faces/recognize"
        else response.model_dump(mode="json")
    )
    complete_process(
        process_id,
        operation_type="identify",
        status=response.status,
        http_status=200,
        face_count=response.detected_face_count,
        faces=[
            {
                "face_id": str(face.face_id),
                "status": face.status,
                "face_index": face.face_index,
            }
            for face in response.faces
        ],
        result=result,
    )
    return public_response if request.url.path == "/faces/recognize" else response


def _public_recognition_response(
    response: FaceIdentifyResponse,
) -> PublicFaceRecognitionResponse:
    return PublicFaceRecognitionResponse(
        process_id=response.process_id,
        status=response.status,
        detected_face_count=response.detected_face_count,
        faces=[
            PublicRecognizedFaceResponse(
                face_id=face.face_id,
                status=face.status,
                name=(
                    f"{face.person.first_name} {face.person.last_name}"
                    if face.person is not None
                    else None
                ),
                metadata=(
                    {"description": face.person.description}
                    if face.person is not None and face.person.description is not None
                    else {} if face.person is not None else None
                ),
                bounding_box=face.bounding_box,
                confidence=face.similarity,
            )
            for face in response.faces
        ],
    )
