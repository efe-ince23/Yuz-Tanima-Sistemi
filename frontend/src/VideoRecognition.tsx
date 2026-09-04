import {
  AlertCircle,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock3,
  FileVideo2,
  Fingerprint,
  History,
  LoaderCircle,
  RefreshCw,
  ScanFace,
  Trash2,
  Upload,
  Users,
} from "lucide-react";
import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  deleteVideo,
  getVideoFaceHistory,
  getVideoJob,
  getVideoJobs,
  getVideoResult,
  uploadLiveVideo,
  uploadVideo,
} from "./api";
import type {
  VideoBoundingBox,
  VideoAppearance,
  VideoFaceHistory,
  VideoJob,
  LiveVideoManifest,
  VideoResult,
  VideoTrackResult,
} from "./types";
import LiveVideoRecognition from "./LiveVideoRecognition";

const MAX_VIDEO_SIZE = 200 * 1024 * 1024;
const POLL_INTERVAL_MS = 1000;
const OBSERVATION_EDGE_TOLERANCE_MS = 90;
const MAX_OBSERVATION_INTERPOLATION_MS = 2500;
const HISTORY_PAGE_SIZE = 20;

interface VideoViewport {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface ActiveVideoFace {
  track: VideoTrackResult;
  boundingBox: VideoBoundingBox;
}

interface VideoAppearanceEntry {
  track: VideoTrackResult;
  appearance: VideoAppearance;
}

function interpolateValue(first: number, second: number, amount: number): number {
  return first + (second - first) * amount;
}

function interpolateBox(
  first: VideoBoundingBox,
  second: VideoBoundingBox,
  amount: number,
): VideoBoundingBox {
  return {
    x1: interpolateValue(first.x1, second.x1, amount),
    y1: interpolateValue(first.y1, second.y1, amount),
    x2: interpolateValue(first.x2, second.x2, amount),
    y2: interpolateValue(first.y2, second.y2, amount),
  };
}

function boxAtTime(track: VideoTrackResult, timestampMs: number): VideoBoundingBox | null {
  if (!track.observations.length) return null;
  const activeAppearance = track.appearances.length
    ? track.appearances.find(
      (appearance) => timestampMs >= appearance.start_ms - OBSERVATION_EDGE_TOLERANCE_MS
        && timestampMs <= appearance.end_ms + OBSERVATION_EDGE_TOLERANCE_MS,
    )
    : null;
  if (track.appearances.length && !activeAppearance) return null;

  const observations = activeAppearance
    ? track.observations.filter(
      (observation) => observation.timestamp_ms >= activeAppearance.start_ms
        && observation.timestamp_ms <= activeAppearance.end_ms,
    )
    : track.observations;
  if (!observations.length) return null;

  const first = observations[0];
  const last = observations[observations.length - 1];
  if (!activeAppearance && (
    timestampMs < first.timestamp_ms - OBSERVATION_EDGE_TOLERANCE_MS
    || timestampMs > last.timestamp_ms + OBSERVATION_EDGE_TOLERANCE_MS
  )) return null;
  if (timestampMs <= first.timestamp_ms) return first.bounding_box;
  if (timestampMs >= last.timestamp_ms) return last.bounding_box;

  const nextIndex = observations.findIndex(
    (observation) => observation.timestamp_ms >= timestampMs,
  );
  if (nextIndex <= 0) return first.bounding_box;
  const previous = observations[nextIndex - 1];
  const next = observations[nextIndex];
  const elapsed = next.timestamp_ms - previous.timestamp_ms;
  if (elapsed <= 0) return previous.bounding_box;
  if (elapsed > MAX_OBSERVATION_INTERPOLATION_MS) {
    if (timestampMs - previous.timestamp_ms <= OBSERVATION_EDGE_TOLERANCE_MS) {
      return previous.bounding_box;
    }
    if (next.timestamp_ms - timestampMs <= OBSERVATION_EDGE_TOLERANCE_MS) {
      return next.bounding_box;
    }
    return null;
  }
  const amount = (timestampMs - previous.timestamp_ms) / elapsed;
  return interpolateBox(previous.bounding_box, next.bounding_box, amount);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(milliseconds: number): string {
  const totalSeconds = milliseconds / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
}

function statusLabel(status: VideoJob["status"]): string {
  if (status === "completed") return "Tamamlandı";
  if (status === "processing") return "İşleniyor";
  if (status === "queued") return "Kuyrukta";
  if (status === "cancelled") return "İptal edildi";
  return "Başarısız";
}

function VideoRecognition() {
  const inputRef = useRef<HTMLInputElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [job, setJob] = useState<VideoJob | null>(null);
  const [result, setResult] = useState<VideoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);
  const [videoViewport, setVideoViewport] = useState<VideoViewport | null>(null);
  const [videoAspectRatio, setVideoAspectRatio] = useState<number | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [history, setHistory] = useState<VideoJob[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [deletingVideoId, setDeletingVideoId] = useState<string | null>(null);
  const [faceVideoHistory, setFaceVideoHistory] = useState<VideoFaceHistory | null>(null);
  const [faceHistoryLoading, setFaceHistoryLoading] = useState(false);
  const [liveCameraOpen, setLiveCameraOpen] = useState(false);
  const [liveManifest, setLiveManifest] = useState<LiveVideoManifest | null>(null);
  const [pollWarning, setPollWarning] = useState<string | null>(null);

  const loadHistory = useCallback(async (append = false) => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const offset = append ? history.length : 0;
      const response = await getVideoJobs(HISTORY_PAGE_SIZE, offset);
      setHistory((current) => append ? [...current, ...response.items] : response.items);
      setHistoryTotal(response.total);
    } catch (requestError) {
      setHistoryError(
        requestError instanceof Error
          ? requestError.message
          : "Video geçmişi alınamadı.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, [history.length]);

  const updateVideoViewport = useCallback(() => {
    const stage = stageRef.current;
    const video = videoRef.current;
    if (!stage || !video || video.videoWidth <= 0 || video.videoHeight <= 0) {
      setVideoViewport(null);
      return;
    }
    const stageWidth = stage.clientWidth;
    const stageHeight = stage.clientHeight;
    const videoRatio = video.videoWidth / video.videoHeight;
    const stageRatio = stageWidth / stageHeight;
    if (videoRatio >= stageRatio) {
      const height = stageWidth / videoRatio;
      setVideoViewport({
        left: 0,
        top: (stageHeight - height) / 2,
        width: stageWidth,
        height,
      });
    } else {
      const width = stageHeight * videoRatio;
      setVideoViewport({
        left: (stageWidth - width) / 2,
        top: 0,
        width,
        height: stageHeight,
      });
    }
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth <= 0 || video.videoHeight <= 0) return;
    const ratio = video.videoWidth / video.videoHeight;
    setVideoAspectRatio(Math.min(2.4, Math.max(0.5, ratio)));
    updateVideoViewport();
  }, [updateVideoViewport]);

  useEffect(() => () => {
    if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    void loadHistory();
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !previewUrl) return;
    const observer = new ResizeObserver(updateVideoViewport);
    observer.observe(stage);
    updateVideoViewport();
    return () => observer.disconnect();
  }, [previewUrl, updateVideoViewport]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !previewUrl) return;

    let stopped = false;
    let videoFrameRequest: number | null = null;
    let animationFrameRequest: number | null = null;

    if (typeof video.requestVideoFrameCallback === "function") {
      const updateFromVideoFrame: VideoFrameRequestCallback = (_now, metadata) => {
        if (stopped) return;
        setPlaybackTimeMs(metadata.mediaTime * 1000);
        videoFrameRequest = video.requestVideoFrameCallback(updateFromVideoFrame);
      };
      videoFrameRequest = video.requestVideoFrameCallback(updateFromVideoFrame);
    } else {
      const updateFromAnimationFrame = () => {
        if (stopped) return;
        setPlaybackTimeMs(video.currentTime * 1000);
        animationFrameRequest = window.requestAnimationFrame(updateFromAnimationFrame);
      };
      animationFrameRequest = window.requestAnimationFrame(updateFromAnimationFrame);
    }

    return () => {
      stopped = true;
      if (videoFrameRequest != null && typeof video.cancelVideoFrameCallback === "function") {
        video.cancelVideoFrameCallback(videoFrameRequest);
      }
      if (animationFrameRequest != null) {
        window.cancelAnimationFrame(animationFrameRequest);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    let cancelled = false;
    let timer: number | undefined;
    let consecutiveFailures = 0;

    const poll = async () => {
      try {
        const current = await getVideoJob(job.process_id);
        if (cancelled) return;
        if (current.status === "completed") {
          const completedResult = await getVideoResult(current.process_id);
          if (cancelled) return;
          consecutiveFailures = 0;
          setPollWarning(null);
          setJob(current);
          setResult(completedResult);
          setPreviewUrl(completedResult.video_url);
          setSelectedTrackId(completedResult.tracks[0]?.track_id ?? null);
          void loadHistory();
          return;
        }
        consecutiveFailures = 0;
        setPollWarning(null);
        setJob(current);
        if (current.status === "failed" || current.status === "cancelled") {
          setError(current.error_detail ?? "Video işlemi tamamlanamadı.");
          return;
        }
        timer = window.setTimeout(poll, POLL_INTERVAL_MS);
      } catch {
        if (cancelled) return;
        consecutiveFailures += 1;
        setPollWarning("Sunucu bağlantısı yenileniyor; işlem arka planda devam ediyor.");
        const retryDelay = Math.min(5000, POLL_INTERVAL_MS * consecutiveFailures);
        timer = window.setTimeout(poll, retryDelay);
      }
    };

    timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.process_id, job?.status]);

  const chooseFile = useCallback((selected: File | undefined) => {
    if (!selected) return;
    setError(null);
    setPollWarning(null);
    setJob(null);
    setResult(null);
    setPlaybackTimeMs(0);
    setVideoViewport(null);
    setVideoAspectRatio(null);
    setSelectedTrackId(null);
    setLiveManifest(null);
    const extensionIsMp4 = selected.name.toLowerCase().endsWith(".mp4");
    if (selected.type !== "video/mp4" && !extensionIsMp4) {
      setError("Yalnızca MP4 video seçilebilir.");
      return;
    }
    if (selected.size <= 0 || selected.size > MAX_VIDEO_SIZE) {
      setError("Video boyutu 200 MB sınırını aşamaz.");
      return;
    }
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    chooseFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  };

  const clear = () => {
    setFile(null);
    setPreviewUrl(null);
    setJob(null);
    setResult(null);
    setError(null);
    setPollWarning(null);
    setPlaybackTimeMs(0);
    setVideoViewport(null);
    setVideoAspectRatio(null);
    setSelectedTrackId(null);
    setLiveManifest(null);
  };

  const openHistoryItem = async (selectedJob: VideoJob) => {
    setFile(null);
    setJob(selectedJob);
    setResult(null);
    setError(null);
    setPollWarning(null);
    setPlaybackTimeMs(0);
    setVideoViewport(null);
    setVideoAspectRatio(null);
    setSelectedTrackId(null);
    setLiveManifest(null);
    setPreviewUrl(`/api/videos/${selectedJob.process_id}/content`);
    if (selectedJob.status !== "completed") {
      if (selectedJob.status === "failed" || selectedJob.status === "cancelled") {
        setError(selectedJob.error_detail ?? "Video işlemi tamamlanamadı.");
      }
      return;
    }
    try {
      const completedResult = await getVideoResult(selectedJob.process_id);
      setResult(completedResult);
      setPreviewUrl(completedResult.video_url);
      setSelectedTrackId(completedResult.tracks[0]?.track_id ?? null);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Video sonucu yeniden açılamadı.",
      );
    }
  };

  const removeHistoryItem = async (selectedJob: VideoJob) => {
    if (selectedJob.status === "queued" || selectedJob.status === "processing") return;
    if (!window.confirm(`“${selectedJob.original_filename}” video analizini silmek istiyor musunuz?`)) return;
    setDeletingVideoId(selectedJob.process_id);
    setHistoryError(null);
    try {
      await deleteVideo(selectedJob.process_id);
      setHistory((current) => current.filter((item) => item.process_id !== selectedJob.process_id));
      setHistoryTotal((current) => Math.max(0, current - 1));
      if (job?.process_id === selectedJob.process_id) clear();
    } catch (requestError) {
      setHistoryError(
        requestError instanceof Error
          ? requestError.message
          : "Video analizi silinemedi.",
      );
    } finally {
      setDeletingVideoId(null);
    }
  };

  const submitVideo = async (
    selectedFile: File,
    selectedManifest: LiveVideoManifest | null = null,
  ) => {
    setUploading(true);
    setError(null);
    setPollWarning(null);
    setResult(null);
    try {
      const isLiveWebm = selectedFile.type.toLowerCase().startsWith("video/webm");
      if (isLiveWebm && !selectedManifest) {
        throw new Error("Canlı video analiz bilgisi bulunamadı.");
      }
      const createdJob = isLiveWebm
        ? await uploadLiveVideo(selectedFile, selectedManifest as LiveVideoManifest)
        : await uploadVideo(selectedFile);
      setJob(createdJob);
      void loadHistory();
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Video yüklenemedi.",
      );
    } finally {
      setUploading(false);
    }
  };

  const start = async () => {
    if (file) await submitVideo(file, liveManifest);
  };

  const acceptLiveRecording = (
    recordedFile: File,
    manifest: LiveVideoManifest,
  ) => {
    setFile(recordedFile);
    setLiveManifest(manifest);
    setJob(null);
    setResult(null);
    setError(null);
    setPollWarning(null);
    setPlaybackTimeMs(0);
    setVideoViewport(null);
    setVideoAspectRatio(null);
    setSelectedTrackId(null);
    setPreviewUrl(URL.createObjectURL(recordedFile));
    void submitVideo(recordedFile, manifest);
  };

  const busy = uploading || job?.status === "queued" || job?.status === "processing";
  const rawLiveRecording = job?.content_type === "video/webm";
  const processingTitle = uploading
    ? "Kayıt yükleniyor"
    : rawLiveRecording && job?.status === "queued"
      ? "Video hazırlık kuyruğunda"
      : rawLiveRecording
        ? "Video MP4 formatına dönüştürülüyor"
        : job?.status === "queued"
          ? "Video kuyrukta"
          : "Video analiz ediliyor";
  const processingDetail = uploading
    ? "Kamera kapatıldı; yükleme arka planda devam ediyor."
    : rawLiveRecording && job?.status === "queued"
      ? "Dönüştürme işlemi için sıra bekleniyor."
      : rawLiveRecording
        ? "Kayıt analiz için hızlı biçimde hazırlanıyor."
        : job?.status === "queued"
          ? "İşlem sırası bekleniyor."
          : "Yüzler tespit edilip kimliklerle eşleştiriliyor.";
  const knownCount = result?.tracks.filter((track) => track.status === "known").length ?? 0;
  const appearanceEntries = useMemo<VideoAppearanceEntry[]>(() => {
    if (!result) return [];
    return result.tracks
      .flatMap((track) => {
        const appearances = track.appearances.length
          ? track.appearances
          : [{
            start_ms: track.first_seen_ms,
            end_ms: track.last_seen_ms,
            start_frame: track.best_frame_number ?? 0,
            end_frame: track.best_frame_number ?? 0,
            observation_count: track.observation_count,
            max_recognition_confidence: track.best_recognition_confidence,
            average_recognition_confidence: track.best_recognition_confidence,
          }];
        return appearances.map((appearance) => ({ track, appearance }));
      })
      .sort((first, second) => (
        first.appearance.start_ms - second.appearance.start_ms
        || first.appearance.end_ms - second.appearance.end_ms
        || first.track.track_id - second.track.track_id
      ));
  }, [result]);
  const activeFaces = useMemo<ActiveVideoFace[]>(() => {
    if (!result) return [];
    return result.tracks.flatMap((track) => {
      const boundingBox = boxAtTime(track, playbackTimeMs);
      return boundingBox ? [{ track, boundingBox }] : [];
    });
  }, [playbackTimeMs, result]);
  const selectedTrack = result?.tracks.find(
    (track) => track.track_id === selectedTrackId,
  ) ?? null;

  useEffect(() => {
    if (!selectedTrack) {
      setFaceVideoHistory(null);
      return;
    }
    let cancelled = false;
    setFaceHistoryLoading(true);
    void getVideoFaceHistory(selectedTrack.face_id)
      .then((response) => {
        if (!cancelled) setFaceVideoHistory(response);
      })
      .catch(() => {
        if (!cancelled) setFaceVideoHistory(null);
      })
      .finally(() => {
        if (!cancelled) setFaceHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTrack?.face_id]);
  const timelineDurationMs = Math.max(
    (result?.duration_seconds ?? 0) * 1000,
    ...(result?.tracks.map((track) => track.last_seen_ms) ?? [0]),
    1,
  );

  const seekToAppearance = (trackId: number, timestampMs: number) => {
    setSelectedTrackId(trackId);
    setPlaybackTimeMs(timestampMs);
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = timestampMs / 1000;
    void video.play().catch(() => undefined);
  };

  return (
    <>
      <section className="page-heading">
        <div>
          <p className="eyebrow">Video analizi</p>
          <h1>Video Tanıma</h1>
        </div>
        <div className="privacy-note"><FileVideo2 size={17} /> MP4 · en fazla 200 MB</div>
      </section>

      <section className="video-workspace">
        <div className="tool-panel video-upload-panel">
          <div className="panel-heading">
            <div><span className="step-number">1</span><h2>Video dosyası</h2></div>
            {file && <span>{formatBytes(file.size)}</span>}
          </div>

          <div
            ref={stageRef}
            className={`video-drop-zone ${previewUrl ? "has-video" : ""} ${dragging ? "dragging" : ""}`}
            style={{ aspectRatio: videoAspectRatio ?? 16 / 9 }}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !previewUrl && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (!previewUrl && (event.key === "Enter" || event.key === " ")) inputRef.current?.click();
            }}
          >
            <input ref={inputRef} type="file" accept="video/mp4,.mp4" onChange={handleFileChange} hidden />
            {previewUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={previewUrl}
                  controls
                  preload="metadata"
                  onLoadedMetadata={handleLoadedMetadata}
                  onTimeUpdate={(event) => setPlaybackTimeMs(event.currentTarget.currentTime * 1000)}
                  onSeeked={(event) => setPlaybackTimeMs(event.currentTarget.currentTime * 1000)}
                />
                {result && (
                  <div
                    className="video-overlay-layer"
                    style={videoViewport ?? undefined}
                    aria-hidden="true"
                  >
                    {activeFaces.map(({ track, boundingBox }) => (
                      <div
                        className={`video-face-box ${track.status === "known" ? "known" : "anonymous"}`}
                        key={track.track_id}
                        style={{
                          left: `${boundingBox.x1 * 100}%`,
                          top: `${boundingBox.y1 * 100}%`,
                          width: `${(boundingBox.x2 - boundingBox.x1) * 100}%`,
                          height: `${(boundingBox.y2 - boundingBox.y1) * 100}%`,
                        }}
                      >
                        <span className={`${boundingBox.y1 < 0.1 ? "inside" : ""} ${boundingBox.x2 > 0.82 ? "align-right" : ""}`.trim()}>
                          <strong>{track.name ?? `Anonim · ${track.face_id.slice(0, 8)}`}</strong>
                          {track.best_recognition_confidence != null && (
                            <small>%{(track.best_recognition_confidence * 100).toFixed(1)}</small>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="video-upload-empty">
                <span><FileVideo2 size={34} /></span>
                <strong>Video seçin</strong>
                <small>MP4 · H.264</small>
                <button className="secondary-button" type="button" onClick={(event) => { event.stopPropagation(); inputRef.current?.click(); }}>
                  <Upload size={17} /> Dosya seç
                </button>
              </div>
            )}
          </div>

          {file && <div className="video-file-line"><FileVideo2 size={17} /><span>{file.name}</span><strong>{formatBytes(file.size)}</strong></div>}
          <div className="upload-actions">
            <button className="icon-button danger" type="button" onClick={clear} disabled={(!file && !job) || uploading} title="Videoyu kaldır"><Trash2 size={18} /></button>
            <button className="secondary-button no-margin camera-open-button" type="button" onClick={() => setLiveCameraOpen(true)} disabled={busy} title="Canlı kamerayı aç">
              <Camera size={18} /><span>Canlı kamerayı aç</span>
            </button>
            <button className="primary-button" type="button" onClick={() => void start()} disabled={!file || busy}>
              {busy ? <LoaderCircle className="spin" size={19} /> : <ScanFace size={19} />}
              {uploading ? "Kayıt yükleniyor" : busy ? "İşleniyor" : file?.type === "video/webm" && error ? "Tekrar yükle" : "Videoyu analiz et"}
            </button>
          </div>
        </div>

        <div className="tool-panel video-result-panel" aria-live="polite">
          <div className="panel-heading">
            <div><span className="step-number">2</span><h2>Analiz sonucu</h2></div>
            {job && <span>{job.progress_percent.toFixed(0)}%</span>}
          </div>

          {error ? (
            <div className="video-state error"><AlertCircle size={34} /><strong>İşlem tamamlanamadı</strong><p>{error}</p></div>
          ) : busy ? (
            <div className="video-state processing">
              <LoaderCircle className="spin" size={36} />
              <strong>{processingTitle}</strong>
              <p>{pollWarning ?? processingDetail}</p>
              {job && <div className="video-progress"><span style={{ width: `${Math.max(4, job.progress_percent)}%` }} /></div>}
            </div>
          ) : result ? (
            <div className="video-completed">
              <div className="result-banner success"><CheckCircle2 size={20} /><span>Video analizi tamamlandı</span></div>
              <div className="video-metrics">
                <div><small>Örnek kare</small><strong>{result.sampled_frame_count}</strong></div>
                <div><small>Yüz gözlemi</small><strong>{result.detected_face_count}</strong></div>
                <div><small>Farklı kişi</small><strong>{result.unique_face_count}</strong></div>
              </div>
              <div className="video-track-list">
                {appearanceEntries.map(({ track, appearance }, index) => {
                  const active = selectedTrackId === track.track_id
                    && playbackTimeMs >= appearance.start_ms
                    && playbackTimeMs <= appearance.end_ms + OBSERVATION_EDGE_TOLERANCE_MS;
                  return (
                    <button
                      className={`video-track-row ${active ? "active" : ""}`}
                      key={`${track.track_id}-${appearance.start_ms}-${appearance.end_ms}-${index}`}
                      type="button"
                      aria-label={`${track.name ?? `Anonim ${track.face_id.slice(0, 8)}`} ${formatTime(appearance.start_ms)} ile ${formatTime(appearance.end_ms)} arasındaki görünmenin başlangıcına git`}
                      onClick={() => seekToAppearance(track.track_id, appearance.start_ms)}
                    >
                      {track.best_image_url ? <img src={track.best_image_url} alt="" /> : <span className="video-track-icon"><Fingerprint size={20} /></span>}
                      <div>
                        <small>{track.status === "known" ? "Bilinen kişi" : track.status === "new_anonymous" ? "Yeni anonim" : "Anonim"}</small>
                        <strong>{track.name ?? `Anonim · ${track.face_id.slice(0, 8)}`}</strong>
                        <span><Clock3 size={13} /> {formatTime(appearance.start_ms)} – {formatTime(appearance.end_ms)}</span>
                      </div>
                      <b>{track.best_recognition_confidence == null ? "—" : `%${(track.best_recognition_confidence * 100).toFixed(1)}`}</b>
                    </button>
                  );
                })}
              </div>
              {selectedTrack && (
                <div className="video-appearance-timeline">
                  <div className="video-timeline-heading">
                    <div>
                      <Clock3 size={15} />
                      <span>Görünme zamanları</span>
                    </div>
                    <strong>{selectedTrack.name ?? `Anonim · ${selectedTrack.face_id.slice(0, 8)}`}</strong>
                  </div>
                  <div className="video-timeline-track">
                    {selectedTrack.appearances.map((appearance, index) => {
                      const active = playbackTimeMs >= appearance.start_ms
                        && playbackTimeMs <= appearance.end_ms + OBSERVATION_EDGE_TOLERANCE_MS;
                      return (
                        <button
                          className={active ? "active" : ""}
                          key={`${appearance.start_ms}-${appearance.end_ms}-${index}`}
                          type="button"
                          title={`${formatTime(appearance.start_ms)} – ${formatTime(appearance.end_ms)}`}
                          aria-label={`${formatTime(appearance.start_ms)} ile ${formatTime(appearance.end_ms)} arasına git`}
                          style={{
                            left: `${(appearance.start_ms / timelineDurationMs) * 100}%`,
                            width: `${Math.max(1.5, ((appearance.end_ms - appearance.start_ms) / timelineDurationMs) * 100)}%`,
                          }}
                          onClick={() => seekToAppearance(selectedTrack.track_id, appearance.start_ms)}
                        />
                      );
                    })}
                    <span
                      className="video-timeline-playhead"
                      style={{ left: `${Math.min(100, Math.max(0, (playbackTimeMs / timelineDurationMs) * 100))}%` }}
                    />
                  </div>
                  <div className="video-appearance-list">
                    {selectedTrack.appearances.map((appearance, index) => (
                      <button
                        className={playbackTimeMs >= appearance.start_ms && playbackTimeMs <= appearance.end_ms + OBSERVATION_EDGE_TOLERANCE_MS ? "active" : ""}
                        key={`${appearance.start_frame}-${appearance.end_frame}-${index}`}
                        type="button"
                        aria-label={`${formatTime(appearance.start_ms)} ile ${formatTime(appearance.end_ms)} arasındaki görünmenin başlangıcına git`}
                        onClick={() => seekToAppearance(selectedTrack.track_id, appearance.start_ms)}
                      >
                        <Clock3 size={13} />
                        {formatTime(appearance.start_ms)} – {formatTime(appearance.end_ms)}
                      </button>
                    ))}
                  </div>
                  <div className="video-face-history">
                    <div className="video-face-history-heading">
                      <span><History size={14} /> Video geçmişi</span>
                      <strong>{faceVideoHistory?.total ?? 0} video</strong>
                    </div>
                    {faceHistoryLoading ? (
                      <div className="video-face-history-state">
                        <LoaderCircle className="spin" size={16} /> Geçmiş yükleniyor
                      </div>
                    ) : faceVideoHistory?.items.length ? (
                      <div className="video-face-history-list">
                        {faceVideoHistory.items.map((historyItem) => (
                          <div className="video-face-history-row" key={historyItem.process_id}>
                            <div>
                              <strong>{historyItem.original_filename}</strong>
                              <small>{formatDate(historyItem.created_at)}</small>
                            </div>
                            <span>
                              {historyItem.appearances.map((appearance, index) => (
                                <small key={`${appearance.start_ms}-${appearance.end_ms}-${index}`}>
                                  {formatTime(appearance.start_ms)} – {formatTime(appearance.end_ms)}
                                </small>
                              ))}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="video-face-history-state">Kayıtlı video geçmişi yok.</div>
                    )}
                  </div>
                </div>
              )}
              <div className="video-result-summary"><Users size={16} /> {knownCount} bilinen, {result.tracks.length - knownCount} anonim kişi</div>
            </div>
          ) : (
            <div className="video-state"><FileVideo2 size={36} /><strong>Sonuç bekleniyor</strong><p>Bir video dosyası seçin.</p></div>
          )}

          {job && <div className="process-reference" title={job.process_id}><span>Process ID</span><strong>{job.process_id}</strong></div>}
        </div>
      </section>

      <section className="video-history-section">
        <div className="video-history-heading">
          <div>
            <p className="eyebrow">Kalıcı kayıtlar</p>
            <h2>Video geçmişi</h2>
            <span>{historyTotal} analiz</span>
          </div>
          <button
            className="icon-button"
            type="button"
            title="Geçmişi yenile"
            aria-label="Geçmişi yenile"
            disabled={historyLoading}
            onClick={() => void loadHistory()}
          >
            <RefreshCw className={historyLoading ? "spin" : ""} size={18} />
          </button>
        </div>

        {historyError ? (
          <div className="video-history-message error"><AlertCircle size={18} /> {historyError}</div>
        ) : history.length ? (
          <div className="video-history-list">
            {history.map((historyJob) => (
              <div
                className={`video-history-row ${job?.process_id === historyJob.process_id ? "active" : ""}`}
                key={historyJob.process_id}
              >
                <button
                  className="video-history-open"
                  type="button"
                  onClick={() => void openHistoryItem(historyJob)}
                >
                  <span className={`video-history-status ${historyJob.status}`}><FileVideo2 size={18} /></span>
                  <span className="video-history-name">
                    <strong>{historyJob.original_filename}</strong>
                    <small>{formatDate(historyJob.created_at)} · {statusLabel(historyJob.status)}</small>
                  </span>
                  <span className="video-history-metrics">
                    <small>{historyJob.duration_seconds == null ? "—" : formatTime(historyJob.duration_seconds * 1000)}</small>
                    <strong>{historyJob.unique_face_count} kişi</strong>
                  </span>
                  <ChevronRight size={18} />
                </button>
                <button
                  className="icon-button danger video-history-delete"
                  type="button"
                  title="Video analizini sil"
                  aria-label={`${historyJob.original_filename} video analizini sil`}
                  disabled={
                    historyJob.status === "queued"
                    || historyJob.status === "processing"
                    || deletingVideoId === historyJob.process_id
                  }
                  onClick={() => void removeHistoryItem(historyJob)}
                >
                  {deletingVideoId === historyJob.process_id
                    ? <LoaderCircle className="spin" size={17} />
                    : <Trash2 size={17} />}
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="video-history-message"><History size={20} /> Henüz video analizi bulunmuyor.</div>
        )}

        {history.length < historyTotal && (
          <button
            className="secondary-button video-history-more"
            type="button"
            disabled={historyLoading}
            onClick={() => void loadHistory(true)}
          >
            {historyLoading ? <LoaderCircle className="spin" size={17} /> : <History size={17} />}
            Daha eski analizler
          </button>
        )}
      </section>
      {liveCameraOpen && (
        <LiveVideoRecognition
          onClose={() => setLiveCameraOpen(false)}
          onRecordingReady={acceptLiveRecording}
        />
      )}
    </>
  );
}

export default VideoRecognition;
