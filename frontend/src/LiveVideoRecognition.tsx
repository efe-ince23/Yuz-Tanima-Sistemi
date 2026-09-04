import {
  AlertCircle,
  Camera,
  CameraOff,
  Clock3,
  Fingerprint,
  LoaderCircle,
  Pause,
  Play,
  ScanFace,
  Square,
  SwitchCamera,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { identifyFace } from "./api";
import type {
  IdentifiedFace,
  IdentifyResponse,
  LiveVideoManifest,
  LiveVideoObservation,
} from "./types";

interface LiveVideoRecognitionProps {
  onClose: () => void;
  onRecordingReady: (file: File, manifest: LiveVideoManifest) => void;
}

interface CapturedFrame {
  file: File;
  width: number;
  height: number;
}

type FacingMode = "user" | "environment";

const ANALYSIS_DELAY_MS = 900;
const MAX_ANALYSIS_WIDTH = 960;

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "Kamera izni verilmedi. Tarayıcı ayarlarından kamera iznini açın.";
    }
    if (error.name === "NotFoundError") {
      return "Kullanılabilir bir kamera bulunamadı.";
    }
    if (error.name === "NotReadableError") {
      return "Kamera başka bir uygulama tarafından kullanılıyor olabilir.";
    }
  }
  return "Kamera başlatılamadı. Kamera bağlantısını ve tarayıcı iznini kontrol edin.";
}

function captureFrame(video: HTMLVideoElement, facingMode: FacingMode): Promise<CapturedFrame> {
  const scale = Math.min(1, MAX_ANALYSIS_WIDTH / video.videoWidth);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return Promise.reject(new Error("Kamera karesi hazırlanamadı."));

  if (facingMode === "user") {
    context.translate(width, 0);
    context.scale(-1, 1);
  }
  context.drawImage(video, 0, 0, width, height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Kamera karesi hazırlanamadı."));
        return;
      }
      resolve({
        file: new File([blob], `canli-kare-${Date.now()}.jpg`, { type: "image/jpeg" }),
        width,
        height,
      });
    }, "image/jpeg", 0.84);
  });
}

function faceLabel(face: IdentifiedFace): string {
  if (face.status === "known" && face.person) {
    return `${face.person.first_name} ${face.person.last_name}`;
  }
  return `Anonim · ${face.face_id.slice(0, 8)}`;
}

function LiveVideoRecognition({ onClose, onRecordingReady }: LiveVideoRecognitionProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordedMimeTypeRef = useRef("video/webm");
  const pendingRecordingRef = useRef<Blob | null>(null);
  const recordingStartedAtRef = useRef<number | null>(null);
  const liveObservationsRef = useRef<LiveVideoObservation[]>([]);
  const analysisCountRef = useRef(0);
  const firstAnalysisAtRef = useRef<number | null>(null);
  const lastAnalysisAtRef = useRef<number | null>(null);
  const closeRequestRef = useRef<() => void>(onClose);
  const cameraRequestRef = useRef(0);
  const [facingMode, setFacingMode] = useState<FacingMode>("user");
  const [cameraCount, setCameraCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [result, setResult] = useState<IdentifyResponse | null>(null);
  const [frameSize, setFrameSize] = useState({ width: 1, height: 1 });
  const [cameraRatio, setCameraRatio] = useState(4 / 3);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [recording, setRecording] = useState(false);
  const [saving, setSaving] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingError, setRecordingError] = useState<string | null>(null);

  const stopCurrentStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startRecording = useCallback((stream: MediaStream) => {
    if (recorderRef.current?.state === "recording") return;
    if (typeof MediaRecorder === "undefined") {
      setRecordingError("Bu tarayici canli video kaydini desteklemiyor.");
      return;
    }

    const preferredMimeTypes = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4;codecs=avc1",
      "video/mp4",
    ];
    const mimeType = preferredMimeTypes.find((candidate) => (
      MediaRecorder.isTypeSupported(candidate)
    ));
    try {
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 1_500_000 })
        : new MediaRecorder(stream, { videoBitsPerSecond: 1_500_000 });
      recordedChunksRef.current = [];
      pendingRecordingRef.current = null;
      liveObservationsRef.current = [];
      analysisCountRef.current = 0;
      firstAnalysisAtRef.current = null;
      lastAnalysisAtRef.current = null;
      recordedMimeTypeRef.current = recorder.mimeType || mimeType || "video/webm";
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        setRecording(false);
        setRecordingError("Canli kamera kaydi sirasinda bir hata olustu.");
      };
      recorderRef.current = recorder;
      recorder.start(1000);
      recordingStartedAtRef.current = Date.now();
      setRecordingSeconds(0);
      setRecordingError(null);
      setRecording(true);
    } catch {
      setRecordingError("Canli kamera kaydi baslatilamadi.");
    }
  }, []);

  const stopRecording = useCallback(() => new Promise<Blob | null>((resolve) => {
    if (pendingRecordingRef.current) {
      resolve(pendingRecordingRef.current);
      return;
    }
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      resolve(null);
      return;
    }
    recorder.addEventListener("stop", () => {
      const blob = recordedChunksRef.current.length
        ? new Blob(recordedChunksRef.current, { type: recordedMimeTypeRef.current })
        : null;
      pendingRecordingRef.current = blob;
      setRecording(false);
      resolve(blob);
    }, { once: true });
    try {
      recorder.requestData();
    } catch {
      // Some browsers flush the final chunk only when stop() is called.
    }
    try {
      recorder.stop();
    } catch {
      setRecording(false);
      resolve(null);
    }
  }), []);

  const startCamera = useCallback(async (mode: FacingMode) => {
    stopCurrentStream();
    const requestVersion = ++cameraRequestRef.current;
    setLoading(true);
    setReady(false);
    setCameraError(null);
    setAnalysisError(null);
    setResult(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setLoading(false);
      setCameraError("Bu tarayıcı kamera erişimini desteklemiyor.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 960 },
          height: { ideal: 540 },
        },
      });
      if (requestVersion !== cameraRequestRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      const devices = await navigator.mediaDevices.enumerateDevices();
      setCameraCount(devices.filter((device) => device.kind === "videoinput").length);
    } catch (error) {
      if (requestVersion === cameraRequestRef.current) {
        setLoading(false);
        setCameraError(cameraErrorMessage(error));
      }
    }
  }, [stopCurrentStream]);

  useEffect(() => {
    void startCamera("user");
    return () => {
      cameraRequestRef.current += 1;
      if (recorderRef.current?.state !== "inactive") recorderRef.current?.stop();
      stopCurrentStream();
    };
  }, [startCamera, stopCurrentStream]);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeRequestRef.current();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  useEffect(() => {
    if (!recording || recordingStartedAtRef.current == null) return;
    const updateElapsed = () => {
      setRecordingSeconds(Math.max(
        0,
        Math.floor((Date.now() - (recordingStartedAtRef.current ?? Date.now())) / 1000),
      ));
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 500);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => {
    if (!ready || !running || cameraError) return;
    let cancelled = false;
    let timer: number | undefined;
    let activeRequest: AbortController | null = null;

    const analyze = async () => {
      const video = videoRef.current;
      if (cancelled || !video || video.videoWidth <= 0 || video.videoHeight <= 0) return;
      setAnalyzing(true);
      setAnalysisError(null);
      const startedAt = performance.now();
      let nextDelay = ANALYSIS_DELAY_MS;
      try {
        const capturedAtMs = recordingStartedAtRef.current == null
          ? 0
          : Math.max(0, Date.now() - recordingStartedAtRef.current);
        const captured = await captureFrame(video, facingMode);
        activeRequest = new AbortController();
        const response = await identifyFace(
          captured.file,
          activeRequest.signal,
          "live_video_frame",
        );
        activeRequest = null;
        if (cancelled) return;
        // Keep model warm-up outside the saved video timeline. The first request can
        // take several seconds while the inference providers initialize.
        if (recordingStartedAtRef.current == null && streamRef.current) {
          startRecording(streamRef.current);
        }
        const observations = response.faces.map<LiveVideoObservation>((face) => {
          const sourceLeft = Math.min(1, Math.max(0, face.bounding_box.x1 / captured.width));
          const sourceRight = Math.min(1, Math.max(0, face.bounding_box.x2 / captured.width));
          const x1 = facingMode === "user" ? 1 - sourceRight : sourceLeft;
          const x2 = facingMode === "user" ? 1 - sourceLeft : sourceRight;
          return {
            timestamp_ms: capturedAtMs,
            face_id: face.face_id,
            status: face.status,
            name: face.status === "known" && face.person
              ? `${face.person.first_name} ${face.person.last_name}`
              : null,
            metadata: face.status === "known" && face.person
              ? { description: face.person.description }
              : null,
            bounding_box: {
              x1,
              y1: Math.min(1, Math.max(0, face.bounding_box.y1 / captured.height)),
              x2,
              y2: Math.min(1, Math.max(0, face.bounding_box.y2 / captured.height)),
            },
            detection_confidence: face.detection_confidence,
            recognition_confidence: face.similarity,
            matched_image_url: face.matched_image_url,
          };
        });
        if (recordingStartedAtRef.current != null) {
          liveObservationsRef.current.push(...observations);
          analysisCountRef.current += 1;
          if (firstAnalysisAtRef.current == null) firstAnalysisAtRef.current = capturedAtMs;
          lastAnalysisAtRef.current = capturedAtMs;
        }
        setFrameSize({ width: captured.width, height: captured.height });
        setResult(response);
        setAnalysisCount((current) => current + 1);
        setLatencyMs(Math.round(performance.now() - startedAt));
      } catch (error) {
        if (cancelled) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        nextDelay = 2500;
        setAnalysisError(
          error instanceof Error ? error.message : "Canlı kare analiz edilemedi.",
        );
      } finally {
        if (!cancelled) {
          setAnalyzing(false);
          timer = window.setTimeout(() => void analyze(), nextDelay);
        }
      }
    };

    timer = window.setTimeout(() => void analyze(), 250);
    return () => {
      cancelled = true;
      activeRequest?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
      setAnalyzing(false);
    };
  }, [cameraError, facingMode, ready, running, startRecording]);

  const switchCamera = async () => {
    if (recording || saving) return;
    const nextMode: FacingMode = facingMode === "user" ? "environment" : "user";
    setFacingMode(nextMode);
    await startCamera(nextMode);
  };

  const finishAndSave = useCallback(async () => {
    if (saving) return;
    const hasRecording = recording || pendingRecordingRef.current != null;
    if (!hasRecording) {
      stopCurrentStream();
      onClose();
      return;
    }

    setSaving(true);
    setRunning(false);
    setRecordingError(null);
    try {
      const blob = pendingRecordingRef.current ?? await stopRecording();
      if (!blob || blob.size === 0) {
        throw new Error("Canlı kamera kaydı oluşturulamadı.");
      }
      const isMp4 = blob.type.toLowerCase().startsWith("video/mp4");
      const extension = isMp4 ? "mp4" : "webm";
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const file = new File(
        [blob],
        `canli-kamera-${timestamp}.${extension}`,
        { type: isMp4 ? "video/mp4" : "video/webm" },
      );
      const manifest: LiveVideoManifest = {
        version: 1,
        duration_ms: Math.max(
          1,
          Date.now() - (recordingStartedAtRef.current ?? Date.now()),
        ),
        analysis_count: analysisCountRef.current,
        first_analysis_ms: firstAnalysisAtRef.current,
        last_analysis_ms: lastAnalysisAtRef.current,
        observations: liveObservationsRef.current,
      };
      onRecordingReady(file, manifest);
      stopCurrentStream();
      onClose();
    } catch (error) {
      setRecordingError(
        error instanceof Error ? error.message : "Canlı kamera kaydı kaydedilemedi.",
      );
    } finally {
      setSaving(false);
    }
  }, [onClose, onRecordingReady, recording, saving, stopCurrentStream, stopRecording]);

  closeRequestRef.current = () => {
    void finishAndSave();
  };

  const visibleFaces = useMemo(() => result?.faces ?? [], [result]);
  const statusText = cameraError
    ? "Kamera kapalı"
    : saving
      ? "Kayıt sonlandırılıyor"
      : !ready
        ? "Kamera hazırlanıyor"
        : !running
          ? "Analiz duraklatıldı"
          : analyzing
            ? "Yüzler analiz ediliyor"
            : recording
              ? "Canlı analiz ve kayıt aktif"
              : "Canlı analiz aktif";

  return (
    <div className="modal-backdrop live-video-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) void finishAndSave();
    }}>
      <div className="modal-dialog live-video-dialog" role="dialog" aria-modal="true" aria-labelledby="live-video-title">
        <div className="modal-header">
          <div>
            <span className="modal-icon"><ScanFace size={20} /></span>
            <div><p>Video tanıma</p><h2 id="live-video-title">Canlı kamera analizi</h2></div>
          </div>
          <button className="icon-button" type="button" onClick={() => void finishAndSave()} disabled={saving} title="Kaydı bitir ve kaydet" aria-label="Kaydı bitir ve kaydet"><X size={18} /></button>
        </div>

        <div className="live-video-stage" style={{ aspectRatio: cameraRatio }}>
          <video
            ref={videoRef}
            className={facingMode === "user" ? "mirrored" : ""}
            autoPlay
            muted
            playsInline
            onLoadedMetadata={(event) => {
              const video = event.currentTarget;
              if (video.videoWidth > 0 && video.videoHeight > 0) {
                setCameraRatio(video.videoWidth / video.videoHeight);
              }
              setReady(true);
              setLoading(false);
            }}
          />

          {ready && !cameraError && visibleFaces.map((face) => {
            const box = face.bounding_box;
            const left = box.x1 / frameSize.width;
            const top = box.y1 / frameSize.height;
            const width = (box.x2 - box.x1) / frameSize.width;
            const height = (box.y2 - box.y1) / frameSize.height;
            return (
              <div
                className={`video-face-box live-face-box ${face.status === "known" ? "known" : "anonymous"}`}
                key={`${face.face_index}-${face.face_id}`}
                style={{
                  left: `${left * 100}%`,
                  top: `${top * 100}%`,
                  width: `${width * 100}%`,
                  height: `${height * 100}%`,
                }}
              >
                <span className={`${top < 0.12 ? "inside" : ""} ${left + width > 0.82 ? "align-right" : ""}`.trim()}>
                  <strong>{faceLabel(face)}</strong>
                  <small>%{((face.similarity ?? face.detection_confidence) * 100).toFixed(1)}</small>
                </span>
              </div>
            );
          })}

          <div className={`live-video-status ${running && ready && !cameraError ? "active" : ""}`}>
            {analyzing ? <LoaderCircle className="spin" size={14} /> : <span />}
            {statusText}
          </div>

          {recording && (
            <div className="live-recording-indicator">
              <span /> REC <Clock3 size={13} /> {Math.floor(recordingSeconds / 60).toString().padStart(2, "0")}:{(recordingSeconds % 60).toString().padStart(2, "0")}
            </div>
          )}

          {loading && !cameraError && (
            <div className="camera-overlay"><LoaderCircle className="spin" size={30} /><span>Kamera hazırlanıyor</span></div>
          )}
          {cameraError && (
            <div className="camera-overlay error"><CameraOff size={34} /><strong>Kamera açılamadı</strong><span>{cameraError}</span></div>
          )}
        </div>

        <div className="live-video-results" aria-live="polite">
          <div className="live-video-result-heading">
            <div><strong>Anlık sonuç</strong><small>{analysisCount} kare analiz edildi{latencyMs != null ? ` · ${latencyMs} ms` : ""}</small></div>
            <span>{visibleFaces.length} yüz</span>
          </div>
          {recordingError ? (
            <div className="live-video-message error"><AlertCircle size={16} /> {recordingError}</div>
          ) : analysisError ? (
            <div className="live-video-message error"><AlertCircle size={16} /> {analysisError}</div>
          ) : visibleFaces.length ? (
            <div className="live-face-list">
              {visibleFaces.map((face) => (
                <div className={face.status === "known" ? "known" : "anonymous"} key={`result-${face.face_index}-${face.face_id}`}>
                  <span>{face.status === "known" ? <ScanFace size={18} /> : <Fingerprint size={18} />}</span>
                  <div><strong>{faceLabel(face)}</strong><small>{face.status === "known" ? "Bilinen kişi" : face.status === "new_anonymous" ? "Yeni anonim" : "Anonim"}</small></div>
                  <b>%{((face.similarity ?? face.detection_confidence) * 100).toFixed(1)}</b>
                </div>
              ))}
            </div>
          ) : (
            <div className="live-video-message"><Camera size={17} /> {result?.status === "no_face" ? "Kamerada yüz bulunamadı." : "İlk analiz sonucu bekleniyor."}</div>
          )}
          {result && <div className="live-process-id" title={result.process_id}>Process ID · {result.process_id}</div>}
        </div>

        <div className="camera-actions live-camera-actions">
          <div>
            {cameraCount > 1 && !cameraError && (
              <button className="secondary-button no-margin" type="button" onClick={() => void switchCamera()} disabled={loading || recording || saving} title={recording ? "Kayıt sırasında kamera değiştirilemez" : "Kamerayı değiştir"}>
                <SwitchCamera size={18} /> Kamerayı değiştir
              </button>
            )}
          </div>
          {cameraError ? (
            <button className="primary-button compact" type="button" onClick={() => void startCamera(facingMode)} disabled={loading}>
              <Camera size={18} /> Tekrar dene
            </button>
          ) : (
            <div className="live-camera-primary-actions">
              <button className={running ? "secondary-button no-margin" : "primary-button compact"} type="button" onClick={() => setRunning((current) => !current)} disabled={!ready || saving}>
                {running ? <Pause size={18} /> : <Play size={18} />}
                {running ? "Analizi duraklat" : "Analize devam et"}
              </button>
              <button className="primary-button compact live-save-button" type="button" onClick={() => void finishAndSave()} disabled={!ready || saving}>
                {saving ? <LoaderCircle className="spin" size={18} /> : <Square size={17} />}
                {saving ? "Kaydediliyor" : pendingRecordingRef.current ? "Kaydetmeyi tekrar dene" : "Kaydı bitir ve kaydet"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default LiveVideoRecognition;
