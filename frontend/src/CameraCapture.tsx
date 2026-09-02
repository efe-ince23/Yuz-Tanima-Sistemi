import {
  AlertCircle,
  Camera,
  CameraOff,
  LoaderCircle,
  SwitchCamera,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  onClose: () => void;
}

type FacingMode = "user" | "environment";

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

function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const requestVersionRef = useRef(0);
  const [facingMode, setFacingMode] = useState<FacingMode>("user");
  const [cameraCount, setCameraCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stopCurrentStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async (mode: FacingMode) => {
    stopCurrentStream();
    const requestVersion = ++requestVersionRef.current;
    setLoading(true);
    setReady(false);
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setLoading(false);
      setError("Bu tarayıcı kamera erişimini desteklemiyor.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
      });

      if (requestVersion !== requestVersionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      const devices = await navigator.mediaDevices.enumerateDevices();
      setCameraCount(devices.filter((device) => device.kind === "videoinput").length);
    } catch (cameraError) {
      if (requestVersion === requestVersionRef.current) {
        setLoading(false);
        setError(cameraErrorMessage(cameraError));
      }
    }
  }, [stopCurrentStream]);

  useEffect(() => {
    void startCamera("user");
    return () => {
      requestVersionRef.current += 1;
      stopCurrentStream();
    };
  }, [startCamera, stopCurrentStream]);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !capturing) onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [capturing, onClose]);

  const switchCamera = async () => {
    const nextMode: FacingMode = facingMode === "user" ? "environment" : "user";
    setFacingMode(nextMode);
    await startCamera(nextMode);
  };

  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || !ready || video.videoWidth === 0 || video.videoHeight === 0) return;

    setCapturing(true);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setCapturing(false);
      setError("Fotoğraf hazırlanamadı. Lütfen tekrar deneyin.");
      return;
    }

    if (facingMode === "user") {
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCapturing(false);
        setError("Fotoğraf hazırlanamadı. Lütfen tekrar deneyin.");
        return;
      }
      onCapture(new File([blob], `kamera-${Date.now()}.jpg`, { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !capturing) onClose();
    }}>
      <div className="modal-dialog camera-dialog" role="dialog" aria-modal="true" aria-labelledby="camera-title">
        <div className="modal-header">
          <div>
            <span className="modal-icon"><Camera size={20} /></span>
            <div><p>Canlı görüntü</p><h2 id="camera-title">Kameradan fotoğraf çek</h2></div>
          </div>
          <button className="icon-button" type="button" onClick={onClose} disabled={capturing} title="Kamerayı kapat"><X size={18} /></button>
        </div>

        <div className={`camera-stage ${facingMode === "user" ? "mirrored" : ""}`}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            onLoadedMetadata={() => {
              setReady(true);
              setLoading(false);
            }}
          />
          {!error && <div className="camera-face-guide" aria-hidden="true" />}
          {loading && !error && <div className="camera-overlay"><LoaderCircle className="spin" size={30} /><span>Kamera hazırlanıyor</span></div>}
          {error && <div className="camera-overlay error"><CameraOff size={34} /><strong>Kamera açılamadı</strong><span>{error}</span></div>}
        </div>

        <div className="camera-actions">
          {cameraCount > 1 && !error ? (
            <button className="secondary-button no-margin" type="button" onClick={() => void switchCamera()} disabled={loading || capturing}>
              <SwitchCamera size={18} /> Kamerayı değiştir
            </button>
          ) : <span />}
          {error ? (
            <button className="primary-button compact" type="button" onClick={() => void startCamera(facingMode)} disabled={loading}>
              <Camera size={18} /> Tekrar dene
            </button>
          ) : (
            <button className="primary-button compact camera-shutter" type="button" onClick={captureFrame} disabled={!ready || capturing}>
              {capturing ? <LoaderCircle className="spin" size={19} /> : <Camera size={19} />}
              {capturing ? "Hazırlanıyor" : "Çek ve kullan"}
            </button>
          )}
        </div>
        {error && <div className="camera-error-line"><AlertCircle size={15} /> Kamera izni adres çubuğundaki site ayarlarından değiştirilebilir.</div>}
      </div>
    </div>
  );
}

export default CameraCapture;
