import {
  AlertCircle,
  BarChart3,
  Camera,
  Check,
  CheckCircle2,
  Clapperboard,
  Cpu,
  Fingerprint,
  ImagePlus,
  LoaderCircle,
  LogOut,
  RefreshCw,
  Search,
  ScanFace,
  Trash2,
  Upload,
  Users,
  UserCog,
  UserRound,
  UserPlus,
  UserX,
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

import { getCurrentUser, getHealth, getPersons, identifyFace, logout } from "./api";
import AdminUsers from "./AdminUsers";
import AdvancedSearch from "./AdvancedSearch";
import AuthScreen from "./AuthScreen";
import CameraCapture from "./CameraCapture";
import EnrollAnonymousModal from "./EnrollAnonymousModal";
import IdentityManager from "./IdentityManager";
import PersonsManager from "./PersonsManager";
import PhotoHistory from "./PhotoHistory";
import StatisticsDashboard from "./StatisticsDashboard";
import VideoRecognition from "./VideoRecognition";
import type { AppUser, HealthResponse, IdentifyResponse, Person, PhotoHistoryItem } from "./types";

type RecognitionState = "idle" | "ready" | "loading" | "recognized" | "unknown" | "no_face" | "error";
type AppView = "recognition" | "video" | "search" | "persons" | "identities" | "statistics" | "users";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const recognitionWorkspaceRef = useRef<HTMLElement>(null);
  const [activeView, setActiveView] = useState<AppView>("recognition");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [state, setState] = useState<RecognitionState>("idle");
  const [result, setResult] = useState<IdentifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [persons, setPersons] = useState<Person[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [systemError, setSystemError] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [enrollFaceId, setEnrollFaceId] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<AppUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [photoHistoryVersion, setPhotoHistoryVersion] = useState(0);

  useEffect(() => {
    let mounted = true;
    getCurrentUser()
      .then((user) => { if (mounted) setCurrentUser(user); })
      .catch(() => { if (mounted) setCurrentUser(null); })
      .finally(() => { if (mounted) setAuthLoading(false); });
    return () => { mounted = false; };
  }, []);

  const loadSystemData = useCallback(async () => {
    if (!currentUser) return;
    setSystemError(false);
    const [healthResult, personsResult] = await Promise.allSettled([
      getHealth(),
      currentUser.role === "admin" ? getPersons() : Promise.resolve([]),
    ]);

    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setHealth(null);
      setSystemError(true);
    }

    if (personsResult.status === "fulfilled") {
      setPersons(personsResult.value);
    } else {
      setSystemError(true);
    }
  }, [currentUser]);

  useEffect(() => {
    if (currentUser) void loadSystemData();
  }, [currentUser, loadSystemData]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const totalFaceImages = useMemo(
    () => persons.reduce((total, person) => total + person.face_image_count, 0),
    [persons],
  );

  const chooseFile = useCallback(
    (selectedFile: File | undefined) => {
      if (!selectedFile) return;

      setResult(null);
      setError(null);
      if (!selectedFile.type.startsWith("image/")) {
        setState("error");
        setError("Yalnızca JPG veya PNG türünde bir fotoğraf seçin.");
        return;
      }
      if (selectedFile.size > MAX_FILE_SIZE) {
        setState("error");
        setError("Fotoğraf boyutu 10 MB sınırını geçemez.");
        return;
      }

      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setState("ready");
    },
    [],
  );

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    chooseFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  };

  const clearSelection = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setState("idle");
  };

  const runRecognition = async () => {
    if (!file) return;
    setState("loading");
    setError(null);
    setResult(null);
    try {
      const response = await identifyFace(file);
      setResult(response);
      setState(response.status === "no_face" ? "no_face" : response.recognized ? "recognized" : "unknown");
      setPhotoHistoryVersion((current) => current + 1);
    } catch (requestError) {
      setState("error");
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Tanıma işlemi tamamlanamadı.",
      );
    }
  };

  const handleAnonymousEnrolled = (person: Person) => {
    setResult((current) => {
      if (!current) return current;
      const enrolledPerson = {
        id: person.id,
        first_name: person.first_name,
        last_name: person.last_name,
        description: person.description,
      };
      const faces = current.faces.map((face) => face.face_id === person.face_id
        ? { ...face, status: "known" as const, recognized: true, person: enrolledPerson }
        : face);
      const enrolledFace = faces.find((face) => face.face_id === person.face_id);
      return {
        ...current,
        status: "recognized",
        recognized: true,
        similarity: enrolledFace?.similarity ?? current.similarity,
        person: enrolledPerson,
        face_id: person.face_id,
        faces,
      };
    });
    setState("recognized");
    setEnrollFaceId(null);
    void loadSystemData();
  };

  const similarityPercent = result?.similarity == null
    ? null
    : Math.max(0, Math.min(100, result.similarity * 100));

  const openPhotoHistoryItem = (item: PhotoHistoryItem) => {
    setFile(null);
    setPreviewUrl(item.image_url);
    setResult(item.result);
    setError(null);
    setState(
      !item.result
        ? "error"
        : item.result.status === "no_face"
          ? "no_face"
          : item.result.recognized
            ? "recognized"
            : "unknown",
    );
    window.requestAnimationFrame(() => {
      recognitionWorkspaceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  if (authLoading) {
    return <div className="auth-loading"><Fingerprint size={34} /><LoaderCircle className="spin" size={24} /></div>;
  }
  if (!currentUser) {
    return <AuthScreen onAuthenticated={(user) => { setCurrentUser(user); setActiveView("recognition"); }} />;
  }

  const handleLogout = async () => {
    await logout();
    setCurrentUser(null);
    setPersons([]);
    setResult(null);
    setActiveView("recognition");
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><ScanFace size={24} /></span>
          <span><strong>Yüz Tanıma</strong><small>Kimlik eşleştirme</small></span>
        </div>

        <nav className="nav-list" aria-label="Ana menü">
          <button className={`nav-item ${activeView === "recognition" ? "active" : ""}`} type="button" onClick={() => setActiveView("recognition")}>
            <ScanFace size={19} /><span>Tanıma</span>
          </button>
          <button className={`nav-item ${activeView === "video" ? "active" : ""}`} type="button" onClick={() => setActiveView("video")}>
            <Clapperboard size={19} /><span>Video tanıma</span>
          </button>
          <button className={`nav-item ${activeView === "search" ? "active" : ""}`} type="button" onClick={() => setActiveView("search")}>
            <Search size={19} /><span>Gelişmiş arama</span>
          </button>
          {currentUser.role === "admin" && <button className={`nav-item ${activeView === "persons" ? "active" : ""}`} type="button" onClick={() => setActiveView("persons")}>
            <Users size={19} /><span>Kayıtlı kişiler</span><b>{persons.length}</b>
          </button>}
          <button className={`nav-item ${activeView === "identities" ? "active" : ""}`} type="button" onClick={() => setActiveView("identities")}>
            <Fingerprint size={19} /><span>{currentUser.role === "admin" ? "Kimlikler" : "Kimliklerim"}</span>
          </button>
          <button className={`nav-item ${activeView === "statistics" ? "active" : ""}`} type="button" onClick={() => setActiveView("statistics")}>
            <BarChart3 size={19} /><span>İstatistikler</span>
          </button>
          {currentUser.role === "admin" && <button className={`nav-item ${activeView === "users" ? "active" : ""}`} type="button" onClick={() => setActiveView("users")}>
            <UserCog size={19} /><span>Kullanıcılar</span>
          </button>}
        </nav>

        {currentUser.role === "admin" && <div className="sidebar-summary">
          <div><span>Kişi</span><strong>{persons.length}</strong></div>
          <div><span>Yüz kaydı</span><strong>{totalFaceImages}</strong></div>
        </div>}
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="mobile-brand"><ScanFace size={22} /><strong>Yüz Tanıma</strong></div>
          <nav className="mobile-view-nav" aria-label="Mobil ana menü">
            <button className={activeView === "recognition" ? "active" : ""} type="button" onClick={() => setActiveView("recognition")} title="Tanıma"><ScanFace size={18} /></button>
            <button className={activeView === "video" ? "active" : ""} type="button" onClick={() => setActiveView("video")} title="Video tanıma"><Clapperboard size={18} /></button>
            <button className={activeView === "search" ? "active" : ""} type="button" onClick={() => setActiveView("search")} title="Gelişmiş arama"><Search size={18} /></button>
            {currentUser.role === "admin" && <button className={activeView === "persons" ? "active" : ""} type="button" onClick={() => setActiveView("persons")} title="Kayıtlı kişiler"><Users size={18} /></button>}
            <button className={activeView === "identities" ? "active" : ""} type="button" onClick={() => setActiveView("identities")} title="Kimlikler"><Fingerprint size={18} /></button>
            <button className={activeView === "statistics" ? "active" : ""} type="button" onClick={() => setActiveView("statistics")} title="İstatistikler"><BarChart3 size={18} /></button>
            {currentUser.role === "admin" && <button className={activeView === "users" ? "active" : ""} type="button" onClick={() => setActiveView("users")} title="Kullanıcılar"><UserCog size={18} /></button>}
          </nav>
          <div className={`system-status ${health ? "online" : "offline"}`}>
            <span className="status-dot" />
            {health ? "Sistem aktif" : "Bağlantı yok"}
          </div>
          <div className="provider-status">
            <Cpu size={17} />
            {health?.execution_providers.includes("CUDAExecutionProvider")
              ? "CUDA aktif"
              : "CPU modu"}
          </div>
          {systemError && (
            <button className="icon-button" type="button" onClick={() => void loadSystemData()} title="Yeniden bağlan">
              <RefreshCw size={18} />
            </button>
          )}
          <div className="account-summary" title={currentUser.email}>
            <span><UserRound size={17} /></span>
            <div><strong>{currentUser.full_name}</strong><small>{currentUser.role === "admin" ? "Yönetici" : "Kullanıcı"}</small></div>
          </div>
          <button className="icon-button" type="button" onClick={() => void handleLogout()} title="Çıkış yap"><LogOut size={18} /></button>
        </header>

        <div className="content-wrap">
          {activeView === "recognition" ? (
            <>
          <section className="page-heading">
            <div>
              <p className="eyebrow">Canlı eşleştirme</p>
              <h1>Yüz Tanıma</h1>
            </div>
          </section>

          <section className="recognition-workspace" ref={recognitionWorkspaceRef}>
            <div className="tool-panel upload-panel">
              <div className="panel-heading">
                <div><span className="step-number">1</span><h2>Test fotoğrafı</h2></div>
                <span>JPG veya PNG · en fazla 10 MB</span>
              </div>

              <div
                className={`drop-zone ${previewUrl ? "has-image" : ""} ${isDragging ? "dragging" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => !previewUrl && inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (!previewUrl && (event.key === "Enter" || event.key === " ")) inputRef.current?.click();
                }}
              >
                <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleFileChange} hidden />
                {previewUrl ? (
                  <img src={previewUrl} alt="Seçilen test fotoğrafı" />
                ) : (
                  <div className="empty-upload">
                    <span className="upload-icon"><ImagePlus size={30} /></span>
                    <strong>Fotoğraf seçin</strong>
                    <span>veya buraya sürükleyin</span>
                    <button className="secondary-button" type="button" onClick={(event) => { event.stopPropagation(); inputRef.current?.click(); }}>
                      <Upload size={17} /> Dosya seç
                    </button>
                  </div>
                )}
              </div>

              <div className="upload-actions">
                <button className="icon-button danger" type="button" onClick={clearSelection} disabled={!previewUrl || state === "loading"} title="Fotoğrafı kaldır">
                  <Trash2 size={18} />
                </button>
                <button className="secondary-button no-margin camera-open-button" type="button" onClick={() => setCameraOpen(true)} disabled={state === "loading"} title="Kamerayı aç">
                  <Camera size={18} /><span>Kamerayı aç</span>
                </button>
                <button className="primary-button" type="button" onClick={() => void runRecognition()} disabled={!file || state === "loading"}>
                  {state === "loading" ? <LoaderCircle className="spin" size={19} /> : <ScanFace size={19} />}
                  {state === "loading" ? "Analiz ediliyor" : "Kişiyi tanı"}
                </button>
              </div>
            </div>

            <div className={`tool-panel result-panel state-${state}`} aria-live="polite">
              <div className="panel-heading">
                <div><span className="step-number">2</span><h2>Tanıma sonucu</h2></div>
              </div>

              {state === "idle" || state === "ready" ? (
                <div className="result-placeholder">
                  <span><ScanFace size={36} /></span>
                  <strong>{state === "ready" ? "Fotoğraf hazır" : "Sonuç bekleniyor"}</strong>
                  <p>{state === "ready" ? "Tanıma işlemini başlatın." : "Bir test fotoğrafı seçin."}</p>
                </div>
              ) : state === "loading" ? (
                <div className="result-placeholder loading">
                  <span><LoaderCircle className="spin" size={36} /></span>
                  <strong>Yüz analiz ediliyor</strong>
                  <p>Veritabanındaki kayıtlar karşılaştırılıyor.</p>
                </div>
              ) : result && result.faces.length > 1 && (state === "recognized" || state === "unknown") ? (
                <div className="multi-face-result">
                  <div className={`result-banner ${result.recognized ? "success" : "neutral"}`}>
                    <Users size={20} />
                    <span>{result.detected_face_count} yüz ayrı ayrı incelendi</span>
                  </div>
                  <div className="face-result-list">
                    {result.faces.map((face) => {
                      const faceSimilarity = face.similarity == null
                        ? null
                        : Math.max(0, Math.min(100, face.similarity * 100));
                      return (
                        <div className={`face-result-row ${face.recognized ? "recognized" : "unrecognized"}`} key={face.face_index}>
                          <span className="face-result-status" aria-hidden="true">
                            {face.recognized ? <CheckCircle2 size={21} /> : <UserX size={21} />}
                          </span>
                          <div className="face-result-copy">
                            <small>
                              Yüz {face.face_index + 1} · {face.status === "known"
                                ? "Bilinen kişi"
                                : face.status === "anonymous"
                                  ? "Daha önce görülen anonim yüz"
                                  : "Yeni anonim yüz"}
                            </small>
                            {face.person ? (
                              <strong>{face.person.first_name} {face.person.last_name}</strong>
                            ) : (
                              <strong>Anonim yüz · {face.face_id.slice(0, 8)}</strong>
                            )}
                          </div>
                          <div className="face-result-actions">
                            {faceSimilarity !== null && (
                              <span className="face-result-score">%{faceSimilarity.toFixed(2)}</span>
                            )}
                            {face.status !== "known" && (
                              <button className="face-enroll-button" type="button" onClick={() => setEnrollFaceId(face.face_id)} title="Anonim yüzü isimlendir">
                                <UserPlus size={17} />
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="multi-face-summary">
                    <Check size={16} />
                    {result.faces.filter((face) => face.status === "known").length} bilinen, {result.faces.filter((face) => face.status !== "known").length} anonim
                  </div>
                  <div className="provider-line"><Check size={16} /> {result.execution_providers[0]}</div>
                </div>
              ) : state === "recognized" && result?.person ? (
                <div className="recognized-result">
                  <div className="result-banner success"><CheckCircle2 size={20} /><span>Kişi tanındı</span></div>
                  <div className="identity-row">
                    <div className="identity-copy">
                      <span className="person-id">Kayıt #{result.person.id}</span>
                      <h3>{result.person.first_name} {result.person.last_name}</h3>
                      {result.person.description && <p>{result.person.description}</p>}
                    </div>
                    {result.matched_image_url && (
                      <img className="matched-face" src={result.matched_image_url} alt={`${result.person.first_name} ${result.person.last_name} referans fotoğrafı`} />
                    )}
                  </div>
                  <div className="confidence-block">
                    <div><span>Benzerlik</span><strong>%{similarityPercent?.toFixed(2)}</strong></div>
                    <div className="confidence-track"><span style={{ width: `${similarityPercent ?? 0}%` }} /></div>
                    <small>Eşik değeri %{(result.threshold * 100).toFixed(0)}</small>
                  </div>
                  <div className="provider-line"><Check size={16} /> {result.execution_providers[0]}</div>
                  {result.ignored_face_count > 0 && (
                    <div className="background-face-note">
                      <Users size={16} /> Arka plandaki {result.ignored_face_count} küçük yüz yok sayıldı.
                    </div>
                  )}
                </div>
              ) : state === "no_face" ? (
                <div className="unknown-result no-face-result">
                  <span className="result-icon unknown"><ScanFace size={34} /></span>
                  <h3>Fotoğrafta yüz bulunamadı</h3>
                  <p>Yüzün net ve görünür olduğu başka bir görüntü deneyin.</p>
                </div>
              ) : state === "unknown" && result?.faces.length === 1 ? (
                <div className="unknown-result anonymous-result">
                  <span className="result-icon anonymous"><Fingerprint size={34} /></span>
                  <h3>{result.faces[0].status === "new_anonymous" ? "Yeni anonim yüz" : "Anonim yüz yeniden görüldü"}</h3>
                  <p>
                    {result.faces[0].status === "new_anonymous"
                      ? "Bu yüz ilk kez görüldü ve kalıcı bir kimlik oluşturuldu."
                      : "Bu yüz daha önce görülen anonim kimlikle eşleşti."}
                  </p>
                  <div className="anonymous-face-id" title={result.faces[0].face_id}>
                    Face ID <strong>{result.faces[0].face_id}</strong>
                  </div>
                  {similarityPercent !== null && (
                    <div className="unknown-score">Benzerlik <strong>%{similarityPercent.toFixed(2)}</strong></div>
                  )}
                  <button className="secondary-button anonymous-enroll-button" type="button" onClick={() => setEnrollFaceId(result.faces[0].face_id)}>
                    <UserPlus size={17} /> Kimlik bilgisi ekle
                  </button>
                </div>
              ) : state === "unknown" ? (
                <div className="unknown-result">
                  <span className="result-icon unknown"><UserX size={34} /></span>
                  <h3>Kişi tanınmadı</h3>
                  <p>Fotoğraf kayıtlı kişilerle yeterince eşleşmedi.</p>
                  {similarityPercent !== null && (
                    <div className="unknown-score">Benzerlik <strong>%{similarityPercent.toFixed(2)}</strong></div>
                  )}
                  {result && result.ignored_face_count > 0 && (
                    <div className="background-face-note">
                      <Users size={16} /> Arka plandaki {result.ignored_face_count} küçük yüz yok sayıldı.
                    </div>
                  )}
                </div>
              ) : (
                <div className="unknown-result error-result">
                  <span className="result-icon error"><AlertCircle size={34} /></span>
                  <h3>Analiz tamamlanamadı</h3>
                  <p>{error}</p>
                </div>
              )}
              {currentUser.role === "admin" && result?.process_id && (
                <div className="process-reference" title={result.process_id}>
                  <span>Process ID</span><strong>{result.process_id}</strong>
                </div>
              )}
            </div>
          </section>
          <PhotoHistory
            refreshVersion={photoHistoryVersion}
            isAdmin={currentUser.role === "admin"}
            selectedProcessId={result?.process_id ?? null}
            onSelect={openPhotoHistoryItem}
          />
            </>
          ) : activeView === "video" ? (
            <VideoRecognition />
          ) : activeView === "search" ? (
            <AdvancedSearch isAdmin={currentUser.role === "admin"} />
          ) : activeView === "persons" ? (
            <PersonsManager persons={persons} systemError={systemError} onRefresh={loadSystemData} />
          ) : activeView === "identities" ? (
            <IdentityManager
              onPersonsChanged={loadSystemData}
              isAdmin={currentUser.role === "admin"}
            />
          ) : activeView === "statistics" ? (
            <StatisticsDashboard />
          ) : (
            <AdminUsers currentUserId={currentUser.id} />
          )}
        </div>
      </main>
      {cameraOpen && (
        <CameraCapture
          onClose={() => setCameraOpen(false)}
          onCapture={(capturedFile) => {
            chooseFile(capturedFile);
            setCameraOpen(false);
          }}
        />
      )}
      {enrollFaceId && (
        <EnrollAnonymousModal
          faceId={enrollFaceId}
          onClose={() => setEnrollFaceId(null)}
          onEnrolled={handleAnonymousEnrolled}
        />
      )}
    </div>
  );
}

export default App;
