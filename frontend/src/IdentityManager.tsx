import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Fingerprint,
  History,
  LoaderCircle,
  Pencil,
  Save,
  Search,
  Trash2,
  UserPlus,
  UserRound,
  X,
} from "lucide-react";
import { type FormEvent, type UIEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { deleteIdentity, getIdentities, getIdentityHistory, getProcess, updateIdentity } from "./api";
import EnrollAnonymousModal from "./EnrollAnonymousModal";
import type { FaceHistory, Identity, Person, PersonInput, RecognitionProcess } from "./types";


interface IdentityManagerProps {
  onPersonsChanged: () => Promise<void>;
  isAdmin: boolean;
}

type IdentityFilter = "all" | "known" | "anonymous";
const HISTORY_PAGE_SIZE = 8;
const DIRECTORY_RESULT_LIMIT = 200;


function ProcessDetailModal({ process, loading, error, onClose }: {
  process: RecognitionProcess | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-dialog process-detail-dialog" role="dialog" aria-modal="true" aria-labelledby="process-detail-title">
        <div className="modal-header">
          <div><span className="modal-icon"><History size={20} /></span><div><p>İşlem kaydı</p><h2 id="process-detail-title">Process detayı</h2></div></div>
          <button className="icon-button" type="button" onClick={onClose} title="Kapat"><X size={18} /></button>
        </div>
        {loading ? <div className="process-detail-state"><LoaderCircle className="spin" size={22} /> İşlem yükleniyor</div>
          : error ? <div className="form-error"><AlertCircle size={17} /> {error}</div>
          : process && <div className="process-detail-body">
            <dl className="process-metadata">
              <div><dt>Process ID</dt><dd>{process.process_id}</dd></div>
              <div><dt>İşlem türü</dt><dd>{process.operation_type}</dd></div>
              <div><dt>Durum</dt><dd>{process.status}</dd></div>
              <div><dt>İşlenen yüz</dt><dd>{process.face_count}</dd></div>
              <div><dt>Başlangıç</dt><dd>{new Date(process.created_at).toLocaleString("tr-TR")}</dd></div>
              <div><dt>Tamamlanma</dt><dd>{process.completed_at ? new Date(process.completed_at).toLocaleString("tr-TR") : "Devam ediyor"}</dd></div>
            </dl>
            <section className="process-face-section">
              <h3>İşlemdeki yüzler</h3>
              {process.events.length ? <div className="process-face-list">{process.events.map((event) => (
                <div className="process-face-row" key={event.id}>
                  <Fingerprint size={17} />
                  <span><strong>{event.face_status ?? "durum yok"}</strong><small>{event.face_id ?? "Face ID oluşturulmadı"}</small></span>
                  <b>{event.similarity === null ? "-" : `%${Math.round(event.similarity * 100)}`}</b>
                </div>
              ))}</div> : <div className="identity-history-empty">Bu işlemde yüz kaydı bulunmuyor.</div>}
            </section>
          </div>}
      </div>
    </div>
  );
}


function IdentityEditModal({ identity, onClose, onSaved }: {
  identity: Identity;
  onClose: () => void;
  onSaved: (input: PersonInput) => Promise<void>;
}) {
  const [firstName, setFirstName] = useState(identity.first_name ?? "");
  const [lastName, setLastName] = useState(identity.last_name ?? "");
  const [description, setDescription] = useState(identity.description ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await onSaved({ first_name: firstName, last_name: lastName, description: description.trim() || null });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Kimlik güncellenemedi.");
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="identity-edit-title">
        <div className="modal-header">
          <div><span className="modal-icon"><Fingerprint size={20} /></span><div><p>Face ID · {identity.face_id.slice(0, 8)}</p><h2 id="identity-edit-title">Kimliği güncelle</h2></div></div>
          <button className="icon-button" type="button" onClick={onClose} disabled={saving} title="Kapat"><X size={18} /></button>
        </div>
        <form className="person-form" onSubmit={(event) => void submit(event)}>
          <div className="form-grid">
            <label><span>Ad</span><input autoFocus required value={firstName} onChange={(event) => setFirstName(event.target.value)} maxLength={100} /></label>
            <label><span>Soyad</span><input required value={lastName} onChange={(event) => setLastName(event.target.value)} maxLength={100} /></label>
          </div>
          <label><span>Açıklama</span><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={1000} rows={4} /></label>
          {error && <div className="form-error"><AlertCircle size={17} /> {error}</div>}
          <div className="modal-actions">
            <button className="secondary-button no-margin" type="button" onClick={onClose} disabled={saving}>Vazgeç</button>
            <button className="primary-button compact" type="submit" disabled={saving}>{saving ? <LoaderCircle className="spin" size={18} /> : <Save size={18} />}{saving ? "Kaydediliyor" : "Kaydet"}</button>
          </div>
        </form>
      </div>
    </div>
  );
}


function IdentityDeleteDialog({ identity, busy, error, onClose, onConfirm }: {
  identity: Identity;
  busy: boolean;
  error: string | null;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-dialog confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="identity-delete-title">
        <span className="danger-dialog-icon"><Trash2 size={24} /></span>
        <h2 id="identity-delete-title">Kimlik silinsin mi?</h2>
        <p><strong>{identity.first_name ? `${identity.first_name} ${identity.last_name}` : `Anonim · ${identity.face_id.slice(0, 8)}`}</strong> ve bu kimliğe ait bütün aktif yüz örnekleri silinecek.</p>
        {error && <div className="form-error"><AlertCircle size={17} /> {error}</div>}
        <div className="modal-actions">
          <button className="secondary-button no-margin" type="button" onClick={onClose} disabled={busy}>Vazgeç</button>
          <button className="danger-button" type="button" onClick={() => void onConfirm()} disabled={busy}>{busy ? <LoaderCircle className="spin" size={18} /> : <Trash2 size={18} />}{busy ? "Siliniyor" : "Kalıcı olarak sil"}</button>
        </div>
      </div>
    </div>
  );
}


function IdentityManager({ onPersonsChanged, isAdmin }: IdentityManagerProps) {
  const loadMoreLocked = useRef(false);
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<IdentityFilter>("all");
  const [visibleLimit, setVisibleLimit] = useState(DIRECTORY_RESULT_LIMIT);
  const [selectedFaceId, setSelectedFaceId] = useState<string | null>(null);
  const [editing, setEditing] = useState<Identity | null>(null);
  const [enrollingFaceId, setEnrollingFaceId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<Identity | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [history, setHistory] = useState<FaceHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const historyRequest = useRef(0);
  const [processDetail, setProcessDetail] = useState<RecognitionProcess | null>(null);
  const [processLoading, setProcessLoading] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);
  const [processOpen, setProcessOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getIdentities();
      setIdentities(result);
      setSelectedFaceId((current) => current && result.some((item) => item.face_id === current) ? current : result[0]?.face_id ?? null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Kimlikler alınamadı.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const loadHistory = useCallback(async (faceId: string, offset = 0, append = false) => {
    const requestId = ++historyRequest.current;
    setHistoryLoading(true);
    setHistoryError(null);
    if (!append) setHistory(null);
    try {
      const result = await getIdentityHistory(faceId, HISTORY_PAGE_SIZE, offset);
      if (requestId !== historyRequest.current) return;
      setHistory((current) => append && current?.face_id === faceId
        ? { ...result, appearances: [...current.appearances, ...result.appearances] }
        : result);
    } catch (requestError) {
      if (requestId !== historyRequest.current) return;
      setHistoryError(requestError instanceof Error ? requestError.message : "Görülme geçmişi alınamadı.");
    } finally {
      if (requestId === historyRequest.current) setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedFaceId) void loadHistory(selectedFaceId);
    else {
      historyRequest.current += 1;
      setHistory(null);
      setHistoryError(null);
      setHistoryLoading(false);
    }
  }, [loadHistory, selectedFaceId]);

  const openProcess = async (processId: string) => {
    setProcessOpen(true);
    setProcessDetail(null);
    setProcessError(null);
    setProcessLoading(true);
    try {
      setProcessDetail(await getProcess(processId));
    } catch (requestError) {
      setProcessError(requestError instanceof Error ? requestError.message : "İşlem detayı alınamadı.");
    } finally {
      setProcessLoading(false);
    }
  };

  const visible = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("tr-TR");
    return identities.filter((identity) => {
      if (filter !== "all" && identity.status !== filter) return false;
      const text = `${identity.first_name ?? ""} ${identity.last_name ?? ""} ${identity.face_id}`.toLocaleLowerCase("tr-TR");
      return !normalized || text.includes(normalized);
    }).sort((identityA, identityB) => {
      const sourcePriority = Number(identityA.description === "LFW Dataset")
        - Number(identityB.description === "LFW Dataset");
      return sourcePriority || (identityA.person_id ?? Number.MAX_SAFE_INTEGER) - (identityB.person_id ?? Number.MAX_SAFE_INTEGER);
    });
  }, [filter, identities, query]);
  const displayedIdentities = useMemo(
    () => visible.slice(0, visibleLimit),
    [visible, visibleLimit],
  );
  const selected = identities.find((identity) => identity.face_id === selectedFaceId) ?? null;

  useEffect(() => {
    loadMoreLocked.current = false;
    setVisibleLimit(DIRECTORY_RESULT_LIMIT);
  }, [filter, query]);

  const handleDirectoryScroll = (event: UIEvent<HTMLDivElement>) => {
    const directory = event.currentTarget;
    const distanceToBottom = directory.scrollHeight - directory.scrollTop - directory.clientHeight;
    if (distanceToBottom > 48) {
      loadMoreLocked.current = false;
      return;
    }
    if (loadMoreLocked.current || visibleLimit >= visible.length) return;
    loadMoreLocked.current = true;
    setVisibleLimit((current) => Math.min(current + DIRECTORY_RESULT_LIMIT, visible.length));
  };

  const saveIdentity = async (input: PersonInput) => {
    if (!editing) return;
    await updateIdentity(editing.face_id, input);
    setEditing(null);
    setNotice("Kimlik bilgileri güncellendi.");
    await Promise.all([load(), onPersonsChanged()]);
  };

  const enrolled = async (_person: Person) => {
    setEnrollingFaceId(null);
    setNotice("Anonim kimlik kayıtlı kişiye dönüştürüldü.");
    await Promise.all([load(), onPersonsChanged()]);
  };

  const remove = async () => {
    if (!deleting) return;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      await deleteIdentity(deleting.face_id);
      setDeleting(null);
      setSelectedFaceId(null);
      setNotice("Kimlik ve aktif yüz örnekleri silindi.");
      await Promise.all([load(), onPersonsChanged()]);
    } catch (requestError) {
      setDeleteError(requestError instanceof Error ? requestError.message : "Kimlik silinemedi.");
    } finally {
      setDeleteBusy(false);
    }
  };

  return (
    <>
      <section className="management-heading identity-heading">
        <div><p className="eyebrow">Face ID yönetimi</p><h1>Kimlik Kayıtları</h1><p>{identities.filter((item) => item.status === "known").length} bilinen · {identities.filter((item) => item.status === "anonymous").length} anonim</p></div>
      </section>
      {notice && <div className="notice-banner success" role="status"><CheckCircle2 size={18} /><span>{notice}</span><button type="button" onClick={() => setNotice(null)} title="Kapat"><X size={16} /></button></div>}
      {error && <div className="notice-banner error" role="alert"><AlertCircle size={18} /><span>{error}</span><button type="button" onClick={() => void load()} title="Yeniden dene"><LoaderCircle size={16} /></button></div>}

      <section className="management-layout identity-layout">
        <div className="directory-panel">
          <div className="identity-toolbar">
            <div className="search-field"><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ad veya face ID ara" aria-label="Kimlik ara" /></div>
            <div className="identity-filter" aria-label="Kimlik filtresi">
              {(["all", "known", "anonymous"] as const).map((value) => <button className={filter === value ? "active" : ""} type="button" onClick={() => setFilter(value)} key={value}>{value === "all" ? "Tümü" : value === "known" ? "Bilinen" : "Anonim"}</button>)}
            </div>
          </div>
          <div className="person-directory identity-directory" data-testid="identity-directory" onScroll={handleDirectoryScroll}>
            {loading ? <div className="faces-loading"><LoaderCircle className="spin" size={22} /> Kimlikler yükleniyor</div> : displayedIdentities.map((identity) => (
              <button className={`directory-row ${selectedFaceId === identity.face_id ? "selected" : ""}`} type="button" key={identity.face_id} data-face-id={identity.face_id} onClick={() => setSelectedFaceId(identity.face_id)}>
                <span className={`directory-avatar ${identity.sample_image_urls[0] ? "has-photo" : ""}`}>
                  {identity.sample_image_urls[0]
                    ? <img src={identity.sample_image_urls[0]} alt="" />
                    : identity.status === "known" ? <UserRound size={20} /> : <Fingerprint size={20} />}
                </span>
                <span className="directory-copy"><strong>{identity.first_name ? `${identity.first_name} ${identity.last_name}` : "Anonim yüz"}</strong><small>{identity.face_id.slice(0, 8)} · {identity.sample_count} örnek</small></span>
                <ChevronRight size={18} />
              </button>
            ))}
            {!loading && !visible.length && <div className="directory-empty"><Fingerprint size={26} /><strong>Kimlik bulunamadı</strong><span>Arama veya filtreyi değiştirin.</span></div>}
            {!loading && !!displayedIdentities.length && (
              <div className="directory-progress" data-testid="identity-directory-progress">
                {displayedIdentities.length} / {visible.length}
              </div>
            )}
          </div>
        </div>

        <div className="person-detail-panel identity-detail">
          {!selected ? <div className="detail-placeholder"><span><Fingerprint size={34} /></span><strong>Kimlik seçin</strong><p>Face ID ve kayıt bilgileri burada görüntülenir.</p></div> : <>
            <div className="detail-header">
              <div className="detail-identity"><span>{selected.status === "known" ? <UserRound size={25} /> : <Fingerprint size={25} />}</span><div><small>{selected.status === "known" ? "Bilinen kimlik" : "Anonim kimlik"}</small><h2>{selected.first_name ? `${selected.first_name} ${selected.last_name}` : `Anonim · ${selected.face_id.slice(0, 8)}`}</h2><p>{selected.description || (selected.status === "anonymous" ? "Kişisel bilgi bulunmuyor" : "Açıklama yok")}</p></div></div>
              <div className="detail-actions">
                {selected.status === "known" ? <button className="icon-button" type="button" onClick={() => setEditing(selected)} title="Kimliği güncelle"><Pencil size={17} /></button> : <button className="icon-button" type="button" onClick={() => setEnrollingFaceId(selected.face_id)} title="Anonim kimliği isimlendir"><UserPlus size={17} /></button>}
                <button className="icon-button danger" type="button" onClick={() => { setDeleteError(null); setDeleting(selected); }} title="Kimliği sil"><Trash2 size={17} /></button>
              </div>
            </div>
            <section className="identity-samples" aria-label="Yüz örnekleri">
              <div className="identity-samples-heading">
                <h3>Yüz örnekleri</h3>
                <span>{selected.sample_image_urls.length} görsel</span>
              </div>
              {selected.sample_image_urls.length ? (
                <div className="identity-sample-grid">
                  {selected.sample_image_urls.map((imageUrl, index) => (
                    <img
                      src={imageUrl}
                      alt={`${selected.first_name ? `${selected.first_name} ${selected.last_name}` : "Anonim yüz"} örneği ${index + 1}`}
                      key={imageUrl}
                    />
                  ))}
                </div>
              ) : (
                <div className="identity-sample-empty"><Fingerprint size={22} /> Görsel örneği bulunmuyor</div>
              )}
            </section>
            <dl className="identity-metadata">
              <div><dt>Face ID</dt><dd>{selected.face_id}</dd></div>
              <div><dt>Durum</dt><dd><span className={`identity-status ${selected.status}`}>{selected.status}</span></dd></div>
              <div><dt>Toplam yüz örneği</dt><dd>{selected.sample_count}</dd></div>
              <div><dt>Referans fotoğrafı</dt><dd>{selected.reference_image_count}</dd></div>
              {isAdmin && <>
                <div><dt>Görülme sayısı</dt><dd>{selected.observation_count}</dd></div>
                <div><dt>Son görülme</dt><dd>{selected.last_seen_at ? new Date(selected.last_seen_at).toLocaleString("tr-TR") : "Henüz yok"}</dd></div>
              </>}
              {!isAdmin && <>
                <div><dt>Fotoğrafta görülme</dt><dd>{selected.photo_observation_count}</dd></div>
                <div><dt>Fotoğrafta son görülme</dt><dd>{selected.photo_last_seen_at ? new Date(selected.photo_last_seen_at).toLocaleString("tr-TR") : "Henüz yok"}</dd></div>
                <div><dt>Videoda görülme</dt><dd>{selected.video_observation_count}</dd></div>
                <div><dt>Videoda son görülme</dt><dd>{selected.video_last_seen_at ? new Date(selected.video_last_seen_at).toLocaleString("tr-TR") : "Henüz yok"}</dd></div>
              </>}
            </dl>
            <section className="identity-history" aria-label="Görülme geçmişi">
              <div className="identity-history-heading">
                <div><History size={18} /><h3>{isAdmin ? "Görülme geçmişi" : "Fotoğraf geçmişi"}</h3></div>
                <span>{history?.total ?? selected.observation_count} kayıt</span>
              </div>
              {historyLoading && !history ? <div className="identity-history-state"><LoaderCircle className="spin" size={18} /> Geçmiş yükleniyor</div>
                : historyError ? <div className="identity-history-error"><AlertCircle size={17} /><span>{historyError}</span><button type="button" onClick={() => void loadHistory(selected.face_id)}>Yeniden dene</button></div>
                : history && history.appearances.length ? <>
                  <div className="identity-history-list">{history.appearances.map((entry) => (
                    <button type="button" className="identity-history-row" key={entry.event_id} disabled={!entry.process_id} onClick={() => entry.process_id && void openProcess(entry.process_id)} title={entry.process_id ? "İşlem detayını aç" : "Bu eski kaydın process ID bilgisi yok"}>
                      <span className={`history-status ${entry.status ?? "unknown"}`}><Fingerprint size={16} /></span>
                      <span className="history-main"><strong>{entry.status ?? "durum yok"}</strong><small><Clock3 size={13} />{new Date(entry.timestamp).toLocaleString("tr-TR")}</small></span>
                      <span className="history-process"><small>Process ID</small><strong>{entry.process_id ? entry.process_id.slice(0, 8) : "Eski kayıt"}</strong></span>
                      <span className="history-score">{entry.similarity === null ? "-" : `%${Math.round(entry.similarity * 100)}`}</span>
                      {entry.process_id && <ChevronRight size={17} />}
                    </button>
                  ))}</div>
                  {history.appearances.length < history.total && <button className="history-more-button" type="button" disabled={historyLoading} onClick={() => void loadHistory(selected.face_id, history.appearances.length, true)}>{historyLoading ? <LoaderCircle className="spin" size={16} /> : <History size={16} />}Daha eski kayıtları göster</button>}
                </> : <div className="identity-history-empty"><History size={20} /> Bu yüz henüz bir tanıma işleminde görülmedi.</div>}
            </section>
          </>}
        </div>
      </section>

      {editing && <IdentityEditModal identity={editing} onClose={() => setEditing(null)} onSaved={saveIdentity} />}
      {enrollingFaceId && <EnrollAnonymousModal faceId={enrollingFaceId} onClose={() => setEnrollingFaceId(null)} onEnrolled={(person) => void enrolled(person)} />}
      {deleting && <IdentityDeleteDialog identity={deleting} busy={deleteBusy} error={deleteError} onClose={() => { if (!deleteBusy) setDeleting(null); }} onConfirm={remove} />}
      {processOpen && <ProcessDetailModal process={processDetail} loading={processLoading} error={processError} onClose={() => setProcessOpen(false)} />}
    </>
  );
}

export default IdentityManager;
