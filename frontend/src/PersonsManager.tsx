import {
  AlertCircle,
  Camera,
  CheckCircle2,
  ChevronRight,
  Fingerprint,
  ImagePlus,
  LoaderCircle,
  Pencil,
  Plus,
  Save,
  Search,
  Trash2,
  Upload,
  UserRound,
  Users,
  X,
} from "lucide-react";
import { type FormEvent, type UIEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  createPerson,
  deleteFaceImage,
  deletePerson,
  getPersonFaceImages,
  updatePerson,
  uploadPersonFaceImage,
} from "./api";
import type { FaceImage, Person, PersonInput } from "./types";

interface PersonsManagerProps {
  persons: Person[];
  systemError: boolean;
  onRefresh: () => Promise<void>;
}

interface PersonFormModalProps {
  person?: Person;
  onClose: () => void;
  onSubmit: (input: PersonInput, faceImage?: File) => Promise<void>;
}

type DeleteTarget =
  | { type: "person"; person: Person }
  | { type: "face"; face: FaceImage };

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const DIRECTORY_RESULT_LIMIT = 200;

function PersonFormModal({ person, onClose, onSubmit }: PersonFormModalProps) {
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [firstName, setFirstName] = useState(person?.first_name ?? "");
  const [lastName, setLastName] = useState(person?.last_name ?? "");
  const [description, setDescription] = useState(person?.description ?? "");
  const [faceImage, setFaceImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, saving]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const selectFaceImage = (file: File | undefined) => {
    setError(null);
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Yalnızca JPG veya PNG türünde bir fotoğraf seçin.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("Fotoğraf boyutu 10 MB sınırını geçemez.");
      return;
    }
    setFaceImage(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!firstName.trim() || !lastName.trim()) {
      setError("Ad ve soyad alanları zorunludur.");
      return;
    }
    if (!person && !faceImage) {
      setError("Yeni kişi için bir referans fotoğrafı seçin.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      await onSubmit({
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        description: description.trim() || null,
      }, faceImage ?? undefined);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Kişi kaydedilemedi.");
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !saving) onClose();
    }}>
      <div className="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="person-form-title">
        <div className="modal-header">
          <div>
            <span className="modal-icon"><UserRound size={20} /></span>
            <div><p>{person ? `Kayıt #${person.id}` : "Yeni kayıt"}</p><h2 id="person-form-title">{person ? "Kişiyi düzenle" : "Yeni kişi ekle"}</h2></div>
          </div>
          <button className="icon-button" type="button" onClick={onClose} disabled={saving} title="Kapat"><X size={18} /></button>
        </div>

        <form className="person-form" onSubmit={(event) => void handleSubmit(event)}>
          <div className="form-grid">
            <label><span>Ad</span><input autoFocus value={firstName} onChange={(event) => setFirstName(event.target.value)} maxLength={100} placeholder="Ad" /></label>
            <label><span>Soyad</span><input value={lastName} onChange={(event) => setLastName(event.target.value)} maxLength={100} placeholder="Soyad" /></label>
          </div>
          <label><span>Açıklama</span><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={1000} rows={4} placeholder="Kısa açıklama" /></label>

          {!person && (
            <div className="initial-photo-field">
              <span>Referans fotoğrafı</span>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                hidden
                onChange={(event) => {
                  selectFaceImage(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
              <button className={`initial-photo-picker ${previewUrl ? "has-preview" : ""}`} type="button" onClick={() => imageInputRef.current?.click()} disabled={saving}>
                {previewUrl ? (
                  <><img src={previewUrl} alt="Seçilen referans fotoğrafı" /><span><Upload size={17} /> Fotoğrafı değiştir</span></>
                ) : (
                  <><span className="initial-photo-icon"><ImagePlus size={25} /></span><strong>İlk yüz fotoğrafını seçin</strong><small>JPG veya PNG · en fazla 10 MB</small></>
                )}
              </button>
            </div>
          )}

          {error && <div className="form-error"><AlertCircle size={17} /> {error}</div>}

          <div className="modal-actions">
            <button className="secondary-button no-margin" type="button" onClick={onClose} disabled={saving}>Vazgeç</button>
            <button className="primary-button compact" type="submit" disabled={saving}>
              {saving ? <LoaderCircle className="spin" size={18} /> : <Save size={18} />}
              {saving ? "Kaydediliyor" : "Kaydet"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

interface ConfirmDialogProps {
  target: DeleteTarget;
  busy: boolean;
  error: string | null;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
}

function ConfirmDialog({ target, busy, error, onCancel, onConfirm }: ConfirmDialogProps) {
  const isPerson = target.type === "person";
  const personName = isPerson
    ? `${target.person.first_name} ${target.person.last_name}`
    : "bu referans fotoğrafı";

  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-dialog confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-title">
        <span className="danger-dialog-icon"><Trash2 size={24} /></span>
        <h2 id="delete-title">{isPerson ? "Kişi silinsin mi?" : "Fotoğraf silinsin mi?"}</h2>
        <p><strong>{personName}</strong> {isPerson ? "ve kişiye ait bütün referans fotoğrafları kalıcı olarak silinecek." : "kalıcı olarak silinecek."}</p>
        {error && <div className="form-error"><AlertCircle size={17} /> {error}</div>}
        <div className="modal-actions">
          <button className="secondary-button no-margin" type="button" onClick={onCancel} disabled={busy}>Vazgeç</button>
          <button className="danger-button" type="button" onClick={() => void onConfirm()} disabled={busy}>
            {busy ? <LoaderCircle className="spin" size={18} /> : <Trash2 size={18} />}
            {busy ? "Siliniyor" : "Kalıcı olarak sil"}
          </button>
        </div>
      </div>
    </div>
  );
}

function PersonsManager({ persons, systemError, onRefresh }: PersonsManagerProps) {
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const loadMoreLocked = useRef(false);
  const [query, setQuery] = useState("");
  const [visibleLimit, setVisibleLimit] = useState(DIRECTORY_RESULT_LIMIT);
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [faceImages, setFaceImages] = useState<FaceImage[]>([]);
  const [facesLoading, setFacesLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [formPerson, setFormPerson] = useState<Person | "new" | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const selectedPerson = persons.find((person) => person.id === selectedPersonId) ?? null;

  const filteredPersons = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase("tr-TR");
    if (!normalizedQuery) return persons;
    return persons.filter((person) =>
      `${person.id} ${person.face_id} ${person.first_name} ${person.last_name} ${person.description ?? ""}`
        .toLocaleLowerCase("tr-TR")
        .includes(normalizedQuery),
    );
  }, [persons, query]);

  const sortedPersons = useMemo(() => {
    return [...filteredPersons]
      .sort((personA, personB) => {
        const sourcePriority = Number(personA.description === "LFW Dataset")
          - Number(personB.description === "LFW Dataset");
        return sourcePriority || personA.id - personB.id;
      });
  }, [filteredPersons]);

  const displayedPersons = useMemo(
    () => sortedPersons.slice(0, visibleLimit),
    [sortedPersons, visibleLimit],
  );

  useEffect(() => {
    loadMoreLocked.current = false;
    setVisibleLimit(DIRECTORY_RESULT_LIMIT);
  }, [query]);

  const handleDirectoryScroll = (event: UIEvent<HTMLDivElement>) => {
    const directory = event.currentTarget;
    const distanceToBottom = directory.scrollHeight - directory.scrollTop - directory.clientHeight;
    if (distanceToBottom > 48) {
      loadMoreLocked.current = false;
      return;
    }
    if (loadMoreLocked.current || visibleLimit >= sortedPersons.length) return;
    loadMoreLocked.current = true;
    setVisibleLimit((current) => Math.min(current + DIRECTORY_RESULT_LIMIT, sortedPersons.length));
  };

  const loadFaces = async (personId: number) => {
    setFacesLoading(true);
    try {
      setFaceImages(await getPersonFaceImages(personId));
    } catch (loadError) {
      setFaceImages([]);
      setNotice({ type: "error", text: loadError instanceof Error ? loadError.message : "Fotoğraflar yüklenemedi." });
    } finally {
      setFacesLoading(false);
    }
  };

  useEffect(() => {
    if (selectedPersonId !== null) void loadFaces(selectedPersonId);
    else setFaceImages([]);
  }, [selectedPersonId]);

  useEffect(() => {
    if (selectedPersonId !== null && !persons.some((person) => person.id === selectedPersonId)) {
      setSelectedPersonId(null);
    }
  }, [persons, selectedPersonId]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 4500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const handleFormSubmit = async (input: PersonInput, faceImage?: File) => {
    if (formPerson === "new") {
      const created = await createPerson(input);
      try {
        if (!faceImage) throw new Error("Referans fotoğrafı seçilmedi.");
        await uploadPersonFaceImage(created.id, faceImage);
      } catch (uploadError) {
        try {
          await deletePerson(created.id);
        } catch {
          throw new Error(`Kişi #${created.id} oluşturuldu ancak fotoğraf yüklenemedi. Kaydı listeden silin.`);
        }
        throw uploadError;
      }
      await onRefresh();
      setSelectedPersonId(created.id);
      setFormPerson(null);
      setNotice({ type: "success", text: "Kişi ve ilk referans fotoğrafı kaydedildi." });
      return;
    }
    if (formPerson) {
      await updatePerson(formPerson.id, input);
      await onRefresh();
      setFormPerson(null);
      setNotice({ type: "success", text: "Kişi bilgileri güncellendi." });
    }
  };

  const handleUpload = async (file: File | undefined) => {
    if (!file || !selectedPerson) return;
    if (!file.type.startsWith("image/")) {
      setNotice({ type: "error", text: "Yalnızca bir fotoğraf dosyası seçin." });
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setNotice({ type: "error", text: "Fotoğraf boyutu 10 MB sınırını geçemez." });
      return;
    }

    setUploading(true);
    try {
      await uploadPersonFaceImage(selectedPerson.id, file);
      await Promise.all([loadFaces(selectedPerson.id), onRefresh()]);
      setNotice({ type: "success", text: "Referans fotoğrafı kaydedildi." });
    } catch (uploadError) {
      setNotice({ type: "error", text: uploadError instanceof Error ? uploadError.message : "Fotoğraf yüklenemedi." });
    } finally {
      setUploading(false);
      if (uploadInputRef.current) uploadInputRef.current.value = "";
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      if (deleteTarget.type === "person") {
        await deletePerson(deleteTarget.person.id);
        setSelectedPersonId(null);
        await onRefresh();
        setNotice({ type: "success", text: "Kişi ve referans fotoğrafları silindi." });
      } else if (selectedPerson) {
        await deleteFaceImage(selectedPerson.id, deleteTarget.face.id);
        await Promise.all([loadFaces(selectedPerson.id), onRefresh()]);
        setNotice({ type: "success", text: "Referans fotoğrafı silindi." });
      }
      setDeleteTarget(null);
    } catch (requestError) {
      setDeleteError(requestError instanceof Error ? requestError.message : "Silme işlemi tamamlanamadı.");
    } finally {
      setDeleteBusy(false);
    }
  };

  return (
    <>
      <section className="management-heading">
        <div><p className="eyebrow">Veritabanı yönetimi</p><h1>Kayıtlı Kişiler</h1><p>{persons.length} kişi ve {persons.reduce((sum, person) => sum + person.face_image_count, 0)} referans yüz</p></div>
        <button className="primary-button compact" type="button" onClick={() => setFormPerson("new")}><Plus size={19} /> Yeni kişi ekle</button>
      </section>

      {notice && <div className={`notice-banner ${notice.type}`} role="status">
        {notice.type === "success" ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
        <span>{notice.text}</span><button type="button" onClick={() => setNotice(null)} title="Kapat"><X size={16} /></button>
      </div>}

      <section className="management-layout">
        <div className="directory-panel">
          <div className="directory-toolbar">
            <div className="search-field"><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Kişi veya ID ara" aria-label="Kişi ara" /></div>
            <span>{displayedPersons.length} / {filteredPersons.length} kayıt gösteriliyor</span>
          </div>

          <div className="person-directory" data-testid="person-directory" onScroll={handleDirectoryScroll}>
            {displayedPersons.map((person) => (
              <button
                className={`directory-row ${selectedPersonId === person.id ? "selected" : ""}`}
                key={person.id}
                type="button"
                onClick={() => setSelectedPersonId(person.id)}
                data-person-id={person.id}
              >
                <span className={`directory-avatar ${person.sample_image_url ? "has-photo" : ""}`}>
                  {person.sample_image_url
                    ? <img src={person.sample_image_url} alt="" />
                    : <UserRound size={20} />}
                </span>
                <span className="directory-copy"><strong>{person.first_name} {person.last_name}</strong><small>{person.face_id.slice(0, 8)} · {person.face_image_count} referans</small></span>
                <ChevronRight size={18} />
              </button>
            ))}
            {!filteredPersons.length && (
              <div className="directory-empty"><Users size={26} /><strong>{systemError ? "Kayıtlar alınamadı" : "Kişi bulunamadı"}</strong><span>{query ? "Arama ifadesini değiştirin." : "Yeni bir kişi ekleyin."}</span></div>
            )}
            {!!displayedPersons.length && (
              <div className="directory-progress" data-testid="person-directory-progress">
                {displayedPersons.length} / {filteredPersons.length}
              </div>
            )}
          </div>
        </div>

        <div className="person-detail-panel" data-testid="person-detail">
          {!selectedPerson ? (
            <div className="detail-placeholder"><span><UserRound size={34} /></span><strong>Kişi seçin</strong><p>Bilgileri ve referans fotoğrafları burada görüntülenir.</p></div>
          ) : (
            <>
              <div className="detail-header">
                <div className="detail-identity"><span><UserRound size={25} /></span><div><small>Kayıt #{selectedPerson.id}</small><h2>{selectedPerson.first_name} {selectedPerson.last_name}</h2><p>{selectedPerson.description || "Açıklama yok"}</p></div></div>
                <div className="detail-actions">
                  <button className="icon-button" type="button" onClick={() => setFormPerson(selectedPerson)} title="Kişiyi düzenle"><Pencil size={17} /></button>
                  <button className="icon-button danger" type="button" onClick={() => { setDeleteError(null); setDeleteTarget({ type: "person", person: selectedPerson }); }} title="Kişiyi sil"><Trash2 size={17} /></button>
                </div>
              </div>

              <div className="person-face-id">
                <Fingerprint size={15} /><span>Face ID</span><strong>{selectedPerson.face_id}</strong>
              </div>

              <div className="photos-heading">
                <div><h3>Referans fotoğrafları</h3><span>{selectedPerson.face_image_count} kayıt</span></div>
                <input ref={uploadInputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(event) => void handleUpload(event.target.files?.[0])} />
                <button className="secondary-button no-margin" type="button" onClick={() => uploadInputRef.current?.click()} disabled={uploading}>
                  {uploading ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />}
                  {uploading ? "Yükleniyor" : "Fotoğraf ekle"}
                </button>
              </div>

              {facesLoading ? (
                <div className="faces-loading"><LoaderCircle className="spin" size={24} /> Fotoğraflar yükleniyor</div>
              ) : faceImages.length ? (
                <div className="face-image-grid">
                  {faceImages.map((face) => (
                    <article className="face-image-card" key={face.id}>
                      <img src={face.image_url} alt={`${selectedPerson.first_name} ${selectedPerson.last_name} referans fotoğrafı`} />
                      <div><span><Camera size={14} /> #{face.id}</span><strong>%{(face.detection_confidence * 100).toFixed(1)}</strong></div>
                      <button className="image-delete-button" type="button" onClick={() => { setDeleteError(null); setDeleteTarget({ type: "face", face }); }} title="Fotoğrafı sil"><Trash2 size={16} /></button>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="no-face-images"><span><ImagePlus size={28} /></span><strong>Referans fotoğrafı yok</strong><p>Tanıma yapabilmek için en az bir fotoğraf ekleyin.</p></div>
              )}
            </>
          )}
        </div>
      </section>

      {formPerson && (
        <PersonFormModal
          person={formPerson === "new" ? undefined : formPerson}
          onClose={() => setFormPerson(null)}
          onSubmit={handleFormSubmit}
        />
      )}
      {deleteTarget && (
        <ConfirmDialog
          target={deleteTarget}
          busy={deleteBusy}
          error={deleteError}
          onCancel={() => { if (!deleteBusy) setDeleteTarget(null); }}
          onConfirm={handleDelete}
        />
      )}
    </>
  );
}

export default PersonsManager;
