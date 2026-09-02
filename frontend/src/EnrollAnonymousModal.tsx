import { AlertCircle, Fingerprint, LoaderCircle, Save, X } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";

import { enrollAnonymousIdentity } from "./api";
import type { Person } from "./types";


interface EnrollAnonymousModalProps {
  faceId: string;
  onClose: () => void;
  onEnrolled: (person: Person) => void;
}


function EnrollAnonymousModal({ faceId, onClose, onEnrolled }: EnrollAnonymousModalProps) {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, saving]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const person = await enrollAnonymousIdentity(faceId, {
        first_name: firstName,
        last_name: lastName,
        description: description.trim() || null,
      });
      onEnrolled(person);
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Anonim yüz isimlendirilemedi.",
      );
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !saving) onClose();
    }}>
      <div className="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="anonymous-enroll-title">
        <div className="modal-header">
          <div>
            <span className="modal-icon"><Fingerprint size={20} /></span>
            <div><p>Face ID · {faceId.slice(0, 8)}</p><h2 id="anonymous-enroll-title">Anonim yüzü isimlendir</h2></div>
          </div>
          <button className="icon-button" type="button" onClick={onClose} disabled={saving} title="Kapat"><X size={18} /></button>
        </div>

        <form className="person-form" onSubmit={(event) => void handleSubmit(event)}>
          <div className="form-grid">
            <label><span>Ad</span><input autoFocus required value={firstName} onChange={(event) => setFirstName(event.target.value)} maxLength={100} placeholder="Ad" /></label>
            <label><span>Soyad</span><input required value={lastName} onChange={(event) => setLastName(event.target.value)} maxLength={100} placeholder="Soyad" /></label>
          </div>
          <label><span>Açıklama</span><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={1000} rows={4} placeholder="Kısa açıklama" /></label>

          <div className="enroll-face-id-note">
            <Fingerprint size={17} />
            <span>Mevcut face ID korunacak.<strong>{faceId}</strong></span>
          </div>
          {error && <div className="form-error"><AlertCircle size={17} /> {error}</div>}

          <div className="modal-actions">
            <button className="secondary-button no-margin" type="button" onClick={onClose} disabled={saving}>Vazgeç</button>
            <button className="primary-button compact" type="submit" disabled={saving}>
              {saving ? <LoaderCircle className="spin" size={18} /> : <Save size={18} />}
              {saving ? "Kaydediliyor" : "Kimliği kaydet"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default EnrollAnonymousModal;
