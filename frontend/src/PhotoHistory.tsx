import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Fingerprint,
  History,
  Image as ImageIcon,
  LoaderCircle,
  ScanFace,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { getPhotoHistory } from "./api";
import type { PhotoHistoryItem } from "./types";

const PAGE_SIZE = 12;

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function resultSummary(item: PhotoHistoryItem): string {
  if (item.result?.status === "no_face") return "Yüz bulunamadı";
  const known = item.result?.faces.filter((face) => face.status === "known").length ?? 0;
  const anonymous = item.result?.faces.length ? item.result.faces.length - known : 0;
  return `${known} bilinen, ${anonymous} anonim`;
}

interface PhotoHistoryProps {
  refreshVersion: number;
  isAdmin: boolean;
  selectedProcessId: string | null;
  onSelect: (item: PhotoHistoryItem) => void;
}

function PhotoHistory({ refreshVersion, isAdmin, selectedProcessId, onSelect }: PhotoHistoryProps) {
  const [items, setItems] = useState<PhotoHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (append = false) => {
    setLoading(true);
    setError(null);
    try {
      const offset = append ? items.length : 0;
      const response = await getPhotoHistory(PAGE_SIZE, offset);
      setItems((current) => append ? [...current, ...response.items] : response.items);
      setTotal(response.total);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Fotoğraf geçmişi alınamadı.");
    } finally {
      setLoading(false);
    }
  }, [items.length]);

  useEffect(() => {
    void load(false);
  }, [refreshVersion]);

  return (
    <section className="photo-history-section" aria-label="Fotoğraf geçmişi">
      <div className="photo-history-heading">
        <div>
          <p className="eyebrow">İŞLEM KAYITLARI</p>
          <h2>Fotoğraf geçmişi</h2>
        </div>
        <span><History size={16} /> {total} işlem</span>
      </div>

      {loading && items.length === 0 ? (
        <div className="photo-history-state"><LoaderCircle className="spin" size={22} /> Geçmiş yükleniyor</div>
      ) : error ? (
        <div className="photo-history-state error"><AlertCircle size={22} /> <span>{error}</span><button type="button" onClick={() => void load(false)}>Yeniden dene</button></div>
      ) : items.length === 0 ? (
        <div className="photo-history-state"><ImageIcon size={25} /><strong>Henüz fotoğraf işlemi yok</strong><span>Analiz ettiğiniz fotoğraflar ve sonuçları burada görünecek.</span></div>
      ) : (
        <>
          <div className="photo-history-list">
            {items.map((item) => (
              <button
                className={`photo-history-row ${selectedProcessId === item.process_id ? "active" : ""}`}
                key={item.process_id}
                type="button"
                onClick={() => onSelect(item)}
                aria-label={`${item.original_filename ?? "Fotoğraf"} sonucunu aç`}
              >
                <img src={item.image_url} alt={item.original_filename ?? "İşlenen fotoğraf"} loading="lazy" />
                <div className="photo-history-main">
                  <div className="photo-history-meta">
                    <strong>{item.original_filename ?? "Kamera fotoğrafı"}</strong>
                    <span><Clock3 size={13} /> {formatDate(item.created_at)}</span>
                  </div>
                  <div className="photo-history-summary">
                    <span className={item.result?.status === "recognized" ? "known" : "anonymous"}>
                      {item.result?.status === "recognized" ? <CheckCircle2 size={15} /> : <ScanFace size={15} />}
                      {resultSummary(item)}
                    </span>
                    {isAdmin && <small title={item.process_id}>Process ID: {item.process_id.slice(0, 8)}</small>}
                  </div>
                  {item.result && item.result.faces.length > 0 && (
                    <div className="photo-history-faces">
                      {item.result.faces.map((face) => (
                        <div key={`${item.process_id}-${face.face_index}`}>
                          <Fingerprint size={15} />
                          <span>
                            <strong>{face.person ? `${face.person.first_name} ${face.person.last_name}` : `Anonim ${face.face_id.slice(0, 8)}`}</strong>
                            <small>{face.status === "known" ? "Bilinen kişi" : face.status === "anonymous" ? "Daha önce görüldü" : "Yeni anonim"}</small>
                          </span>
                          <b>{face.similarity == null ? "-" : `%${Math.max(0, face.similarity * 100).toFixed(1)}`}</b>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
          {items.length < total && (
            <button className="photo-history-more" type="button" disabled={loading} onClick={() => void load(true)}>
              {loading ? <LoaderCircle className="spin" size={17} /> : <History size={17} />} Daha eski işlemleri göster
            </button>
          )}
        </>
      )}
    </section>
  );
}

export default PhotoHistory;
