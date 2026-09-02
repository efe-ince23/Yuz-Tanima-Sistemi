import {
  CalendarDays,
  Camera,
  ChevronLeft,
  ChevronRight,
  CircleUserRound,
  Clapperboard,
  Clock3,
  FilterX,
  LoaderCircle,
  Play,
  Search,
  ShieldCheck,
  UserRoundSearch,
  X,
} from "lucide-react";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { searchAppearances } from "./api";
import type {
  AppearanceSearchItem,
  AppearanceSearchParams,
  AppearanceSearchResponse,
} from "./types";

const PAGE_SIZE = 25;

interface SearchForm {
  q: string;
  source_type: "all" | "photo" | "video";
  identity_status: "" | "known" | "anonymous";
  date_from: string;
  date_to: string;
  min_confidence: string;
  max_confidence: string;
  sort: "newest" | "oldest" | "confidence";
}

const EMPTY_FORM: SearchForm = {
  q: "",
  source_type: "all",
  identity_status: "",
  date_from: "",
  date_to: "",
  min_confidence: "",
  max_confidence: "",
  sort: "newest",
};

function toIso(value: string, endOfDay = false): string | undefined {
  if (!value) return undefined;
  const time = endOfDay ? "T23:59:59" : "T00:00:00";
  return new Date(`${value}${time}`).toISOString();
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatTime(milliseconds: number): string {
  const seconds = Math.max(0, milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function identityName(item: AppearanceSearchItem): string {
  if (item.status === "known") {
    return `${item.first_name ?? ""} ${item.last_name ?? ""}`.trim() || "İsim bilgisi yok";
  }
  return `Anonim · ${item.face_id.slice(0, 8)}`;
}

function AdvancedSearch({ isAdmin }: { isAdmin: boolean }) {
  const [form, setForm] = useState<SearchForm>(EMPTY_FORM);
  const [applied, setApplied] = useState<SearchForm>(EMPTY_FORM);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<AppearanceSearchResponse | null>(null);
  const [selected, setSelected] = useState<AppearanceSearchItem | null>(null);
  const [seekToMs, setSeekToMs] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const params = useMemo<AppearanceSearchParams>(() => ({
    q: applied.q.trim() || undefined,
    source_type: applied.source_type,
    identity_status: applied.identity_status || undefined,
    date_from: toIso(applied.date_from),
    date_to: toIso(applied.date_to, true),
    min_confidence: applied.min_confidence === "" ? undefined : Number(applied.min_confidence) / 100,
    max_confidence: applied.max_confidence === "" ? undefined : Number(applied.max_confidence) / 100,
    sort: applied.sort,
    limit: PAGE_SIZE,
    offset,
  }), [applied, offset]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    searchAppearances(params)
      .then((response) => { if (active) setData(response); })
      .catch((reason: Error) => { if (active) setError(reason.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [params]);

  useEffect(() => {
    if (!selected || selected.source_type !== "video") return;
    const video = videoRef.current;
    if (!video) return;
    const seek = () => {
      video.currentTime = seekToMs / 1000;
      void video.play().catch(() => undefined);
    };
    if (video.readyState >= 1) seek();
    else video.addEventListener("loadedmetadata", seek, { once: true });
  }, [selected, seekToMs]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setOffset(0);
    setApplied({ ...form });
  };

  const reset = () => {
    setForm(EMPTY_FORM);
    setApplied(EMPTY_FORM);
    setOffset(0);
  };

  const openItem = (item: AppearanceSearchItem, startMs = item.first_seen_ms ?? 0) => {
    setSeekToMs(startMs);
    setSelected(item);
  };

  return (
    <section className="advanced-search-page">
      <div className="page-heading advanced-search-heading">
        <div>
          <p className="eyebrow">İŞLEM GEÇMİŞİ</p>
          <h1>Gelişmiş Arama</h1>
          <p className="advanced-search-description">Fotoğraf ve videolardaki yüz görünümlerini tek ekrandan inceleyin.</p>
        </div>
        <div className="search-scope-note"><ShieldCheck size={18} />Yalnızca kendi kayıtlarınız</div>
      </div>

      <form className="advanced-filter-panel" onSubmit={submit}>
        <label className="advanced-search-query">
          <span>Kişi veya face ID</span>
          <div><Search size={18} /><input value={form.q} onChange={(event) => setForm({ ...form, q: event.target.value })} placeholder="Ad, soyad veya face ID" /></div>
        </label>
        <div className="advanced-source-control" aria-label="Kaynak türü">
          {(["all", "photo", "video"] as const).map((source) => (
            <button key={source} className={form.source_type === source ? "active" : ""} type="button" onClick={() => setForm({ ...form, source_type: source })}>
              {source === "all" ? "Tümü" : source === "photo" ? "Fotoğraf" : "Video"}
            </button>
          ))}
        </div>
        <label><span>Kimlik durumu</span><select value={form.identity_status} onChange={(event) => setForm({ ...form, identity_status: event.target.value as SearchForm["identity_status"] })}><option value="">Tümü</option><option value="known">Bilinen</option><option value="anonymous">Anonim</option></select></label>
        <label><span>Başlangıç tarihi</span><input type="date" value={form.date_from} onChange={(event) => setForm({ ...form, date_from: event.target.value })} /></label>
        <label><span>Bitiş tarihi</span><input type="date" value={form.date_to} onChange={(event) => setForm({ ...form, date_to: event.target.value })} /></label>
        <label><span>En az güven (%)</span><input type="number" min="0" max="100" step="1" value={form.min_confidence} onChange={(event) => setForm({ ...form, min_confidence: event.target.value })} placeholder="0" /></label>
        <label><span>En fazla güven (%)</span><input type="number" min="0" max="100" step="1" value={form.max_confidence} onChange={(event) => setForm({ ...form, max_confidence: event.target.value })} placeholder="100" /></label>
        <label><span>Sıralama</span><select value={form.sort} onChange={(event) => setForm({ ...form, sort: event.target.value as SearchForm["sort"] })}><option value="newest">En yeni</option><option value="oldest">En eski</option><option value="confidence">En yüksek güven</option></select></label>
        <div className="advanced-filter-actions">
          <button className="secondary-button" type="button" onClick={reset}><FilterX size={17} />Temizle</button>
          <button className="primary-button" type="submit"><Search size={17} />Ara</button>
        </div>
      </form>

      <div className="advanced-results-heading">
        <div><h2>Görünme kayıtları</h2><span>{data?.total ?? 0} sonuç</span></div>
        {data && data.total > PAGE_SIZE && <div className="advanced-pagination"><button type="button" disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} title="Önceki sayfa"><ChevronLeft size={18} /></button><span>{Math.floor(offset / PAGE_SIZE) + 1} / {Math.ceil(data.total / PAGE_SIZE)}</span><button type="button" disabled={offset + PAGE_SIZE >= data.total || loading} onClick={() => setOffset(offset + PAGE_SIZE)} title="Sonraki sayfa"><ChevronRight size={18} /></button></div>}
      </div>

      <div className="advanced-results-list" aria-live="polite">
        {loading ? <div className="advanced-empty"><LoaderCircle className="spin" size={30} /><strong>Kayıtlar aranıyor</strong></div> : error ? <div className="advanced-empty error"><CircleUserRound size={32} /><strong>Arama tamamlanamadı</strong><span>{error}</span></div> : !data?.items.length ? <div className="advanced-empty"><UserRoundSearch size={32} /><strong>Eşleşen kayıt bulunamadı</strong><span>Filtreleri değiştirerek yeniden deneyin.</span></div> : data.items.map((item) => (
          <article className="advanced-result-row" key={`${item.source_type}-${item.process_id}-${item.face_id}`}>
            <button className="advanced-result-main" type="button" onClick={() => openItem(item)}>
              <span className="advanced-result-preview">{item.preview_url ? <img src={item.preview_url} alt="" /> : item.source_type === "video" ? <Clapperboard size={22} /> : <Camera size={22} />}</span>
              <span className="advanced-result-identity"><small>{item.status === "known" ? "Bilinen kişi" : "Anonim kimlik"}</small><strong>{identityName(item)}</strong><em title={item.face_id}>{item.face_id.slice(0, 8)} · {item.observation_count} gözlem</em></span>
              <span className="advanced-result-source">{item.source_type === "video" ? <Clapperboard size={16} /> : <Camera size={16} />}<strong>{item.source_type === "video" ? "Video" : "Fotoğraf"}</strong><small>{item.original_filename ?? "Dosya adı yok"}</small></span>
              <span className="advanced-result-date"><CalendarDays size={16} /><strong>{formatDate(item.occurred_at)}</strong></span>
              <span className="advanced-result-score"><small>Güven</small><strong>{item.confidence === null ? "-" : `%${(item.confidence * 100).toFixed(1)}`}</strong></span>
            </button>
            {item.source_type === "video" && item.intervals.length > 0 && <div className="advanced-result-intervals"><Clock3 size={15} />{item.intervals.map((interval, index) => <button key={`${interval.start_ms}-${index}`} type="button" onClick={() => openItem(item, interval.start_ms)}><Play size={12} />{formatTime(interval.start_ms)} - {formatTime(interval.end_ms)}</button>)}</div>}
          </article>
        ))}
      </div>

      {selected && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setSelected(null); }}><section className="modal-dialog advanced-detail-dialog" role="dialog" aria-modal="true" aria-label="Görünme detayı"><header><div><small>{selected.source_type === "video" ? "Video görünümü" : "Fotoğraf görünümü"}</small><h2>{identityName(selected)}</h2></div><button className="icon-button" type="button" onClick={() => setSelected(null)} title="Kapat"><X size={20} /></button></header><div className="advanced-detail-media">{selected.source_type === "video" ? <video ref={videoRef} src={selected.content_url} controls playsInline /> : <img src={selected.content_url} alt={identityName(selected)} />}</div><div className="advanced-detail-meta"><div><span>Kaynak</span><strong>{selected.original_filename ?? "Dosya adı yok"}</strong></div><div><span>Tarih</span><strong>{formatDate(selected.occurred_at)}</strong></div><div><span>Face ID</span><strong>{selected.face_id}</strong></div><div><span>Güven</span><strong>{selected.confidence === null ? "-" : `%${(selected.confidence * 100).toFixed(1)}`}</strong></div></div>{selected.source_type === "video" && selected.intervals.length > 0 && <div className="advanced-detail-intervals"><span>Görünme zamanları</span><div>{selected.intervals.map((interval, index) => <button key={`${interval.start_ms}-${index}`} type="button" className={seekToMs === interval.start_ms ? "active" : ""} onClick={() => { setSeekToMs(interval.start_ms); if (videoRef.current) { videoRef.current.currentTime = interval.start_ms / 1000; void videoRef.current.play().catch(() => undefined); } }}><Play size={13} />{formatTime(interval.start_ms)} - {formatTime(interval.end_ms)}</button>)}</div></div>}{isAdmin && <div className="advanced-process-id"><span>Process ID</span><strong>{selected.process_id}</strong></div>}</section></div>}
    </section>
  );
}

export default AdvancedSearch;
