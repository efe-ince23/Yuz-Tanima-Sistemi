import {
  Activity,
  CheckCircle2,
  Clock3,
  LoaderCircle,
  Percent,
  RefreshCw,
  UserCheck,
  UserX,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { getStatistics } from "./api";
import type { RecognitionStatistics } from "./types";

function formatEventTime(value: string | null): string {
  if (!value) return "Henüz işlem yok";
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function StatisticsDashboard() {
  const [statistics, setStatistics] = useState<RecognitionStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatistics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setStatistics(await getStatistics());
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "İstatistikler alınamadı.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadStatistics();
  }, [loadStatistics]);

  const recognizedRate = statistics?.success_rate ?? 0;
  const unrecognizedRate = statistics?.total_operations
    ? Math.max(0, 100 - recognizedRate)
    : 0;

  return (
    <>
      <section className="statistics-heading">
        <div>
          <p className="eyebrow">Tanıma performansı</p>
          <h1>İstatistikler</h1>
          <p><Clock3 size={14} /> Son işlem: {formatEventTime(statistics?.latest_event_at ?? null)}</p>
        </div>
        <button className="icon-button" type="button" onClick={() => void loadStatistics()} disabled={loading} title="İstatistikleri yenile">
          <RefreshCw className={loading ? "spin" : ""} size={18} />
        </button>
      </section>

      {loading && !statistics ? (
        <div className="statistics-state"><LoaderCircle className="spin" size={28} /><span>İstatistikler yükleniyor</span></div>
      ) : error ? (
        <div className="statistics-state error"><Activity size={28} /><strong>Veriler alınamadı</strong><span>{error}</span><button className="secondary-button no-margin" type="button" onClick={() => void loadStatistics()}>Tekrar dene</button></div>
      ) : statistics ? (
        <div className="statistics-content" aria-live="polite">
          <section className="metric-grid" aria-label="Tanıma özeti">
            <article className="metric-card total">
              <span><Activity size={21} /></span>
              <div><small>Toplam işlem</small><strong>{statistics.total_operations}</strong></div>
            </article>
            <article className="metric-card recognized">
              <span><UserCheck size={21} /></span>
              <div><small>Tanınan</small><strong>{statistics.recognized_count}</strong></div>
            </article>
            <article className="metric-card unrecognized">
              <span><UserX size={21} /></span>
              <div><small>Tanınmayan</small><strong>{statistics.unrecognized_count}</strong></div>
            </article>
            <article className="metric-card rate">
              <span><Percent size={21} /></span>
              <div><small>Başarı oranı</small><strong>%{statistics.success_rate.toFixed(2)}</strong></div>
            </article>
          </section>

          <section className="distribution-panel">
            <div className="distribution-heading">
              <div><p className="eyebrow">Sonuç dağılımı</p><h2>Tanıma sonuçları</h2></div>
              <span>{statistics.total_operations} tamamlanan işlem</span>
            </div>

            {statistics.total_operations ? (
              <div className="distribution-body">
                <div className="distribution-track" aria-label={`Tanınan yüzde ${recognizedRate.toFixed(2)}, tanınmayan yüzde ${unrecognizedRate.toFixed(2)}`}>
                  <span className="recognized-segment" style={{ width: `${recognizedRate}%` }} />
                  <span className="unrecognized-segment" style={{ width: `${unrecognizedRate}%` }} />
                </div>
                <div className="distribution-legend">
                  <div><span className="legend-mark recognized"><CheckCircle2 size={15} /></span><p>Tanınan<strong>{statistics.recognized_count} · %{recognizedRate.toFixed(2)}</strong></p></div>
                  <div><span className="legend-mark unrecognized"><UserX size={15} /></span><p>Tanınmayan<strong>{statistics.unrecognized_count} · %{unrecognizedRate.toFixed(2)}</strong></p></div>
                </div>
              </div>
            ) : (
              <div className="statistics-empty"><Activity size={30} /><strong>Henüz tanıma işlemi yok</strong></div>
            )}
          </section>
        </div>
      ) : null}
    </>
  );
}

export default StatisticsDashboard;
