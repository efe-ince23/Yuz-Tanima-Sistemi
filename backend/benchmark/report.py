import csv
import html
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

from benchmark.metrics import ThresholdMetric, metric_to_dict


def _format_percent(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def write_reports(
    output_directory: Path,
    result: Dict[str, object],
    threshold_metrics: Iterable[ThresholdMetric],
) -> Mapping[str, str]:
    output_directory.mkdir(parents=True, exist_ok=False)
    json_path = output_directory / "report.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    threshold_path = output_directory / "thresholds.csv"
    threshold_rows = [metric_to_dict(metric) for metric in threshold_metrics]
    with threshold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(threshold_rows[0]) if threshold_rows else ["threshold"],
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    summary_path = output_directory / "summary.csv"
    summary_rows = _summary_rows(result)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    html_path = output_directory / "report.html"
    html_path.write_text(_render_html(result, summary_rows), encoding="utf-8")
    return {
        "json": str(json_path),
        "summary_csv": str(summary_path),
        "thresholds_csv": str(threshold_path),
        "html": str(html_path),
    }


def _summary_rows(result: Dict[str, object]):
    detection = result["detection"]
    verification = result["verification"]
    identification = result["identification"]
    return [
        {"metric": "images_processed", "value": detection["images_processed"]},
        {"metric": "single_face_rate", "value": detection["single_face_rate"]},
        {"metric": "verification_accuracy", "value": verification["configured"]["accuracy"]},
        {"metric": "verification_auc", "value": verification["roc_auc"]},
        {"metric": "best_threshold", "value": verification["recommended"]["threshold"]},
        {"metric": "rank1_accuracy", "value": identification["rank1_accuracy"]},
        {"metric": "unknown_rejection_rate", "value": identification["unknown_rejection_rate"]},
        {"metric": "detection_p95_ms", "value": detection["latency_ms"]["p95"]},
    ]


def _render_html(result: Dict[str, object], summary_rows) -> str:
    verification = result["verification"]
    configured = verification["configured"]
    recommended = verification["recommended"]
    performance_rows = "".join(
        "<tr>"
        f"<td>{row['batch_size']}</td>"
        f"<td>{row['images']}</td>"
        f"<td>{row['elapsed_ms']:.2f}</td>"
        f"<td>{row['images_per_second']:.2f}</td>"
        "</tr>"
        for row in result["performance"]["arcface_batches"]
    )
    summary_html = "".join(
        f"<tr><td>{html.escape(str(row['metric']))}</td><td>{html.escape(str(row['value']))}</td></tr>"
        for row in summary_rows
    )
    warnings = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in result.get("warnings", [])
    ) or "<li>Uyarı bulunmuyor.</li>"
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yüz Tanıma Benchmark Raporu</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #10231f; background: #f4f8f6; }}
    main {{ max-width: 1050px; margin: 0 auto; padding: 32px 20px 60px; }}
    h1, h2 {{ letter-spacing: 0; }}
    .meta {{ color: #557069; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .metric {{ background: white; border: 1px solid #d4e0dc; border-radius: 6px; padding: 16px; }}
    .metric strong {{ display: block; font-size: 26px; margin-top: 8px; }}
    section {{ margin-top: 30px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ text-align: left; border-bottom: 1px solid #dce5e2; padding: 10px; }}
    th {{ background: #eaf3f0; }}
  </style>
</head>
<body><main>
  <h1>Yüz Tanıma Benchmark Raporu</h1>
  <p class="meta">Benchmark ID: {html.escape(str(result['benchmark_id']))} · {html.escape(str(result['created_at']))}</p>
  <div class="grid">
    <div class="metric">Mevcut eşik doğruluğu<strong>{_format_percent(configured['accuracy'])}</strong></div>
    <div class="metric">Önerilen eşik<strong>{recommended['threshold']:.2f}</strong></div>
    <div class="metric">Rank-1 kimlik başarısı<strong>{_format_percent(result['identification']['rank1_accuracy'])}</strong></div>
    <div class="metric">Tek yüz tespit oranı<strong>{_format_percent(result['detection']['single_face_rate'])}</strong></div>
  </div>
  <section><h2>Özet</h2><table><tbody>{summary_html}</tbody></table></section>
  <section><h2>ArcFace Batch Performansı</h2>
    <table><thead><tr><th>Batch</th><th>Görüntü</th><th>Süre (ms)</th><th>Görüntü/sn</th></tr></thead>
    <tbody>{performance_rows}</tbody></table>
  </section>
  <section><h2>Uyarılar ve Yol Haritası</h2><ul>{warnings}</ul></section>
</main></body></html>"""
