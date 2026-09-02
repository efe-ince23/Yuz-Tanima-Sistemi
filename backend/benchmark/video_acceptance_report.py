import csv
import html
import json
from pathlib import Path
from typing import Dict, Mapping


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def _summary_rows(result: Mapping[str, object]):
    rows = []
    for case in result["cases"]:
        metrics = case.get("metrics", {})
        rows.append(
            {
                "case_id": case["id"],
                "status": case["status"],
                "filename": case.get("source", {}).get("filename", ""),
                "identity_recall": metrics.get("identityRecall", ""),
                "temporal_iou": metrics.get("temporalIou", ""),
                "anonymous_recall": metrics.get("anonymousRecall", ""),
                "anonymous_temporal_iou": metrics.get(
                    "anonymousTemporalIou", ""
                ),
                "unexpected_known": metrics.get("unexpectedKnownCount", ""),
                "anonymous_tracks": metrics.get("anonymousTrackCount", ""),
                "total_tracks": metrics.get("totalTrackCount", ""),
                "short_tracks": metrics.get("shortTrackCount", ""),
                "low_confidence_known": metrics.get(
                    "lowConfidenceKnownTrackCount", ""
                ),
                "fragmented_identities": metrics.get(
                    "fragmentedIdentityCount", ""
                ),
                "model_warmup_seconds": metrics.get("modelWarmupSeconds", ""),
                "processing_seconds": metrics.get("processingSeconds", ""),
                "end_to_end_seconds": metrics.get("endToEndSeconds", ""),
                "realtime_factor": metrics.get("realtimeFactor", ""),
                "error": case.get("error", ""),
            }
        )
    return rows


def write_video_acceptance_reports(
    output_directory: Path,
    result: Dict[str, object],
) -> Mapping[str, str]:
    output_directory.mkdir(parents=True, exist_ok=False)
    json_path = output_directory / "report.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = _summary_rows(result)
    csv_path = output_directory / "summary.csv"
    fields = [
        "case_id",
        "status",
        "filename",
        "identity_recall",
        "temporal_iou",
        "anonymous_recall",
        "anonymous_temporal_iou",
        "unexpected_known",
        "anonymous_tracks",
        "total_tracks",
        "short_tracks",
        "low_confidence_known",
        "fragmented_identities",
        "model_warmup_seconds",
        "processing_seconds",
        "end_to_end_seconds",
        "realtime_factor",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    html_path = output_directory / "report.html"
    html_path.write_text(_render_html(result), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }


def _render_case(case: Mapping[str, object]) -> str:
    status = str(case["status"])
    source = case.get("source", {})
    metrics = case.get("metrics", {})
    evaluation_window = case.get("evaluationWindow")
    checks = case.get("checks", [])
    track_diagnostics = case.get("trackDiagnostics", [])
    window_html = ""
    if evaluation_window:
        start_seconds = float(evaluation_window["startMs"]) / 1000
        end_seconds = float(evaluation_window["endMs"]) / 1000
        window_html = (
            '<p class="muted">Degerlendirme penceresi: '
            f"{start_seconds:.1f} - {end_seconds:.1f} sn</p>"
        )
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(check['name']))}</td>"
        f"<td>{html.escape(str(check['actual']))}</td>"
        f"<td>{html.escape(str(check['expected']))}</td>"
        f"<td class=\"{'pass' if check['passed'] else 'fail'}\">"
        f"{'Gecti' if check['passed'] else 'Kaldi'}</td>"
        "</tr>"
        for check in checks
    )
    if status == "error":
        content = f"<p class=\"error\">{html.escape(str(case.get('error', 'Bilinmeyen hata')))}</p>"
    else:
        realtime = metrics.get("realtimeFactor")
        track_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(track.get('trackId') or '-'))}</td>"
            f"<td>{html.escape(str(track.get('name') or 'Anonim'))}</td>"
            f"<td>{html.escape(str(track.get('status') or '-'))}</td>"
            f"<td>{float(track.get('startMs', 0)) / 1000:.1f} - "
            f"{float(track.get('endMs', 0)) / 1000:.1f} sn</td>"
            f"<td>{html.escape(str(track.get('observationCount') or '-'))}</td>"
            f"<td>{_percent(track['confidence']) if track.get('confidence') is not None else '-'}</td>"
            f"<td>{_percent(track['detectionConfidence']) if track.get('detectionConfidence') is not None else '-'}</td>"
            f"<td>{html.escape(', '.join(track.get('flags', [])) or '-')}</td>"
            "</tr>"
            for track in track_diagnostics
        )
        diagnostics_table = (
            "<h3>Track ayrintilari</h3>"
            "<div class=\"table-wrap\"><table><thead><tr>"
            "<th>Track</th><th>Kimlik</th><th>Durum</th><th>Zaman</th>"
            "<th>Gozlem</th><th>Eslesme</th><th>Tespit</th><th>Isaretler</th>"
            f"</tr></thead><tbody>{track_rows}</tbody></table></div>"
            if track_rows
            else "<p class=\"muted\">Track bulunamadi.</p>"
        )
        content = f"""
        <div class="metrics">
          <span>Kimlik yakalama<strong>{_percent(metrics.get('identityRecall', 0))}</strong></span>
          <span>Zaman IoU<strong>{_percent(metrics.get('temporalIou', 0))}</strong></span>
          <span>Anonim yakalama<strong>{_percent(metrics.get('anonymousRecall', 1))}</strong></span>
          <span>Anonim zaman IoU<strong>{_percent(metrics.get('anonymousTemporalIou', 1))}</strong></span>
          <span>Anonim iz<strong>{metrics.get('anonymousTrackCount', 0)}</strong></span>
          <span>Kisa iz<strong>{metrics.get('shortTrackCount', 0)}</strong></span>
          <span>Dusuk marj<strong>{metrics.get('lowConfidenceKnownTrackCount', 0)}</strong></span>
          <span>Parcalanmis kimlik<strong>{metrics.get('fragmentedIdentityCount', 0)}</strong></span>
          <span>Model hazirligi<strong>{metrics.get('modelWarmupSeconds', '-')} sn</strong></span>
          <span>Video analizi<strong>{metrics.get('processingSeconds', '-')} sn</strong></span>
          <span>Gercek zaman katsayisi<strong>{realtime if realtime is not None else '-'}</strong></span>
        </div>
        <table><thead><tr><th>Kontrol</th><th>Olculen</th><th>Beklenen</th><th>Sonuc</th></tr></thead>
        <tbody>{check_rows}</tbody></table>
        {diagnostics_table}
        """
    return f"""
    <section class="case {status}">
      <header><div><small>{html.escape(str(case['id']))}</small>
      <h2>{html.escape(str(source.get('filename') or source.get('processId') or 'Video'))}</h2></div>
      <b>{html.escape(status.upper())}</b></header>
      {window_html}
      {content}
    </section>
    """


def _render_html(result: Mapping[str, object]) -> str:
    summary = result["summary"]
    case_html = "".join(_render_case(case) for case in result["cases"])
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Kabul Testi</title>
  <style>
    body {{ margin:0; background:#f3f7f5; color:#10231f; font-family:Arial,sans-serif; }}
    main {{ max-width:1050px; margin:auto; padding:32px 20px 60px; }}
    h1,h2 {{ letter-spacing:0; }} .muted,small {{ color:#60766f; }}
    .overview,.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
    .overview span,.metrics span {{ background:#fff; border:1px solid #d5e1dd; padding:14px; }}
    .overview strong,.metrics strong {{ display:block; font-size:24px; margin-top:6px; }}
    .case {{ background:#fff; border:1px solid #d5e1dd; margin-top:18px; padding:18px; border-left:5px solid #147d69; }}
    .case.failed,.case.error {{ border-left-color:#b33b32; }}
    .case header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; }}
    .case h2 {{ margin:4px 0 14px; font-size:20px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; }}
    .table-wrap {{ overflow-x:auto; }} h3 {{ margin:24px 0 0; }}
    th,td {{ text-align:left; padding:9px; border-bottom:1px solid #e1e9e6; }}
    th {{ background:#edf4f1; }} .pass {{ color:#08725d; }} .fail,.error {{ color:#a52c25; }}
  </style>
</head>
<body><main>
  <h1>Video Kabul Testi</h1>
  <p class="muted">{html.escape(str(result['suiteName']))} | {html.escape(str(result['createdAt']))}</p>
  <div class="overview">
    <span>Genel sonuc<strong>{html.escape(str(result['status']).upper())}</strong></span>
    <span>Calistirilan<strong>{summary['executedCases']}</strong></span>
    <span>Basarili<strong>{summary['passedCases']}</strong></span>
    <span>Basarisiz<strong>{summary['failedCases']}</strong></span>
  </div>
  {case_html}
</main></body></html>"""
