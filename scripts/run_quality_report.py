from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts" / "quality-reports"


@dataclass
class CheckResult:
    name: str
    category: str
    status: str
    duration_seconds: float
    total: int
    passed: int
    failed: int
    skipped: int
    summary: str
    output: str


def run(command: Sequence[str], timeout: int = 900) -> tuple[int, str, float]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout.strip(), time.perf_counter() - started
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, f"{output}\nTest zaman asimina ugradi.", time.perf_counter() - started


def integer_match(pattern: str, output: str, default: int = 0) -> int:
    match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
    return int(match.group(1)) if match else default


def command_check(
    name: str,
    category: str,
    command: Sequence[str],
    parser,
    timeout: int = 900,
) -> CheckResult:
    code, output, duration = run(command, timeout)
    total, passed, failed, skipped, summary = parser(code, output)
    status = "passed" if code == 0 and failed == 0 else "failed"
    return CheckResult(
        name=name,
        category=category,
        status=status,
        duration_seconds=round(duration, 3),
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        summary=summary,
        output=output[-30000:],
    )


def parse_backend(code: int, output: str) -> tuple[int, int, int, int, str]:
    total = integer_match(r"Ran\s+(\d+)\s+tests?", output)
    failures = integer_match(r"failures=(\d+)", output)
    errors = integer_match(r"errors=(\d+)", output)
    skipped = integer_match(r"skipped=(\d+)", output)
    failed = failures + errors
    if code != 0 and failed == 0:
        failed = max(1, total)
    passed = max(0, total - failed - skipped)
    return total, passed, failed, skipped, f"{passed}/{total} backend testi basarili"


def parse_build(code: int, output: str) -> tuple[int, int, int, int, str]:
    passed = 1 if code == 0 else 0
    return 1, passed, 1 - passed, 0, "TypeScript kontrolu ve Vite uretim derlemesi"


def parse_playwright(code: int, output: str) -> tuple[int, int, int, int, str]:
    passed = integer_match(r"(\d+)\s+passed", output)
    failed = integer_match(r"(\d+)\s+failed", output)
    skipped = integer_match(r"(\d+)\s+skipped", output)
    total = passed + failed + skipped
    if code != 0 and failed == 0:
        failed = 1
        total = max(total, 1)
    return total, passed, failed, skipped, f"{passed}/{total} tarayici senaryosu basarili"


def health_check() -> CheckResult:
    started = time.perf_counter()
    targets = (
        ("Frontend", "http://localhost:3000/"),
        ("Backend API", "http://localhost:8000/health"),
        ("Qdrant", "http://localhost:6333/healthz"),
        ("MinIO", "http://localhost:9000/minio/health/live"),
        ("pgAdmin", "http://localhost:5050/"),
    )
    lines: list[str] = []
    passed = 0
    for name, url in targets:
        try:
            with urlopen(url, timeout=15) as response:
                ok = 200 <= response.status < 400
                lines.append(f"{name}: HTTP {response.status}")
                passed += int(ok)
        except (URLError, TimeoutError, OSError) as error:
            lines.append(f"{name}: ERISILEMEDI ({error})")
    code, database_output, _ = run(
        ["docker", "inspect", "yuz-tanima-database", "--format", "{{.State.Health.Status}}"],
        timeout=30,
    )
    database_ok = code == 0 and database_output.strip() == "healthy"
    lines.append(f"PostgreSQL: {database_output.strip() or 'ERISILEMEDI'}")
    passed += int(database_ok)
    total = len(targets) + 1
    failed = total - passed
    return CheckResult(
        name="Docker servis sagligi",
        category="Altyapi",
        status="passed" if failed == 0 else "failed",
        duration_seconds=round(time.perf_counter() - started, 3),
        total=total,
        passed=passed,
        failed=failed,
        skipped=0,
        summary=f"{passed}/{total} servis saglikli",
        output="\n".join(lines),
    )


def video_check() -> CheckResult:
    command = [
        "docker", "compose", "run", "--rm", "--no-deps", "backend", "python3", "-m",
        "benchmark.video_acceptance_run", "--manifest",
        "/artifacts/video-acceptance/regression-baseline.json",
    ]
    started = time.perf_counter()
    stop_code, stop_output, _ = run(
        ["docker", "compose", "stop", "backend"], timeout=120
    )
    if stop_code != 0:
        return CheckResult(
            name="Gercek video regresyonu",
            category="Yapay zeka",
            status="failed",
            duration_seconds=round(time.perf_counter() - started, 3),
            total=1,
            passed=0,
            failed=1,
            skipped=0,
            summary="Backend kontrollu olarak durdurulamadi",
            output=stop_output,
        )
    try:
        code, output, _ = run(command, timeout=1200)
    finally:
        start_code, start_output, _ = run(
            ["docker", "compose", "up", "-d", "backend"], timeout=180
        )
        restored = False
        if start_code == 0:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                try:
                    with urlopen("http://localhost:8000/health", timeout=10) as response:
                        if response.status == 200:
                            restored = True
                            break
                except (URLError, TimeoutError, OSError):
                    pass
                time.sleep(5)
        if not restored:
            code = code or 1
            output = f"{output}\n\nBackend geri acilamadi:\n{start_output}"
    duration = time.perf_counter() - started
    reports = sorted(
        (ROOT / "artifacts" / "video-acceptance" / "runs").glob("*/report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    total = passed = failed = 0
    summary = "Video regresyon raporu olusturulamadi"
    if reports:
        payload = json.loads(reports[0].read_text(encoding="utf-8"))
        values = payload.get("summary", {})
        total = int(values.get("executedCases", 0))
        passed = int(values.get("passedCases", 0))
        failed = int(values.get("failedCases", max(0, total - passed)))
        summary = f"{passed}/{total} gercek video senaryosu basarili"
        output = f"Kaynak rapor: {reports[0]}\n\n{output}"
    if code != 0 and failed == 0:
        failed = 1
        total = max(total, 1)
    return CheckResult(
        name="Gercek video regresyonu",
        category="Yapay zeka",
        status="passed" if code == 0 and failed == 0 else "failed",
        duration_seconds=round(duration, 3),
        total=total,
        passed=passed,
        failed=failed,
        skipped=0,
        summary=summary,
        output=output[-30000:],
    )


def latest_benchmark() -> dict | None:
    reports = sorted(
        (ROOT / "artifacts" / "benchmarks").glob("*/report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return None
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    return {
        "path": str(reports[0]),
        "benchmark_id": payload.get("benchmark_id"),
        "created_at": payload.get("created_at"),
        "metrics": payload.get("metrics", {}),
        "warnings": payload.get("warnings", []),
    }


def render_html(report: dict) -> str:
    checks = report["checks"]
    cards = "".join(
        f"""
        <article class="check {item['status']}">
          <div class="check-head"><div><span>{html.escape(item['category'])}</span><h3>{html.escape(item['name'])}</h3></div><b>{'BASARILI' if item['status'] == 'passed' else 'BASARISIZ'}</b></div>
          <p>{html.escape(item['summary'])}</p>
          <div class="counts"><strong>{item['passed']} basarili</strong><span>{item['failed']} basarisiz</span><span>{item['skipped']} atlandi</span><span>{item['duration_seconds']:.2f} sn</span></div>
          <details><summary>Teknik ciktiyi goster</summary><pre>{html.escape(item['output'])}</pre></details>
        </article>"""
        for item in checks
    )
    benchmark = report.get("latest_benchmark")
    benchmark_html = ""
    if benchmark:
        metrics = benchmark.get("metrics") or {}
        metric_rows = "".join(
            f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
            for key, value in metrics.items()
            if not isinstance(value, (dict, list))
        )
        benchmark_html = f"""
        <section><div class="section-title"><div><span>REFERANS OLCUM</span><h2>Son LFW benchmark sonucu</h2></div><small>{html.escape(str(benchmark.get('created_at') or ''))}</small></div>
        <div class="benchmark"><table>{metric_rows or '<tr><td>Ozet metrik bulunamadi</td><td>-</td></tr>'}</table><p>Bu bolum bugun yeniden kosulmadi; en son kalici benchmark sonucu referans olarak gosterildi.</p></div></section>"""
    status_text = "KALITE KAPISI GECTI" if report["status"] == "passed" else "KALITE KAPISI KALDI"
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Yuz Tanima Sistemi - Kalite Raporu</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f3f6f5;color:#17231f;font-family:Arial,sans-serif;letter-spacing:0}}header{{background:#142c25;color:white;padding:28px 5vw;border-bottom:5px solid #16856e}}header span,.section-title span{{font-size:12px;font-weight:800;color:#54cbb1}}h1{{margin:6px 0 8px;font-size:30px}}header p{{margin:0;color:#cfe0da}}main{{width:min(1180px,92vw);margin:28px auto 60px}}.gate{{display:grid;grid-template-columns:1.6fr repeat(4,1fr);border:1px solid #cad8d3;background:white}}.gate>div{{padding:20px;border-right:1px solid #dce5e2}}.gate>div:last-child{{border:0}}.gate small{{display:block;color:#61726c;margin-bottom:7px}}.gate strong{{font-size:22px}}.gate .passed strong{{color:#08775f}}.gate .failed strong{{color:#a93a31}}section{{margin-top:30px}}.section-title{{display:flex;justify-content:space-between;align-items:end;margin-bottom:12px}}h2{{margin:5px 0 0;font-size:22px}}.checks{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.check{{background:white;border:1px solid #cad8d3;border-left:5px solid #17836c;padding:18px}}.check.failed{{border-left-color:#b74439}}.check-head{{display:flex;justify-content:space-between;gap:16px}}.check-head span{{font-size:11px;color:#687a74;font-weight:bold}}h3{{margin:4px 0;font-size:18px}}.check-head b{{font-size:12px;color:#08775f}}.failed .check-head b{{color:#a93a31}}.check p{{color:#52645e}}.counts{{display:flex;gap:16px;flex-wrap:wrap;border-top:1px solid #e1e8e6;padding-top:12px;font-size:13px}}details{{margin-top:14px}}summary{{cursor:pointer;color:#166b5a;font-weight:bold}}pre{{overflow:auto;max-height:330px;background:#101b18;color:#dce9e5;padding:14px;font-size:12px;white-space:pre-wrap}}.benchmark{{background:white;border:1px solid #cad8d3;padding:18px}}table{{width:100%;border-collapse:collapse}}td{{padding:9px;border-bottom:1px solid #e2e9e7}}td:last-child{{font-weight:bold;text-align:right}}.benchmark p{{color:#65756f;font-size:13px}}footer{{margin-top:28px;color:#687a74;font-size:13px}}@media(max-width:800px){{.gate,.checks{{grid-template-columns:1fr}}.gate>div{{border-right:0;border-bottom:1px solid #dce5e2}}}}
    </style></head><body><header><span>OTOMATIK QA / CI RAPORU</span><h1>Yuz Tanima Sistemi</h1><p>{html.escape(report['created_at'])} tarihinde tekrarlanabilir kalite kontrolleriyle uretilmistir.</p></header><main>
    <section class="gate"><div class="{report['status']}"><small>Genel sonuc</small><strong>{status_text}</strong></div><div><small>Toplam kontrol</small><strong>{report['totals']['total']}</strong></div><div><small>Basarili</small><strong>{report['totals']['passed']}</strong></div><div><small>Basarisiz</small><strong>{report['totals']['failed']}</strong></div><div><small>Toplam sure</small><strong>{report['duration_seconds']:.1f} sn</strong></div></section>
    <section><div class="section-title"><div><span>KALITE KAPILARI</span><h2>Test ve dogrulama sonuclari</h2></div><small>Rapor ID: {html.escape(report['report_id'])}</small></div><div class="checks">{cards}</div></section>
    {benchmark_html}<footer>Kalici uygulama verileri korunmustur. Video regresyonu salt okunur, arayuz testleri gecici kayitlarini temizleyen senaryolarla calistirilmistir.</footer></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the project quality gates and write HTML/JSON reports.")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPORT_ROOT / report_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print("[1/5] Docker servis sagligi kontrol ediliyor...", flush=True)
    checks = [health_check()]
    print(f"      {checks[-1].summary}", flush=True)
    print("[2/5] Backend testleri calistiriliyor...", flush=True)
    checks.append(command_check(
        "Backend birim ve servis testleri", "Backend",
        ["docker", "compose", "exec", "-T", "backend", "python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        parse_backend,
    ))
    print(f"      {checks[-1].summary}", flush=True)
    print("[3/5] Frontend uretim derlemesi calistiriliyor...", flush=True)
    checks.append(command_check(
        "Frontend tip ve uretim derlemesi", "Frontend",
        ["docker", "compose", "exec", "-T", "frontend", "npm", "run", "build"],
        parse_build,
    ))
    print(f"      {checks[-1].summary}", flush=True)
    if not args.skip_e2e:
        print("[4/5] Playwright kullanici yolculuklari calistiriliyor...", flush=True)
        checks.append(command_check(
            "Playwright kullanici yolculuklari", "Arayuz",
            ["docker", "compose", "exec", "-T", "frontend", "npx", "playwright", "test", "--reporter=line"],
            parse_playwright,
            timeout=1200,
        ))
        print(f"      {checks[-1].summary}", flush=True)
    if not args.skip_video:
        print("[5/5] Izole gercek video regresyonu calistiriliyor...", flush=True)
        checks.append(video_check())
        print(f"      {checks[-1].summary}", flush=True)

    totals = {
        "total": sum(item.total for item in checks),
        "passed": sum(item.passed for item in checks),
        "failed": sum(item.failed for item in checks),
        "skipped": sum(item.skipped for item in checks),
    }
    report = {
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "passed" if all(item.status == "passed" for item in checks) else "failed",
        "duration_seconds": round(time.perf_counter() - started, 3),
        "totals": totals,
        "checks": [asdict(item) for item in checks],
        "latest_benchmark": latest_benchmark(),
    }
    json_path = output_dir / "report.json"
    html_path = output_dir / "report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(json_path), "html": str(html_path), "totals": totals}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
