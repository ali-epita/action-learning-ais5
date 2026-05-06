#!/usr/bin/env python3
"""Download all bibliography PDFs into a `Papers/` directory.

Re-running is safe: existing files are skipped.

    uv run python scripts/download_papers.py            # → ./Papers
    uv run python scripts/download_papers.py --out ../Papers   # parent dir
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (compatible; AIS5-bibliography-fetcher/1.0)"
TIMEOUT = 60
RETRIES = 3
RETRY_BACKOFF = 4

# (number, type, slug, url, title) — matches the AIS 5 bibliography Excel.
PAPERS: list[tuple[int, str, str, str, str]] = [
    (1, "Core", "ferret_ui_lite", "https://arxiv.org/pdf/2509.26539.pdf", "Ferret-UI Lite"),
    (2, "Core", "os_atlas", "https://arxiv.org/pdf/2410.23218.pdf", "OS-Atlas"),
    (3, "Core", "showui", "https://arxiv.org/pdf/2411.17465.pdf", "ShowUI"),
    (4, "Core", "ui_tars_2", "https://arxiv.org/pdf/2509.02544.pdf", "UI-TARS-2"),
    (5, "Core", "qwen2_5_vl", "https://arxiv.org/pdf/2502.13923.pdf", "Qwen2.5-VL"),
    (6, "Core", "paligemma", "https://arxiv.org/pdf/2407.07726.pdf", "PaliGemma"),
    (7, "Core", "gta1", "https://arxiv.org/pdf/2507.05791.pdf", "GTA1"),
    (8, "Core", "lora", "https://arxiv.org/pdf/2106.09685.pdf", "LoRA"),
    (9, "Core", "screenspot_pro", "https://arxiv.org/pdf/2504.07981.pdf", "ScreenSpot-Pro"),
    (10, "Core", "uground", "https://arxiv.org/pdf/2410.05243.pdf", "UGround"),
    (11, "Supp", "seeclick", "https://arxiv.org/pdf/2401.10935.pdf", "SeeClick"),
    (12, "Supp", "cogagent", "https://arxiv.org/pdf/2312.08914.pdf", "CogAgent"),
    (13, "Supp", "ui_tars", "https://arxiv.org/pdf/2501.12326.pdf", "UI-TARS"),
    (14, "Supp", "ferret_ui", "https://arxiv.org/pdf/2404.05719.pdf", "Ferret-UI"),
    (15, "Supp", "internvl", "https://arxiv.org/pdf/2312.14238.pdf", "InternVL"),
    (16, "Supp", "jedi_osworld_g", "https://arxiv.org/pdf/2505.13227.pdf", "Jedi / OSWorld-G"),
    (17, "Supp", "osworld", "https://arxiv.org/pdf/2404.07972.pdf", "OSWorld"),
    (18, "Supp", "androidworld", "https://arxiv.org/pdf/2405.14573.pdf", "AndroidWorld"),
    (
        19,
        "Supp",
        "llava_next",
        "https://llava-vl.github.io/blog/2024-01-30-llava-next/",
        "LLaVA-NeXT (blog)",
    ),
    (
        20,
        "Supp",
        "slm_future_agentic_ai",
        "https://arxiv.org/pdf/2506.02153.pdf",
        "SLMs Are the Future of Agentic AI",
    ),
]


def filename_for(num: int, kind: str, slug: str, url: str) -> str:
    ext = ".pdf" if url.endswith(".pdf") else ".html"
    return f"{num:02d}_{kind}_{slug}{ext}"


def download(url: str, dest: Path) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = "unknown"
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            if not data:
                last_err = "empty response"
                continue
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True, f"{len(data) / 1024:,.1f} KB"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break
    return False, last_err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("Papers"), help="Output directory")
    parser.add_argument("--sleep", type=float, default=1.0, help="Pause between downloads")
    args = parser.parse_args(argv)

    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(PAPERS)} papers into: {out}\n")

    ok = skipped = failed = 0
    failures: list[tuple[int, str, str]] = []

    for num, kind, slug, url, title in PAPERS:
        dest = out / filename_for(num, kind, slug, url)
        tag = f"[{num:02d}] {title:<35}"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{tag} -> exists, skipping ({dest.name})")
            skipped += 1
            continue
        print(f"{tag} -> downloading ...", end=" ", flush=True)
        success, msg = download(url, dest)
        if success:
            print(f"OK ({msg})")
            ok += 1
        else:
            print(f"FAILED — {msg}")
            failed += 1
            failures.append((num, title, msg))
        time.sleep(args.sleep)

    print(f"\nDone. downloaded={ok}  skipped={skipped}  failed={failed}")
    if failures:
        print("\nFailures (re-run to retry):")
        for num, title, msg in failures:
            print(f"  [{num:02d}] {title}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
