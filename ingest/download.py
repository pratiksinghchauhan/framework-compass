"""Download the source documents that can be fetched automatically.

The OWASP PDFs must be downloaded by hand from a browser; see README.md.
"""

from pathlib import Path
import urllib.request

import yaml

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ATLAS_RELEASE = "2026.10"

SOURCES = {
    "nist-ai-100-1.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "nist-ai-600-1.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
    "mitre-atlas.yaml": (
        "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/"
        f"dist/v6/ATLAS-{ATLAS_RELEASE}.yaml"
    ),
}

USER_AGENT = "framework-compass/0.1"

MANUAL = ["owasp-llm-top10-2025.pdf", "owasp-agentic-threats.pdf"]


def check_pdf(data: bytes) -> None:
    if not data.startswith(b"%PDF-"):
        raise ValueError("not a PDF (missing %PDF- header; maybe an HTML error page)")
    if b"%%EOF" not in data[-2048:]:
        raise ValueError("PDF looks truncated (no %%EOF marker at the end)")


def check_atlas(data: bytes) -> None:
    doc = yaml.safe_load(data)
    if not isinstance(doc, dict) or "collection" not in doc:
        raise ValueError("not ATLAS v6 data (no 'collection' key)")
    version = str(doc["collection"].get("version"))
    if version != ATLAS_RELEASE:
        raise ValueError(f"expected ATLAS {ATLAS_RELEASE}, got {version}")


CHECKS = {".pdf": check_pdf, ".yaml": check_atlas}


def validate(path: Path) -> None:
    CHECKS[path.suffix](path.read_bytes())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW_DIR / name
        if target.exists():
            try:
                validate(target)
                print(f"already present: {name}")
                continue
            except Exception as error:
                print(f"replacing invalid or outdated {name}: {error}")
        print(f"downloading {name} ...")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        CHECKS[target.suffix](data)
        target.write_bytes(data)
    missing = [name for name in MANUAL if not (RAW_DIR / name).exists()]
    if missing:
        print("\nStill needed (download by hand, see README.md):")
        for name in missing:
            print(f"  data/raw/{name}")


if __name__ == "__main__":
    main()
