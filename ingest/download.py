"""Download the source documents that can be fetched automatically.

The OWASP PDFs must be downloaded by hand from a browser; see README.md.
"""

from pathlib import Path
import urllib.request

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "nist-ai-100-1.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "nist-ai-600-1.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
    "mitre-atlas.yaml": "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml",
}

USER_AGENT = "framework-compass/0.1"

MANUAL = ["owasp-llm-top10-2025.pdf", "owasp-agentic-threats.pdf"]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"already present: {name}")
            continue
        print(f"downloading {name} ...")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    missing = [name for name in MANUAL if not (RAW_DIR / name).exists()]
    if missing:
        print("\nStill needed (download by hand, see README.md):")
        for name in missing:
            print(f"  data/raw/{name}")


if __name__ == "__main__":
    main()
