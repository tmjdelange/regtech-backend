"""Import a JSON or CSV file of documents through the admin bulk endpoint.

    ADMIN_API_KEY must be set in the environment. It is only ever sent in the
    request header, never printed or written anywhere.

    python3 scripts/import_documents.py --dry-run data-import/export.json
    python3 scripts/import_documents.py data-import/export.json

Rows are matched on title + country: new rows are inserted, rows whose
content, category or source_url changed are updated (and become unverified
again), unchanged rows are skipped. Every imported row is unverified.
Standard library only, so it runs on any machine with Python 3.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE_URL = "https://api.thedataqualitycompany.com"
RESULTS = ("inserted", "updated", "skipped", "failed")


def _multipart(path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    content_type = "application/json" if path.suffix.lower() == ".json" else "text/csv"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _print_report(report: dict) -> None:
    verb = {"inserted": "would insert", "updated": "would update",
            "skipped": "would skip", "failed": "would fail"} if report["dry_run"] else \
           {result: result for result in RESULTS}
    print("DRY RUN: nothing was written.\n" if report["dry_run"] else "")
    for row in report["rows"]:
        detail = row["reason"] or ""
        if row["changed"]:
            detail = "changed: " + ", ".join(row["changed"])
        print(f"  [{row['row']:>3}] {verb[row['result']]:<12} {row['country'] or '-':<4} "
              f"{row['title'] or '(no title)'}" + (f"  ({detail})" if detail else ""))

    print("\nTotals: " + " / ".join(f"{report[r]} {r}" for r in RESULTS))
    print("\nPer country:")
    print(f"  {'country':<8}" + "".join(f"{r:>10}" for r in RESULTS))
    for country, counts in sorted(report["by_country"].items()):
        print(f"  {country:<8}" + "".join(f"{counts[r]:>10}" for r in RESULTS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be inserted, updated or skipped without writing")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    api_key = os.environ.get("ADMIN_API_KEY")
    if not api_key:
        print("ADMIN_API_KEY is not set.", file=sys.stderr)
        return 2

    body, content_type = _multipart(args.file)
    path = "/admin/documents/bulk/dry-run" if args.dry_run else "/admin/documents/bulk"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + path,
        data=body,
        method="POST",
        headers={"x-api-key": api_key, "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            report = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        if exc.code == 404 and args.dry_run:
            print("The backend doesn't have the dry-run route yet; deploy it first.", file=sys.stderr)
        return 1

    _print_report(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
