"""Fetch a bounded, auditable human-preference pilot using only Python's stdlib.

No model runtime is imported. Only the official dataset's train prefix is read.
The rows API is not revision-addressed: metadata checks and the server's optional
x-revision header establish snapshot evidence, not immutable URL semantics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request


DATASET = "CarperAI/openai_summarize_comparisons"
EXPECTED_SHA = "a320fae2cd84f04610a1b57ac83c2a2f674b8bcd"
METADATA_URL = f"https://huggingface.co/api/datasets/{DATASET}"
ROWS_BASE = "https://datasets-server.huggingface.co/rows"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "raw" / "pilot_rows.jsonl"
PAGE_SIZE = 100
MAX_ROWS = 4600  # 46 pages + two metadata requests; leaves room for two probes.
MAX_REQUESTS = 48
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024


def stamp():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # A redirect must not silently add requests or destinations.


class Reader:
    def __init__(self):
        self.request_count = 0
        self.total_bytes = 0
        self.opener = urllib.request.build_opener(NoRedirects())

    def get(self, url):
        if self.request_count >= MAX_REQUESTS:
            raise RuntimeError("HTTP request budget exceeded")
        self.request_count += 1
        request = urllib.request.Request(url, headers={"User-Agent": "rlhf-lab-pilot/1.0"})
        with self.opener.open(request, timeout=30) as response:
            raw = response.read(MAX_PAGE_BYTES + 1)
            if len(raw) > MAX_PAGE_BYTES:
                raise RuntimeError("Single response exceeds 2 MiB bound")
            self.total_bytes += len(raw)
            if self.total_bytes > MAX_TOTAL_BYTES:
                raise RuntimeError("Total HTTP response data exceeds 20 MiB bound")
            record = {
                "url": url,
                "response_url": response.url,
                "retrieved_at_utc": stamp(),
                "raw_bytes": len(raw),
                "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
                "x_revision": response.headers.get("x-revision"),
                "etag": response.headers.get("ETag"),
                "response_date": response.headers.get("Date"),
            }
        return json.loads(raw), record


def fetch(rows):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"rows must be in 1..{MAX_ROWS}")
    provenance_path = Path(str(OUTPUT) + ".provenance.json")
    partial_path = Path(str(OUTPUT) + ".download-partial")
    for path in (OUTPUT, provenance_path, partial_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    reader = Reader()
    metadata_before, before_record = reader.get(METADATA_URL)
    if metadata_before.get("sha") != EXPECTED_SHA:
        raise RuntimeError("Current dataset SHA no longer matches the project's expected revision")
    before_record["dataset_sha"] = metadata_before["sha"]
    started_at = stamp()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pages, row_hashes = [], []
    output_hash = hashlib.sha256()
    output_bytes = 0
    actual_rows = 0
    with partial_path.open("xb") as handle:
        for offset in range(0, rows, PAGE_SIZE):
            length = min(PAGE_SIZE, rows - offset)
            url = ROWS_BASE + "?" + urllib.parse.urlencode({
                "dataset": DATASET, "config": "default", "split": "train",
                "offset": offset, "length": length,
            })
            payload, page_record = reader.get(url)
            revision = page_record["x_revision"]
            if revision is not None and revision != EXPECTED_SHA:
                raise RuntimeError("Rows API x-revision does not match expected dataset SHA")
            if payload.get("partial") is not False:
                raise RuntimeError("Rows API returned a partial/unknown response")
            page_rows = payload.get("rows")
            if not isinstance(page_rows, list) or len(page_rows) != length:
                raise RuntimeError("Rows API did not return the complete requested page")
            for expected_idx, wrapped in enumerate(page_rows, start=offset):
                if wrapped.get("row_idx") != expected_idx or wrapped.get("truncated_cells") != []:
                    raise RuntimeError("Row indices or untruncated-cell guarantees failed")
                row = wrapped.get("row")
                if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in ("prompt", "chosen", "rejected")):
                    raise RuntimeError("Unexpected preference row schema")
                row = {key: row[key] for key in ("prompt", "chosen", "rejected")}
                row_text = canonical(row).encode("utf-8")
                line = row_text + b"\n"
                output_bytes += len(line)
                if output_bytes > MAX_TOTAL_BYTES:
                    raise RuntimeError("JSONL output exceeds 20 MiB bound")
                handle.write(line)
                output_hash.update(line)
                row_hashes.append({"source_row_idx": expected_idx, "canonical_row_sha256": hashlib.sha256(row_text).hexdigest()})
                actual_rows += 1
            page_record.update({"offset": offset, "length": length, "num_rows_total": payload.get("num_rows_total")})
            pages.append(page_record)
            print(f"Fetched {actual_rows}/{rows} real preference rows; {output_bytes} JSONL bytes", flush=True)
    metadata_after, after_record = reader.get(METADATA_URL)
    if metadata_after.get("sha") != EXPECTED_SHA:
        raise RuntimeError("Dataset changed during retrieval; partial artifact is not ready for preparation")
    after_record["dataset_sha"] = metadata_after["sha"]
    declaration = {
        "schema_version": 1,
        "human_preferences": True,
        "synthetic": False,
        "origin": f"https://huggingface.co/datasets/{DATASET}",
        "original_research": "https://github.com/openai/summarize-from-feedback",
        "annotation_method": "Public OpenAI summarization human preference comparisons, repackaged by CarperAI as chosen/rejected; not newly annotated by this project.",
        "dataset_id": DATASET,
        "expected_dataset_sha": EXPECTED_SHA,
        "source_split": "train",
        "source_config": "default",
        "rows_api_is_revision_addressed": False,
        "revision_evidence": "Current repository metadata checked before and after retrieval; any returned x-revision header must match expected SHA. This is a captured API snapshot, not a claim of immutable revision-pinned response bytes.",
        "all_page_revision_headers_match": all(page["x_revision"] == EXPECTED_SHA for page in pages),
        "sampling": f"First {actual_rows} rows (offset 0 inclusive, {actual_rows} exclusive) of official train; bounded ordered prefix, not uniform sampling. No official validation/test rows downloaded.",
        "local_holdout_policy": "The data.prepare local adapter creates fresh prompt-grouped train/valid/test subdivisions within this official train prefix; they are not the official held-out dataset splits.",
        "started_at_utc": started_at,
        "completed_at_utc": stamp(),
        "requested_rows": rows,
        "actual_rows": actual_rows,
        "jsonl_bytes": output_bytes,
        "jsonl_sha256": output_hash.hexdigest(),
        "canonical_row_encoding": "UTF-8 JSON, sorted keys, no extra whitespace, ensure_ascii=False; row hash excludes final newline",
        "metadata_before": before_record,
        "metadata_after": after_record,
        "page_requests": pages,
        "row_hashes": row_hashes,
        "http_requests": reader.request_count,
        "http_response_bytes": reader.total_bytes,
        "limits": {"max_rows": MAX_ROWS, "max_http_requests": MAX_REQUESTS, "max_page_bytes": MAX_PAGE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES},
    }
    provenance_bytes = (json.dumps(declaration, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(provenance_bytes) > MAX_TOTAL_BYTES:
        raise RuntimeError("Provenance output exceeds 20 MiB bound")
    with provenance_path.open("xb") as handle:
        handle.write(provenance_bytes)
    partial_path.rename(OUTPUT)
    return {"output": str(OUTPUT), "provenance": str(provenance_path), "rows": actual_rows, "jsonl_bytes": output_bytes, "sha256": output_hash.hexdigest(), "http_requests": reader.request_count, "http_response_bytes": reader.total_bytes, "all_page_revision_headers_match": declaration["all_page_revision_headers_match"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--execute", action="store_true", help="Explicitly fetch the real human-preference data")
    args = parser.parse_args()
    if not 1 <= args.rows <= MAX_ROWS:
        parser.error(f"--rows must be in 1..{MAX_ROWS}")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "rows": args.rows, "output": str(OUTPUT), "source_split": "train", "expected_sha": EXPECTED_SHA, "will_download": False}, indent=2))
        return
    try:
        print(json.dumps(fetch(args.rows), indent=2))
    except Exception as exc:
        print(f"Fetch failed: {exc}. Any .download-partial file is incomplete and must not be used for training.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
