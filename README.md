# regtech-backend

FastAPI service behind the regtech frontend. It runs with `docker compose`; configuration comes from `.env`.

## Environment variables for upload extraction

`POST /extract` reads a label and a lab report and returns values to pre-fill part of the UK checklist form. It needs:

| Variable | What it is |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key used for the extraction call. Never logged. |
| `EXTRACTION_MODEL` | Model ID to use, for example `claude-opus-5-5`. |

If either is missing, `/extract` answers 503 and everything else keeps working.

Limits (constants in `backend/extraction.py`): 4 label files (PNG, JPG or PDF) and 1 lab report (PDF), 10 MB and 10 pages per file. Each IP can run 5 extractions per hour, and 100 can run per day in total. The counters are kept in memory and reset when the backend restarts.

Uploaded files are processed in memory and never written to disk or logged.
