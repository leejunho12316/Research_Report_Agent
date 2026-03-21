# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Server

```bash
uvicorn main:app --reload
```

The frontend (`index.html`) must be opened separately in a browser (e.g., via Live Server or `file://`). It hardcodes `http://127.0.0.1:8000` as the backend URL.

## Dependencies

No `requirements.txt` exists. Required packages:

```
fastapi
uvicorn
pypdf
python-multipart
```

## Architecture

This is a PDF text-extraction service with an async background-task pattern.

**Request flow:**
1. Browser (`index.html`) uploads a PDF to `POST /upload/`
2. Server saves the file to `uploads/`, inserts a DB row with `status='processing'`, and immediately returns a `task_id`
3. FastAPI `BackgroundTasks` runs `process_pdf_to_file()` (in `pdf_processor.py`) asynchronously — extracts text via `pypdf`, writes it to `outputs/result_<uuid>.txt`, then updates the DB row to `status='completed'` with `result_path`
4. Frontend polls `GET /status/{task_id}` every 2 seconds until status is `completed` or `failed`

**Storage:**
- `uploads/` — incoming PDF files
- `outputs/` — extracted text files (`.txt`)
- `app.db` — SQLite DB with a single `files` table (`id`, `original_name`, `result_path`, `status`)

**Important:** `init_db()` is called on every startup and **drops and recreates** the `files` table, clearing all previous task records.
