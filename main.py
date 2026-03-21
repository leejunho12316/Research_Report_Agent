# main.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sqlite3

from pdf_processor import process_pdf_to_file
from pdf_processor2 import process_pdf_to_vectordb

app = FastAPI()
# 프론트엔드에서 오는 요청을 허용해주는 CORS 설정 추가
# 테스트용이므로 모든 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)


# 1. DB 설정 (상태를 기록할 'status' 컬럼 추가)
def init_db():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # 만약 기존에 만든 테이블이 있다면 충돌할 수 있으니 삭제 후 다시 만듭니다 (테스트용)
    cursor.execute("DROP TABLE IF EXISTS files")
    cursor.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_name TEXT,
            result_path TEXT,
            status TEXT  -- 🌟 작업 상태 (processing, completed, failed)
        )
    """)
    conn.commit()
    conn.close()

init_db()


# 🌟 2. 백그라운드에서 조용히 실행될 전처리 함수
def process_in_background(task_id: int, file_path: str):
    try:
        # 무거운 전처리 작업 실행!
        result_path = process_pdf_to_vectordb(file_path)

        # 작업이 끝나면 DB 상태를 'completed'로 바꾸고 결과 주소를 저장
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE files SET status = 'completed', result_path = ? WHERE id = ?",
            (result_path, task_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # 에러가 나면 상태를 'failed'로 변경
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE files SET status = 'failed' WHERE id = ?", (task_id,))
        conn.commit()
        conn.close()


# 🌟 3. 업로드 API (번호표만 주고 바로 끝냄)
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print('flag1')

    # 일단 DB에 'processing(처리 중)' 상태로 기록하고 번호표(task_id) 발급
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO files (original_name, status) VALUES (?, 'processing')",
        (file.filename,)
    )
    conn.commit()
    task_id = cursor.lastrowid  # 발급된 번호표
    conn.close()

    # 백그라운드 작업 지시 (서버야, 뒤에서 이 함수 좀 실행해 줘!)
    background_tasks.add_task(process_in_background, task_id, file_path)

    # 사용자는 기다리지 않고 바로 응답을 받습니다.
    return {
        "message": "파일 업로드 완료! 전처리가 백그라운드에서 시작되었습니다.",
        "task_id": task_id
    }


# 🌟 4. 상태 확인 API (프론트엔드가 주기적으로 물어볼 주소)
@app.get("/status/{task_id}")
def get_status(task_id: int):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status, result_path FROM files WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "해당 작업 번호를 찾을 수 없습니다."}

    status, result_path = row
    return {
        "task_id": task_id,
        "status": status,
        "result_url": f"/{result_path}" if result_path else None
    }