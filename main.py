# main.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import sqlite3

from dotenv import load_dotenv
load_dotenv() # .env 파일 읽어서 환경변수로 등록

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
            status TEXT,
            progress TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# 🌟 2. 백그라운드에서 조용히 실행될 전처리 함수
def process_in_background(task_id: int, file_path: str):
    from pdf_processor import process_pdf_to_vectordb #함수가 실행될 때만 진행하는 지연 import

    try:
        # 무거운 전처리 작업 실행!
        result_path = process_pdf_to_vectordb(file_path, task_id=task_id)

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
    cursor.execute("SELECT status, result_path, progress FROM files WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "해당 작업 번호를 찾을 수 없습니다."}

    status, result_path, progress = row
    return {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "result_url": f"/{result_path}" if result_path else None
    }


# 5. 기존 처리 완료 폴더 목록 API
@app.get("/files/")
def list_files():
    data_root = os.path.join(os.path.splitdrive(os.getcwd())[0] + os.sep, "data")
    if not os.path.isdir(data_root):
        return {"files": []}

    result = []
    for name in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, name)
        if not os.path.isdir(folder):
            continue
        vectordb_path = os.path.join(folder, "vectordb")
        status = "completed" if os.path.isdir(vectordb_path) else "processing"
        result.append({"name": name, "status": status})

    return {"files": result}


# 6. 채팅 API — 해당 파일의 VectorDB를 로드하고 RAG 체인으로 답변 반환
class ChatRequest(BaseModel):
    file_name: str
    message: str

@app.post("/chat/")
def chat(request: ChatRequest):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    # from langchain.chains.combine_documents import create_stuff_documents_chain
    # from langchain.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain

    drive = os.path.splitdrive(os.getcwd())[0] + os.sep
    vectordb_path = os.path.join(drive, "data", request.file_name, "vectordb")

    if not os.path.isdir(vectordb_path):
        return {"error": f"VectorDB를 찾을 수 없습니다: {request.file_name}"}

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=vectordb_path,
        embedding_function=embedding,
        collection_name="multimodal_rag",
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    template = """당신은 PDF 리포트를 상세히 설명하는 AI 어시스턴트입니다.

1. 주어진 검색 결과를 바탕으로 질문에 대한 답변을 마크다운 문법으로 작성하세요.
2. 검색 결과 중 '## 이미지 콘텐츠'가 있다면 출처를 참고하여 답변 중간에 마크다운 이미지 태그로 포함하세요.
   예시) <img src="/data/파일명/fig/figure-1-1.jpg">
3. 수식은 $$로 감싼 LaTeX 형식으로 작성하세요.
4. 검색 결과에 없는 내용은 답변하지 마세요.

{context}

Question: {input}
Answer:"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    llm_prompt_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, llm_prompt_chain)

    result = qa_chain.invoke({"input": request.message})
    return {"answer": result["answer"]}
