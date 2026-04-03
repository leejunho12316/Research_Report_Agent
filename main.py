# main.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from contextlib import asynccontextmanager
import shutil
import os
import sqlite3

from dotenv import load_dotenv
load_dotenv() # .env 파일 읽어서 환경변수로 등록



# app.state : FastAPI 전역 상태 저장소로, 모든 요청에서 동일한 인스턴스를 사용할 수 있다.
# 하지만 아무래도 모델 다운로드가 느리기 때문에 bge-m3는 실제 서버 가동 시 사용. 테스트 때는 빠른 OpenAI Embedding text-embedding-3-large 사용.
# 실 서버 운영 시 파일 전처리 후 vectordb 구성도 bge-m3로 진행해주어야 함.
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # 서버 시작 시 BGE-m3 모델 로드
#     print("BGE-m3 임베딩 모델 로딩 중...")
#     from langchain_huggingface import HuggingFaceEmbeddings
#     app.state.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
#     print("BGE-m3 임베딩 모델 로딩 완료")
#     yield
#     # 서버 종료 시 정리 작업 (필요 시 추가)


app = FastAPI() #lifespan=lifespan
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

#FastAPI가 data 요청을 받으면 C:\data\에서 데이터를 찾도록 설정
from fastapi.staticfiles import StaticFiles

data_root = os.path.join(os.path.splitdrive(os.getcwd())[0] + os.sep, "data")
app.mount("/data", StaticFiles(directory=data_root), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


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


# # 6. 채팅 API — 해당 파일의 VectorDB를 로드하고 RAG 체인으로 답변 반환
# class ChatRequest(BaseModel):
#     file_name: str
#     message: str

# @app.post("/chat/")
# def chat(request: ChatRequest):
#     from langchain_chroma import Chroma
#     from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#     from langchain_core.prompts import PromptTemplate
#     # from langchain.chains.combine_documents import create_stuff_documents_chain
#     # from langchain.chains import create_retrieval_chain
#     from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#     from langchain_classic.chains import create_retrieval_chain
#
#     drive = os.path.splitdrive(os.getcwd())[0] + os.sep
#     vectordb_path = os.path.join(drive, "data", request.file_name, "vectordb")
#
#     if not os.path.isdir(vectordb_path):
#         return {"error": f"VectorDB를 찾을 수 없습니다: {request.file_name}"}
#
#     embedding = OpenAIEmbeddings(model = 'text-embedding-3-large')
#     vectordb = Chroma(
#         persist_directory=vectordb_path,
#         embedding_function=embedding,
#         collection_name="multimodal_rag",
#     )
#     retriever = vectordb.as_retriever(search_kwargs={"k": 4})
#
#     template = """당신은 PDF 리포트를 상세히 설명하는 AI 어시스턴트입니다.
#
# 1. 주어진 검색 결과를 바탕으로 질문에 대한 답변을 마크다운 문법으로 작성하세요.
# 2. 검색 결과 중 '## 이미지 콘텐츠'가 있다면 출처를 참고하여 답변 중간에 마크다운 이미지 태그로 포함하세요.
# 2-1. 마크다운 이미지 태그의 src 값 앞부분에는 "http://127.0.0.1:8000/"를 붙여주세요
#    예시) <img src="http://127.0.0.1:8000/data/파일명/fig/figure-1-1.jpg">
# 3. 수식은 반드시 $$로 감싼 LaTeX 형식으로 작성하세요.
# 3-1. 수식은 반드시 $$...$$ (display) 또는 $...$ (inline) 형식으로 작성하십시오. []를 사용해서는 안됩니다.
# 4. 검색 결과에 없는 내용은 답변하지 마세요.
#
# {context}
#
# Question: {input}
# Answer:"""
#
#     prompt = PromptTemplate.from_template(template)
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#
#     llm_prompt_chain = create_stuff_documents_chain(llm, prompt)
#     qa_chain = create_retrieval_chain(retriever, llm_prompt_chain)
#
#     result = qa_chain.invoke({"input": request.message})
#
#     print(f'llm raw result : {result}')
#
#     return {"answer": result["answer"]}


# 7. 파일 기반 대화 기록 클래스
import json
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class PdfFileChatHistory(BaseChatMessageHistory):
    """PDF 폴더별로 chat_history.json에 전체 기록 저장, LLM엔 최근 2쌍만 제공"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._all_messages: list[BaseMessage] = []
        self._load()

    @property
    def messages(self) -> list[BaseMessage]:
        return self._all_messages[-4:]  # LLM 컨텍스트: 최근 2쌍(4개)

    def add_message(self, message: BaseMessage) -> None:
        self._all_messages.append(message)
        self._save()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._all_messages.extend(messages)
        self._save()

    def clear(self) -> None:
        self._all_messages = []
        self._save()

    def _load(self):
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for msg in data:
            if msg['type'] == 'human':
                self._all_messages.append(HumanMessage(content=msg['content']))
            elif msg['type'] == 'ai':
                self._all_messages.append(AIMessage(content=msg['content']))

    def _save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        data = []
        for msg in self._all_messages:
            if isinstance(msg, HumanMessage):
                data.append({'type': 'human', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                data.append({'type': 'ai', 'content': msg.content})
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _get_history_path(file_name: str) -> str:
    drive = os.path.splitdrive(os.getcwd())[0] + os.sep
    return os.path.join(drive, "data", file_name, "chat_history.json")


# 8. 대화 기록 조회 API
@app.get("/chat_history/{file_name:path}")
def get_chat_history(file_name: str):
    history_path = _get_history_path(file_name)
    if not os.path.exists(history_path):
        return {"messages": []}
    with open(history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {"messages": data}


# 9. 채팅 API (파일 기반 대화 기록, 이전 2개 대화 참고)
class ChatRequest(BaseModel):
    file_name: str
    message: str

@app.post("/chat/")
def chat(request: ChatRequest):
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain

    drive = os.path.splitdrive(os.getcwd())[0] + os.sep
    vectordb_path = os.path.join(drive, "data", request.file_name, "vectordb")

    if not os.path.isdir(vectordb_path):
        return {"error": f"VectorDB를 찾을 수 없습니다: {request.file_name}"}

    #embedding = app.state.embedding
    embedding = OpenAIEmbeddings(model = 'text-embedding-3-large')
    vectordb = Chroma(
        persist_directory=vectordb_path,
        embedding_function=embedding,
        collection_name="multimodal_rag",
    )

    #k=10 정확도 0.9310
    #k=5 정확도 0.8560
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    template = """당신은 PDF 리포트를 상세히 설명하는 AI 어시스턴트입니다.

1. 주어진 검색 결과를 바탕으로 질문에 대한 답변을 마크다운 문법으로 작성하세요.
2. 검색 결과 중 '## 이미지 콘텐츠'가 있다면 출처를 참고하여 답변 중간에 마크다운 이미지 태그로 포함하세요.
2-1. 마크다운 이미지 태그의 src 값 앞부분에는 "http://127.0.0.1:8000/"를 붙여주세요
   예시) <img src="http://127.0.0.1:8000/data/파일명/fig/figure-1-1.jpg">
3. 수식은 반드시 $$로 감싼 LaTeX 형식으로 작성하세요.
3-1. 수식은 반드시 $$...$$ (display) 또는 $...$ (inline) 형식으로 작성하십시오. []를 사용해서는 안됩니다.
4. 검색 결과에 없는 내용은 답변하지 마세요.

{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm_prompt_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, llm_prompt_chain)

    def get_session_history(session_id: str) -> PdfFileChatHistory:
        return PdfFileChatHistory(_get_history_path(session_id))

    chain_with_history = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key='answer'
    )

    result = chain_with_history.invoke(
        {"input": request.message},
        config={"configurable": {"session_id": request.file_name}},
    )

    print(f'llm raw result : {result}')

    return {"answer": result["answer"]}
