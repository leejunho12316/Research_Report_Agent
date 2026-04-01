# 진행
# 1.QA_evaluation_data_creator.py -> golden_set.csv -> 2.QA_evaluation.py -> evaluation_result.csv -> 3.evaluation_grader.py
#
# evaluation/golden_set.csv : 정답 QA 파일
#
# 아래 비교군 4가지의 vectordb를 만든다.
# 1. 그냥 pdf text 추출 -> chunking -> vectordb : 아무런 처리도 하지 않은 방식
# 2. refined_pages_OCI.json -> chunking -> vectordb : LLM 정제 페이지 방식
# 3. QA_result.json 단독 -> vectordb : QA 합성 데이터만 사용하는 방식
# 4. 2번 + QA_result.json : QA 합성 데이터 추가 방식
#
# golden_set의 question 칼럼을 전체 iteration하면서 LLM에게 질문한다. LLM은 vectordb를 검색한 context를 참고하여 대답한다.
# 통일 파라미터 : system prompt, temperature = 0, 답변은 단답으로.
#
# 답변 저장은 golden_set.csv에 새로운 칼럼을 추가하는 식으로 저장한다.

import os
import json

import fitz
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

load_dotenv()

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(EVAL_DIR, "20260320_OCI홀딩스 (010060_매수).pdf")
REFINED_PAGES_PATH = os.path.join(EVAL_DIR, "json/refined_pages_OCI.json")
QA_RESULT_PATH = os.path.join(EVAL_DIR, "json/QA_result.json")
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.csv")
EVAL_RESULT_PATH = os.path.join(EVAL_DIR, 'evaluation_result.csv')

# ---------------------------------------------------------------------------
# 공통 설정
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 주어진 문서를 참고하여 사용자의 질문에 단답형으로 답변하는 어시스턴트입니다.
반드시 아래 규칙을 따르십시오:

1. 주어진 context에 근거해 답변하십시오.
2. 답변은 반드시 단답형으로, 숫자·단어·짧은 구절 수준으로 작성하십시오.
3. 십억, GW, % 등 단위를 안다면 반드시 포함해주세요.
4. 원(KRW), 달러(USD) 등 통화 단위도 안다면 반드시 포함해주세요.
5. context에 답이 없으면 "알 수 없음"으로 답하십시오.
6. 주어진 context에 없는 내용을 만들거나 유추해 작성하지 마십시오.
7. 부연 설명 없이 답만 출력하십시오.
"""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5
LLM_MODEL = "gpt-4.1"


# ---------------------------------------------------------------------------
# 1. pdf text 추출
# ---------------------------------------------------------------------------

def pdf_to_page_txts(pdf_path: str, output_dir: str = "pages") -> None:

    os.makedirs(output_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:

        total_pages = doc.page_count

        for idx, page in enumerate(doc, start=1):
            text = page.get_text()    # 텍스트 추출
            filename = f"page_{idx}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

    print(f"{total_pages}개 페이지를 {output_dir} 폴더에 저장했습니다.")


# ---------------------------------------------------------------------------
# VectorDB 구축
# ---------------------------------------------------------------------------

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def build_plain_vectordb(pdf_path: str, embeddings: OpenAIEmbeddings) -> Chroma:
    """방법 1: 그냥 PDF 텍스트 추출 → chunking → vectordb"""
    docs = []
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "source": "plain"},
                ))

    splitter = _make_splitter()
    chunks = splitter.split_documents(docs)
    print(f"  [plain] 청크 수: {len(chunks)}")
    return Chroma.from_documents(chunks, embeddings, collection_name="plain_pdf")


def build_refined_vectordb(refined_pages: list[str], embeddings: OpenAIEmbeddings) -> Chroma:
    """방법 2: refined_pages_OCI.json → chunking → vectordb"""
    docs = []
    for page_num, text in enumerate(refined_pages, start=1):
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": "refined"},
            ))

    splitter = _make_splitter()
    chunks = splitter.split_documents(docs)
    print(f"  [refined] 청크 수: {len(chunks)}")
    return Chroma.from_documents(chunks, embeddings, collection_name="refined_pages")


def build_qa_only_vectordb(qa_results: list[str], embeddings: OpenAIEmbeddings) -> Chroma:
    """방법 3: QA_result.json 단독 → vectordb"""
    qa_docs = []
    for i, qa_text in enumerate(qa_results):
        if isinstance(qa_text, str) and qa_text.strip():
            qa_docs.append(Document(
                page_content=qa_text,
                metadata={"source": "qa_result", "index": i},
            ))

    print(f"  [qa_only] 청크 수: {len(qa_docs)}")
    return Chroma.from_documents(qa_docs, embeddings, collection_name="qa_only")


def build_refined_qa_vectordb(
    refined_pages: list[str],
    qa_results: list[str],
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    """방법 4: refined_pages + QA_result.json → vectordb"""
    docs = []

    splitter = _make_splitter()

    # refined pages
    for page_num, text in enumerate(refined_pages, start=1):
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": "refined"},
            ))

    refined_chunks = splitter.split_documents(docs)

    # QA 합성 데이터 (문서 단위로 그대로 추가 - 분할하지 않음)
    qa_docs = []
    for i, qa_text in enumerate(qa_results):
        if isinstance(qa_text, str) and qa_text.strip():
            qa_docs.append(Document(
                page_content=qa_text,
                metadata={"source": "qa_result", "index": i},
            ))

    all_chunks = refined_chunks + qa_docs
    print(f"  [refined+qa] 청크 수: {len(all_chunks)} (refined {len(refined_chunks)} + qa {len(qa_docs)})")
    return Chroma.from_documents(all_chunks, embeddings, collection_name="refined_qa")


# ---------------------------------------------------------------------------
# RAG 질의
# ---------------------------------------------------------------------------

def query_rag(vectordb: Chroma, question: str, llm: ChatOpenAI) -> str:
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    context_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Context]\n{context}\n\n[질문]\n{question}"},
    ]

    response = llm.invoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    # pdf text 추출
    output_folder = os.path.join(EVAL_DIR, "pdf_to_text")
    pdf_to_page_txts(PDF_PATH, output_folder)

    # 데이터 로드
    with open(REFINED_PAGES_PATH, encoding="utf-8") as f:
        refined_pages: list[str] = json.load(f)

    with open(QA_RESULT_PATH, encoding="utf-8") as f:
        qa_results: list[str] = json.load(f)

    golden_set = pd.read_csv(GOLDEN_SET_PATH)
    assert isinstance(golden_set, pd.DataFrame)
    print(f"golden_set 로드 완료: {len(golden_set)}개 질문")

    # 공통 LLM / Embedding 초기화
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # evaluation_result.csv가 이미 있으면 이어서 저장, 없으면 golden_set으로 시작
    if os.path.exists(EVAL_RESULT_PATH):
        result_df = pd.read_csv(EVAL_RESULT_PATH)
        assert isinstance(result_df, pd.DataFrame)
        print(f"기존 결과 파일 로드: {EVAL_RESULT_PATH}")
    else:
        result_df = golden_set.copy()

    # (VectorDB 이름, 칼럼명, 빌더) 목록
    jobs: list[tuple[str, Chroma]] = []

    if "answer_plain" not in result_df.columns:
        print("\n1. Plain PDF VectorDB 구축 중...")
        jobs.append(("answer_plain", build_plain_vectordb(PDF_PATH, embeddings)))

    if "answer_refined" not in result_df.columns:
        print("2. Refined Pages VectorDB 구축 중...")
        jobs.append(("answer_refined", build_refined_vectordb(refined_pages, embeddings)))

    if "answer_qa_only" not in result_df.columns:
        print("3. QA Only VectorDB 구축 중...")
        jobs.append(("answer_qa_only", build_qa_only_vectordb(qa_results, embeddings)))

    if "answer_refined_qa" not in result_df.columns:
        print("4. Refined + QA VectorDB 구축 중...")
        jobs.append(("answer_refined_qa", build_refined_qa_vectordb(refined_pages, qa_results, embeddings)))

    if not jobs:
        print("모든 칼럼이 이미 완료됐습니다.")
        return

    # 칼럼 하나씩 처리 → 즉시 저장
    for col_name, vectordb in jobs:
        print(f"\n[{col_name}] 질문 답변 시작...")
        answers = []
        for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc=col_name):
            answers.append(query_rag(vectordb, row["Question"], llm))
            time.sleep(10)

        result_df[col_name] = answers
        result_df.to_csv(EVAL_RESULT_PATH, index=False, encoding="utf-8-sig")
        print(f"  → {col_name} 저장 완료: {EVAL_RESULT_PATH}")


if __name__ == "__main__":
    main()
