import os
import json
import time
from tqdm import tqdm
from typing import Dict, List, Set
from openai import OpenAI
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()


# ── 2. 질문-문서 쌍 생성 ──────────────────────────────────────────────────────
def generate_query_doc_pairs(docs: List[Document], num_questions_per_doc: int = 2):
    """
    각 문서에 대한 질문과 정답 문서를 생성하는 함수

    Returns:
        queries: 질문 ID -> 질문 텍스트
        corpus: 문서 ID -> 문서 텍스트
        relevant_docs: 질문 ID -> 관련 문서 ID 집합
    """
    client = OpenAI()

    all_queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant_docs: Dict[str, Set[str]] = {}

    prompt_template = """\
다음은 참고할 내용입니다.

---------------------
{context_str}
---------------------

위 내용을 바탕으로 낼 수 있는 질문을 {num_questions_per_chunk}개 만들어주세요.
질문만 작성하고 실제 정답이나 보기 등은 작성하지 않습니다.

해당 질문은 본문을 볼 수 없다고 가정합니다.
따라서 '위 본문을 바탕으로~' 라는 식의 질문은 할 수 없습니다.

질문은 아래와 같은 형식으로 번호를 나열하여 생성하십시오.

1. (질문)
2. (질문)
"""

    for idx, doc in enumerate(tqdm(docs, desc="질문 생성")):

        time.sleep(7)

        doc_id = doc.metadata.get("id", f"doc_{idx}")
        corpus[doc_id] = doc.page_content

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates questions based on provided content."},
            {"role": "user", "content": prompt_template.format(
                context_str=doc.page_content,
                num_questions_per_chunk=num_questions_per_doc,
            )},
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
            )
            result = response.choices[0].message.content.strip().split("\n")

            questions = []
            for line in result:
                if line.strip():
                    parts = line.strip().split(". ", 1)
                    if len(parts) > 1:
                        questions.append(parts[1])
                    elif "?" in line:
                        questions.append(line)

            for q_idx, question in enumerate(questions):
                if question:
                    query_id = f"q_{idx}_{q_idx}"
                    all_queries[query_id] = question
                    relevant_docs.setdefault(query_id, set()).add(doc_id)

        except Exception as e:
            print(f"문서 {doc_id}에 대한 질문 생성 중 오류 발생: {e}")

    return all_queries, corpus, relevant_docs


# ── 3. 결과 저장 ──────────────────────────────────────────────────────────────
def save_query_doc_pairs(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, Set[str]]):
    """queries, corpus, relevant_docs를 JSON 파일로 저장"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    # Set은 JSON 직렬화 불가 → list로 변환
    relevant_docs_serializable = {k: list(v) for k, v in relevant_docs.items()}
    with open(os.path.join(OUTPUT_DIR, "relevant_docs.json"), "w", encoding="utf-8") as f:
        json.dump(relevant_docs_serializable, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUTPUT_DIR}")
    print(f"  - queries.json ({len(queries)}개 질문)")
    print(f"  - corpus.json ({len(corpus)}개 문서)")
    print(f"  - relevant_docs.json")


VECTORDB_PATH = os.path.join(os.path.dirname(__file__), "res", "vectordb")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "res")

# ── 4. 메인 실행 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    #1. vectordb 로딩
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=VECTORDB_PATH,
        embedding_function=embedding,
        collection_name="multimodal_rag",
    )

    print(vectordb.get())

    #2. VectorDB 요소 LangChain Documents 형식으로 전환
    db_result = vectordb.get()
    documents = [
        Document(page_content=text, metadata={"id": doc_id})
        for doc_id, text in zip(
            db_result["ids"],
            db_result["documents"]
        )
    ]
    print(f"VectorDB에서 {len(documents)}개 문서 로드 완료")

    #3. 질의응답 쌍 만들기
    queries, corpus, relevant_docs = generate_query_doc_pairs(documents)
    print(f"\n생성된 질문 수: {len(queries)}")

    print("\n생성된 질문 샘플:")
    for i, (qid, qtext) in enumerate(list(queries.items())[:3]):
        related = list(relevant_docs[qid])
        print(f"질문 {i+1}: {qtext}")
        print(f"  관련 문서 ID: {related[0]}")
        print(f"  문서 내용 일부: {corpus[related[0]][:100]}...\n")

    save_query_doc_pairs(queries, corpus, relevant_docs)