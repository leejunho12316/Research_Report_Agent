import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Any
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

RES_DIR = os.path.join(os.path.dirname(__file__), "res")
VECTORDB_PATH = os.path.join(RES_DIR, "vectordb")
#
# # ── 1. 저장된 질문-문서 쌍 로드 ───────────────────────────────────────────────
def load_query_doc_pairs():
    with open(os.path.join(RES_DIR, "queries.json"), encoding="utf-8") as f:
        queries: Dict[str, str] = json.load(f)

    with open(os.path.join(RES_DIR, "corpus.json"), encoding="utf-8") as f:
        corpus: Dict[str, str] = json.load(f)

    with open(os.path.join(RES_DIR, "relevant_docs.json"), encoding="utf-8") as f:
        relevant_docs_raw: Dict[str, List[str]] = json.load(f)

    relevant_docs: Dict[str, Set[str]] = {k: set(v) for k, v in relevant_docs_raw.items()}

    print(f"로드 완료 - 질문: {len(queries)}개, 문서: {len(corpus)}개")
    return queries, corpus, relevant_docs
#
#
# # ── 2. VectorDB 로드, Document 추출 ──────────────────────────────────────────────────
print("VectorDB 로드 중...")

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=VECTORDB_PATH,
    embedding_function=embedding,
    collection_name="multimodal_rag",
)

db_result = vectordb.get()
documents = [
    Document(page_content=text, metadata={"id": doc_id})
    for doc_id, text in zip(
        db_result["ids"],
        db_result["documents"]
    )
]
print(f"VectorDB에서 {len(documents)}개 문서 로드 완료")

#문서 content 전체로 content id를 찾는 함수.
document_mapping = {doc.page_content: doc.metadata["id"] for doc in documents}
def get_doc_id_by_content(content: str) -> Optional[str]:
    if content in document_mapping:
        return document_mapping[content]
    for doc_content, doc_id in document_mapping.items():
        if content in doc_content or doc_content in content:
            return doc_id
    return None



# # ── 3. 리트리버 생성 ──────────────────────────────────────────────────────────
# print('OpenAI 임베딩 리트리버 생성 중...')
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(documents, embeddings, collection_name="openai")
# embedding_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

# print("BM25 리트리버 생성 중...")
# bm25_retriever = BM25Retriever.from_documents(documents)
# bm25_retriever.k = 10

print('OpenAI Large 임베딩 리트리버 생성 중...')
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vectorstore = Chroma.from_documents(documents, embeddings, collection_name="openai_large")
openai_large_embedding_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

print("bge-m3 임베딩 리트리버 생성 중...")
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
vectorstore = Chroma.from_documents(documents, embeddings, collection_name="bge-m3")
bgem3_embedding_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})




#LangChain 0.9.X 버전 EnsembleRetriever 예토전생 class
class EnsembleRetriever:
    """BM25 + Dense retriever 앙상블. Reciprocal Rank Fusion(RRF) 알고리즘으로 점수 결합."""

    def __init__(self, retrievers: List, weights: List[float], c: int = 60, id_key: Optional[str] = None):
        assert len(retrievers) == len(weights), "retrievers와 weights 길이가 다릅니다."
        self.retrievers = retrievers
        self.weights = weights
        self.c = c
        self.id_key = id_key

    def _get_doc_id(self, doc: Document) -> str:
        if self.id_key and self.id_key in doc.metadata:
            return str(doc.metadata[self.id_key])
        return doc.page_content[:150]

    def _fuse_results(self, all_results: List[List[Document]]) -> List[Document]:
        scores: Dict[str, Dict[str, Any]] = {}
        for i, docs in enumerate(all_results):
            weight = self.weights[i]
            for rank, doc in enumerate(docs):
                key = self._get_doc_id(doc)
                score = weight / (rank + 1 + self.c)
                if key not in scores:
                    scores[key] = {"doc": doc, "score": score}
                else:
                    scores[key]["score"] += score
        return [item["doc"] for item in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]

    def invoke(self, query: str, k: int = 5) -> List[Document]:
        all_results = []
        for retriever in self.retrievers:
            if hasattr(retriever, "invoke"):
                docs = retriever.invoke(query)
            elif hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query)
            else:
                raise AttributeError(f"{retriever} does not support retrieval.")
            all_results.append(docs[:k])
        return self._fuse_results(all_results)[:k]


# print("앙상블 리트리버 생성 중...")
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, embedding_retriever],
#     weights=[0.5, 0.5],
# )

# #########Ensemble Retriever도 5개만 invoke 한다음 k=10일떄 구하는거 아님? Retriever로 10개 가져오도록 해야하는거 아님?


# # ── 4. 평가 메트릭 함수 ───────────────────────────────────────────────────────
#accuracy : 검색한 것들 중 맞은 것이 있는지 1 or 0
def calculate_accuracy_at_k(retrieved_docs: List[Document], answer_ids: Set[str], k: int) -> float:
    for doc in retrieved_docs[:k]:
        doc_id = doc.metadata.get("id") or get_doc_id_by_content(doc.page_content)
        if doc_id in answer_ids:
            return 1.0
    return 0.0

#precision : 맞은 개수 / 검색한 전체 개수
#정밀도. T라고 예측한 것 중 T인 것.
def calculate_precision_at_k(retrieved_docs: List[Document], answer_ids: Set[str], k: int) -> float:
    count = sum(
        1 for doc in retrieved_docs[:k]
        if (doc.metadata.get("id") or get_doc_id_by_content(doc.page_content)) in answer_ids
    )
    return count / len(retrieved_docs[:k])

#recall : 맞은 개수 / 정답 개수
#재현율. 실제 T인 것 중 맞춘 것.
def calculate_recall_at_k(retrieved_docs: List[Document], answer_ids: Set[str], k: int) -> float:
    found = sum(
        1 for doc in retrieved_docs[:k]
        if (doc.metadata.get("id") or get_doc_id_by_content(doc.page_content)) in answer_ids
    )
    return found / len(answer_ids)


def calculate_mrr_at_k(retrieved_docs: List[Document], answer_ids: Set[str], k: int) -> float:
    for i, doc in enumerate(retrieved_docs[:k]):
        doc_id = doc.metadata.get("id") or get_doc_id_by_content(doc.page_content)
        if doc_id in answer_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_at_k(retrieved_docs: List[Document], answer_ids: Set[str], k: int) -> float:
    top_k = retrieved_docs[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc in enumerate(top_k)
        if (doc.metadata.get("id") or get_doc_id_by_content(doc.page_content)) in answer_ids
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(answer_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def calculate_map_at_k(retrieved_docs: List[Document], answer_ids: Set[str], k: int) -> float:
    precisions = []
    relevant_count = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        doc_id = doc.metadata.get("id") or get_doc_id_by_content(doc.page_content)
        if doc_id in answer_ids:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return sum(precisions) / len(answer_ids) if precisions and answer_ids else 0.0


def calculate_all_metrics(retrieved_docs: List[Document], answer_ids: Set[str], k_values: Optional[List[int]] = None) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"Accuracy@{k}"] = calculate_accuracy_at_k(retrieved_docs, answer_ids, k)
        metrics[f"Precision@{k}"] = calculate_precision_at_k(retrieved_docs, answer_ids, k)
        metrics[f"Recall@{k}"] = calculate_recall_at_k(retrieved_docs, answer_ids, k)
        metrics[f"MRR@{k}"] = calculate_mrr_at_k(retrieved_docs, answer_ids, k)
        metrics[f"NDCG@{k}"] = calculate_ndcg_at_k(retrieved_docs, answer_ids, k)
        metrics[f"MAP@{k}"] = calculate_map_at_k(retrieved_docs, answer_ids, k)
    return metrics

#
# # ── 5. 리트리버 평가 함수 ─────────────────────────────────────────────────────
def evaluate_retriever(retriever, queries: Dict[str, str], relevant_pairs: Dict[str, Set[str]], name: str = "", k_values: Optional[List[int]] = None) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    print(f"\n{name} 리트리버 평가 중...")
    results = []

    # sample_queries = dict(list(queries.items())[:20])

    for query_id, query_text in tqdm(queries.items(), desc=name):

        retrieved_docs = retriever.invoke(query_text) #[Document, Document ,,, Document]
        answer_ids = relevant_pairs.get(query_id, set()) #정답 doc id

        #metrics : {'Accuracy@1': 0.0, 'Precision@1': 0.0, 'Recall@1': 0.0 ...
        metrics = calculate_all_metrics(retrieved_docs, answer_ids, k_values)
        print('print metrics')
        print(metrics['Accuracy@1'])
        print(metrics['Precision@1'])
        print(metrics['Recall@1'])
        print(metrics['MRR@1'])

        #질문 하나하나에 대한 metric 저장
        results.append({"query_id": query_id, "query": query_text, **metrics})

    #전체 질문에 대한 metric 평균 계산
    #query_id, query, Accuracy, Precision,,, 중 Metric 관련 칼럼만 선택
    df_results = pd.DataFrame(results)
    metric_cols = [c for c in df_results.columns if any(c.startswith(p) for p in ["Accuracy", "Precision", "Recall", "MRR", "NDCG", "MAP"])]
    avg_metrics = df_results[metric_cols].mean().to_dict()

    return avg_metrics




# ── 6. 검색 결과 비교 분석 ────────────────────────────────────────────────────
def analyze_all_retrievers(queries: Dict[str, str], corpus: Dict[str, str], relevant_pairs: Dict[str, Set[str]], num_samples: int = 5, top_k: int = 3):
    print(f"\n===== 리트리버 검색 결과 비교 분석 (샘플 {num_samples}개) =====\n")

    def find_doc_id(content: str) -> Optional[str]:
        for doc_id, doc_content in corpus.items():
            if content == doc_content or content in doc_content or doc_content in content:
                return doc_id
        return None

    retrievers = {"BM25": bm25_retriever, "임베딩": embedding_retriever, "앙상블": ensemble_retriever}

    for idx, query_id in enumerate(list(queries.keys())[:num_samples]):
        query_text = queries[query_id]
        answer_ids = relevant_pairs.get(query_id, set())

        print(f"\n{'='*80}")
        print(f"\n[질문 {idx+1}] {query_text}")
        print("--" * 20)

        print("\n정답 문서:")
        for doc_id in answer_ids:
            doc_text = corpus.get(doc_id, "문서를 찾을 수 없음")
            print(f"  문서 ID: {doc_id}")
            print(f"  내용: {doc_text[:150]}..." if len(doc_text) > 150 else f"  내용: {doc_text}")

        print("==" * 50)
        for name, retriever in retrievers.items():
            retrieved = retriever.invoke(query_text)
            print(f"\n{name} 리트리버 검색 결과 (상위 {top_k}개):")
            for i, doc in enumerate(retrieved[:top_k]):
                doc_id = doc.metadata.get("id") or find_doc_id(doc.page_content) or f"unknown_{i}"
                mark = "O" if doc_id in answer_ids else "X"
                print(f"  [{i+1}] {mark} 문서 ID: {doc_id}")
                print(f"      내용: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"      내용: {doc.page_content}")
                print("--" * 20)

            correct = sum(
                1 for doc in retrieved[:top_k]
                if (doc.metadata.get("id") or find_doc_id(doc.page_content)) in answer_ids
            )
            denom = min(top_k, len(retrieved))
            print(f"  정확도: {correct}/{denom} ({correct/denom:.2f})" if denom else "  정확도: N/A")

        print(f"\n{'='*80}")


# ── 7. 메인 실행 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries, corpus, relevant_pairs = load_query_doc_pairs()

    # 리트리버 평가 - retriever 객체 / 질문 dict ('q_0_0' : '질문') / 질문 : 문서 dict ('id' : 'id') / print에 쓰이는 이름
    print("\n리트리버 평가 시작...")
    # embedding_metrics = evaluate_retriever(embedding_retriever, queries, relevant_pairs, "임베딩")
    # bm25_metrics = evaluate_retriever(bm25_retriever, queries, relevant_pairs, "BM25")
    # ensemble_metrics = evaluate_retriever(ensemble_retriever, queries, relevant_pairs, "앙상블")

    bgem3_metrics = evaluate_retriever(bgem3_embedding_retriever, queries, relevant_pairs, "BGE-M3")
    openai_metrics = evaluate_retriever(openai_large_embedding_retriever, queries, relevant_pairs, "임베딩 OpenAI Large")

    # 결과 비교 및 저장
    # results_df = pd.DataFrame({"임베딩": embedding_metrics, "BM25": bm25_metrics, "앙상블": ensemble_metrics})
    results_df = pd.DataFrame({"BGE-m3": bgem3_metrics, "OpenAI (large)": openai_metrics})
    results_df = results_df.round(3)

    output_path = os.path.join(os.path.dirname(__file__), "heatmaps/second_result/retriever_comparison_results.csv")
    results_df.to_csv(output_path)
    print(f"\n평가 결과를 '{output_path}'에 저장했습니다.")

    # 실제 검색 결과 비교
    # analyze_all_retrievers(queries, corpus, relevant_pairs)