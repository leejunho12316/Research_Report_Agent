# 임베딩 모델에 따른 RAG 검색 성능 평가

서비스의 RAG 답변 품질의 향상을 위해 VectorDB를 구성하는 다양한 임베딩 모델에 따른 RAG 성능 평가를 진행하였다. 

비교군
- OpenAI text-embedding-ada-002 (기본 임베딩 모델)
- BM25 알고리즘
- Ensemble Retriever (OpenAI text-embedding-ada-002 : BM25 = 5 : 5)
- OpenAI text-embedding-3-large
- BGE-m3

### 데이터셋

소스 데이터 : https://securities.miraeasset.com/bbs/download/2143555.pdf?attachmentId=2143555

<20260402_신세계 (004170_매수).pdf>를 전처리하여 LLM 정제 데이터 + QA 합성 데이터 + 이미지 설명 데이터로 VectorDB를 구성하였다. 
구성된 Documents 하나마다 사용자가 AI 챗봇을 사용하며 실제 할 법한 질문을 LLM으로 2개씩 생성했다.


각 임베딩 모델, 알고리즘 방식으로 생성된 retriever에 모든 query 질문을 하여 k개의 검색 결과를 받아온다.
검색된 k개의 결과를 query와 연결된 정답 corpus와 비교하며 Accuracy, Precision, Recall, MRR, NDCG, MAP의 성능 Metric을 도출한다.

query, corpus 예시

```
"q_29_1": "신세계DF의 2025년과 2026년 연간 매출액 성장률을 비교해 보세요. 어떤 변화가 있었나요?",
"q_30_0": "신세계DF의 매출 성장률 추세는 어떻게 변했나요?",
... 188개
```

```
"c34a6511-0601-4621-8c7f-3af47e45848f": "### 좌측 테이블\n\n- **투자의견(유지)**: 매수\n- **목표주가(상향)**: 450,000원\n- **현재주가(26/4/1)**: 316,500원\n- **상승여력**: 42.2%\n\n#### 영업이익(26F, 십억원)\n- [컨센서스(영업이익/26F, 십억원)] 640\n\n#### EPS 성장률(26F, %)  \n- 1,904.7\n\n#### MKT EPS 성장률(26F, %)  \n- 136.0\n\n#### P/E(26F, x)  \n- 10.1\n\n#### MKT P/E(26F, x)  \n- 10.4\n\n#### KOSPI  \n- 5,478.70\n\n#### 시가총액(26/4/1, 억원)  \n- 3,053\n\n#### 동일업종 시가총액(26/4/1, 억원)  \n- 59,090\n\n#### 유통주식수(백만주)  \n- 6.1\n\n#### 외국인 보유비중(%)  \n- 20.7\n\n#### 배당(12M) 주가수익률(%)  \n- 1.90\n\n#### 52주 최저가(원)  \n- 135,400",
"3ab0be03-d8ec-4817-a650-966159175266": "#### 외국인 보유비중(%)  \n- 20.7\n\n#### 배당(12M) 주가수익률(%)  \n- 1.90\n\n#### 52주 최저가(원)  \n- 135,400\n\n#### 52주 최고가(원)  \n- 376,500\n\n#### (%)  \n- 절대주가(1M, 6M, 12M)  \n  - -14.1, 7.2, 13.2  \n- 상대주가  \n  - -2.1, 8.9, 6.9\n\n(좌 하단 그래프: 신세계, KOSPI 추이, 구체적 수치 생략)\n\n##### [작성자/투동/리서치]\n배송익\nsongyi.bae@miraeasset.com\n\n---\n\n### 우측 본문 및 표\n\n**004170 · 백화점  \n신세계  \n신세계 사기 딱 좋은 시점**",
... 94개
```

# 첫 번째 성능평가

먼저 가장 기본적인 임베딩 검색, BM25 검색, 앙상블 서치에 대한 성능평가를 진행했다.

<img src="./heatmaps/first_result/heatmap_k1.png">
<img src="./heatmaps/first_result/heatmap_k3.png">
<img src="./heatmaps/first_result/heatmap_k5.png">
<img src="./heatmaps/first_result/heatmap_k10.png">

### 지표별 해석
- Accuracy : 검색 문서 중 정답 문서가 포함되어 있는지 여부를 평가한 Metric. RAG 답변 성능에 가장 중요한 지표
- Precision : 검색 문서 개수 중 정답 문서가 포함되어 있는지 여부를 평가한 Metric. 검색 문서 개수가 많아질 수록 분모가 커져 값이 빠르게 줄어든다.
- Recall : 전체 정답 문서 개수 대비 검색한 문서에 포함된 정답 문서가 개수 Metric. 이 성능평가의 경우 정답 문서는 질문 당 하나이기 때문에 Accuracy 점수와 동일.
- MRR : 처음으로 정답 문서가 등장한 순위의 역수. 0.5일 경우 정답 문서가 2번째로 나오고 0.33일 경우 3번째로 나온다는 뜻.

- NDCG : 정답인 문서에 대해 log2(rank+1)합 DCG, 모든 문서가 정답일 때 log2(rank+1)합 Ideal DCG를 구해 DCG/Ideal DCG 값을 사용. 계산 방식은 다르지만 정답 문서의 위치를 고려한 Mertic이기 때문에 MRR값 변동을 따라감.
- MAP : 1위부터 문서의 위치까지 '정답 개수/전체 문서 개수'의 평균. 정답 문서가 빠르게 나올 수록 점수가 높아짐.
Mean Average Precision


### 결과 해석
k=1에서 Accuracy가 평균 0.4인데 이는 첫 번째 검색 결과로 10개 중 4개만 성공한다는 뜻이다. 보통 0.6 이상은 되어야 하기 때문에 서비스에 적합하지 않다고 할 수 있다.<br>
k=5에서 Accuracy가 0.7, MRR은 0.5이다. k값이 늘어남에 따라 검색 성능 자체는 양호하지만 정답 문서가 뒤로 밀려 있는 상태이다.<br>


### 해결 방법 
~~-> cross encoder(reranker 추가)~~ <br>
-> 큰 임베딩 모델 교체 bge-m3, OpenAI text-embedding-3-large


# 두 번쨰 성능평가 
BGE-m3모델과 OpenAI text-embedding-3-large 모델 사용해 성능 평가

<img src="./heatmaps/second_result/heatmap_k1.png">
<img src="./heatmaps/second_result/heatmap_k3.png">
<img src="./heatmaps/second_result/heatmap_k5.png">
<img src="./heatmaps/second_result/heatmap_k10.png">

### 결과 해석
k=5일 때 Accuracy 점수가 BGE-m3 모델은 0.87, OpenAI text-embedding-3-large는 0.85로 준수한 성능을 보이고 있음을 알 수 있다. MRR 점수도 모든 k값 범위에서 상승한 것을 볼 수 있다. <br>
k=10일 때 Accuracy 점수가 BGE-m3모델은 0.97, OpenAI text-embedding-3-large는 0.93으로 꽤 높은 것을 볼 수 있다. 

### 결정
k값을 5로 하면 검색의 신뢰도는 0.9수준으로 유지한 채 토큰 소모는 줄일 수 있다.

k값을 10으로 하면 검색의 신뢰도가 상승하지만 그만큼 토큰 소모가 세다.

Accyracy 점수가 높은 BGE-m3모델을 사용. 토큰 소모량이 많아도 주식, 경제 관련 도메인은 정확도가 중요하므로 k=10 적용.