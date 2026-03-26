# Research_Report_Agent
증권사 기업분석, 투자분석 리서치 리포트 기반 MultiModal-RAG LLM Agent 서비스.
PDF 업로드부터 챗봇을 이용한 질의응답까지 전 과정을 자동화한 AI 파이프라인.

--이미지

## Screenshot

<img src="./images/screenshot.png">

## Note

여러가지 개발하며 노트 했던거

##데이터 처리 파이프라인
**1. 이미지 추출**

Unstructured 패키지 사용해 데이터 시각화 차트 이미지 추출.
소형 노이즈 이미지 자동 필터링.

<img src="./images/figure-7-10.jpg" width="200">          <img src="./images/figure-7-11.jpg" width="200">          <img src="./images/figure-7-12.jpg" width="200">

**2. PDF 데이터 추출**

PyMuPDF 사용해 pdf 전체 페이지 이미지 변환과 원본 텍스트 추출.

<img src="./images/pdf_image.png" width="300">
--pdf 데이터 이미지, 텍스트

**3. 페이지 정제**

gpt-4.1 Vision API로 페이지 이미지를 참고해 페이지 구조를 파악하며 원본 텍스트를 정제.

일반 텍스트부터 복잡한 테이블과 그래프까지 데이터 손실 없이 평문화.


**4. QA 합성 데이터 생성**

gpt-4.1로 정제된 텍스트 데이터를 연속된 두 페이지씩 참고해 QA 형식의 합성 데이터 생성.

→ 페이지가 넘어가며 문맥이 잘리는 현상 방지.

→ 정제된 텍스트를 그대로 ChromaDB에 추가했을 때보다 검색 성능 증가.

**5. 이미지 설명 데이터 생성**

LLM에게 전체 pdf 페이지 이미지를 참고해 맥락을 파악하며 추출된 이미지를 설명하도록 요구.

→ QA 합성 데이터와 이미지 설명 데이터를 합쳐 **Chroma VectorDB**에 임베딩 저장.

→ **SQLite**에 pdf 파일 별 전처리 현황 업데이트로 사용자 UI에 진행 과정 알림.


# 성능비교

성능비교 표 추가
