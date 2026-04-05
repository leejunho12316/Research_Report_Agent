# Research_Report_Agent
증권사 기업분석, 투자분석 리서치 리포트 기반 MultiModal-RAG LLM Agent 서비스. <br>
PDF 업로드부터 챗봇을 이용한 질의응답까지 전 과정을 자동화한 AI 파이프라인.

<img src="res/banner.png">

# 📝Index
- [📷Screenshot](#screenshot)
- [🎯Stacks](#stacks)
- [💎Implementation Details](#implementation-details)
  - [Data Processing Pipeline](#Data-Processing-Pipeline)
  - [Evaluation](#evaluation)
    - [데이터 전처리 방식에 따른 RAG 검색 성능평가](./evaluation_data_processing_method/README.md)
    - [임베딩 모델에 따른 RAG 검색 성능평가](./evaluation_embedding_model/README.md)
  - [Multimodal & Multiturn](#multimodalmultiturn)

<br><br><br>

# 📸Screenshot

<img src="readme_images/screenshot_index.png">
<img src="readme_images/screenshot.png">

<br><br><br>

## 🎯Stacks

**Data** : 
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Unstructured](https://img.shields.io/badge/Unstructured-FF6B35?style=flat-square&logoColor=white)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-00A86B?style=flat-square&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

**AI** : 
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_API-412991?style=flat-square&logo=openai&logoColor=white)
![MultiModal RAG](https://img.shields.io/badge/MultiModal--RAG-FF6F61?style=flat-square&logoColor=white)
![MultiTurn](https://img.shields.io/badge/MultiTurn-4A90D9?style=flat-square&logoColor=white)
![LLM Agent](https://img.shields.io/badge/LLM_Agent-F5A623?style=flat-square&logoColor=white)

**Database** : 
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF5A1F?style=flat-square&logoColor=white)

**Backend** : 
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

**Frontend** : 
![HTML](https://img.shields.io/badge/HTML-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS-663399?style=flat-square&logo=css&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

<br>

### Special Requirements
1. tesseract <br> https://github.com/UB-Mannheim/tesseract/wiki
<br>C:\Program Files\Tesseract-OCR PATH 추가

2. poppler <br> https://github.com/oschwartz10612/poppler-windows/releases <br>
Library/bin/ 폴더 PATH에 추가

3. pip install <br> 
`!pip install -U "unstructured[all-docs]" lxml pillow==9.5.0 pdf2image==1.16.3 layoutparser[layoutmodels,tesseract]==0.3.4`


<br><br><br>

# 💎Implementation Details

## Data Processing Pipeline

<img src="readme_images/figure-7-10.jpg" width="200">

### - [데이터 처리 파이프라인](./README_data_pipeline.md)

<br>

## Evaluation

<img src="evaluation_data_processing_method/evaluation_visualization.png" width="200">

### - [데이터 전처리 방식에 따른 RAG 검색 성능평가](./evaluation_data_processing_method/README.md)

<img src="evaluation_embedding_model/heatmaps/first_result/heatmap_k10.png" width = '200'>

### - [임베딩 모델에 따른 RAG 검색 성능평가](./evaluation_embedding_model/README.md)

<br>

## Multimodal&Multiturn

<img src="readme_images/멀티모달 예시 2.png" width = '200'>

### - [MultiModal & Multiturn](./README_multi.md)

<br><br><br><br><br><br>

---


<details>
<summary>Notes</summary>

정답 채점 시 반올림 한 것은 어떻게 해야할까? -> ㅇ. 정보를 찾았다는 뜻이기 때문.

QA 데이터 정체 성능평가: 정제, QA 따로 일때 알지 못했던 내용들을 잘 검색하는 성과를 보여줌.
틀린 문제의 절반은 십억원, 억원 등 액수 단위에 있어서의 오류였다. 이는 검색 자체는 올바르게 되었지만 더 넓은 문맥에서의 정보가 포함되지 않아 이해가 부족하다는 의미.

임베딩 모델 별 평가
? LLM 정제 데이터 + QA 합성 데이터 + 이미지 전저리 데이터 전부 다 vectordb 구성하면 **답변**퀄리티는 좋아진다. 하지만 정확한 측저을 해야 하는 임베딩 모델 평가 과정에서는 데이터 양이 너무 많아지면 특정한 하나의 문서를 찾아낼 확률이 급격히 낮아진다. 따라서 임베딩 모델에 따른 **검색 성능 평가**는 의도적으로 데이터 개수를 줄여서 진행한다.

점수 너무 낮은 이유 가설
~~1. 평가 매트릭 함수가 잘못됨. -> retriever의 k가 5로 설정되어있었음~~
~~2. 질문이 검색이 잘 안되도록 잘못 만들어짐.~~
~~3. doc이 너무 많아 검색이 제대로 될 리가 없음.~~

임베딩 모델의 크기 -> bge-m3, OpenAI large model 성능 측정

</details>
