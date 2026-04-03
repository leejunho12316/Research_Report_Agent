# Multimodal Processing
이미지 캡션 데이터 생성 시 출처와 함께 저장. 대화 진행 중 LLM이 이미지 출처가 포함되어 있는 데이터를 발견하면 html 형식으로 전환해 답변 사이에 첨부. Browser는 자동으로 img src를 요청하고 FastAPI StaticFiles가 URL로 직접 서빙.


```
## 이미지 콘텐츠
                                                                                                                                                                                                         
  이 이미지는 동아쏘시오홀딩스 관련 PDF 보고서(2026.3.27 작성)의 한 부분을 발췌한 것입니다. 전체 이미지에서 이 그래프는 "그림 4. 동아쏘시오홀딩스 연간 실적 추이" 제목 아래에 위치하며, 연도별 매출액과
  영업이익 변화를 바(Bar) 차트로 나타냅니다.                                                                                                                                                             

  - X축: 2021~2025년
  - Y축: 십억원 단위, 매출액(회색)·영업이익(주황색)
  - 매출액: 2021년 8,820억 → 2025년 1조 4,300억으로 꾸준히 증가

  출처: /data\20260327_동아쏘시오홀딩스 (000640_Not Rated)\fig\figure-3-8.jpg
```

-> LLM이 html 형식으로 출처 전환 <br>

</img src="http://127.0.0.1:8000/data/.../figure-3-8.jpg"> <br>

-> FastAPI StaticFiles가 URL로 직접 서빙

<img src="./readme_images/멀티모달 예시 1.png" width = '500'>

특정 그래프 해설을 요구했을 때 해당 그래프 이미지를 가져와 해설하는 것이 가능하다.

<img src="./readme_images/멀티모달 예시 3.png" width = '500'>

특정 이미지/그래프를 언급하지 않아도 필요 시 답변에 이미지를 추가하는 모습.

<br><br><br>

# Multiturn Processing

대화 기록을 PDF 파일 단위로 저장하고 context에는 최근 2쌍 (user, assistant 4개의 매세지)를 전달하는 방식으로 작동한다.


**흐름**

1. 채팅 요청 수신 시 `file_name`의 대화 기록 로드.
2. `RunnableWithMessageHistory`가 RAG 체인에 대화 기록을 자동으로 첨부.
3. 답변 완료 후 HumanMessage·AIMessage를 JSON 저장 파일에 추가 저장.
4. 서버 재시작, 채팅방 종료 후 재접속 후에도 이전 대화 맥락 유지.


<img src="./readme_images/멀티턴 예시 1.png" width = '500'>

OCI 홀딩스의 2023년 ROE를 먼저 물은 후 2027년, 2025년에 대해 물었을 때 ROE를 명시하지 않아도 ROE에 대해 대답을 하는 모습.

<img src="./readme_images/멀티턴 예시 2.png" width = '500'>

동아씨오 홀딩스 매출액 추이에 대해 먼저 물은 후 그 이유를 물으니 매출액을 명시하지 않아도 매출액 변동 원인에 대해 대답을 하는 모습.

<br><br><br>

<br><br><br>