# API 명세서

Base URL: `http://127.0.0.1:8000`

---

## 1. PDF 업로드

**POST** `/upload/`

PDF 파일을 업로드하고 백그라운드 전처리를 시작합니다. 처리가 끝나길 기다리지 않고 즉시 `task_id`를 반환합니다.

### Request

| 항목 | 값 |
|------|----|
| Content-Type | `multipart/form-data` |

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| file | File | O | 업로드할 PDF 파일 |

### Response `200 OK`

```json
{
  "message": "파일 업로드 완료! 전처리가 백그라운드에서 시작되었습니다.",
  "task_id": 1
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| message | string | 처리 시작 안내 메시지 |
| task_id | integer | 이후 상태 조회에 사용하는 작업 번호 |

---

## 2. 처리 상태 조회

**GET** `/status/{task_id}`

업로드한 파일의 전처리 진행 상태를 조회합니다. 프론트엔드는 2초마다 폴링합니다.

### Path Parameter

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| task_id | integer | `/upload/` 에서 받은 작업 번호 |

### Response `200 OK`

```json
{
  "task_id": 1,
  "status": "completed",
  "progress": "6/6",
  "result_url": "/data/삼성전자/vectordb"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| task_id | integer | 작업 번호 |
| status | string | 현재 상태 (`processing` \| `completed` \| `failed`) |
| progress | string \| null | 진행 단계 (예: `"3/6"`) |
| result_url | string \| null | 완료 시 결과 경로, 미완료 시 `null` |

### Response — task_id 없을 때

```json
{
  "error": "해당 작업 번호를 찾을 수 없습니다."
}
```

---

## 3. 처리 완료 파일 목록 조회

**GET** `/files/`

`/data/` 디렉터리를 탐색하여 기존에 전처리 완료된 파일 목록을 반환합니다. 페이지 최초 로드 시 호출됩니다.

### Request

없음

### Response `200 OK`

```json
{
  "files": [
    { "name": "삼성전자", "status": "completed" },
    { "name": "SK하이닉스", "status": "processing" }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| files | array | 파일 목록 |
| files[].name | string | PDF 파일명 (확장자 제외) |
| files[].status | string | `completed` (vectordb 폴더 존재) \| `processing` (미완료) |

### Response — `/data/` 디렉터리 없을 때

```json
{
  "files": []
}
```

---

## 4. 채팅 (RAG 질의응답)

**POST** `/chat/`

지정한 파일의 VectorDB를 로드하고, RAG 체인(Chroma retriever + GPT-4o)으로 질문에 답변합니다.

### Request

| 항목 | 값 |
|------|----|
| Content-Type | `application/json` |

```json
{
  "file_name": "삼성전자",
  "message": "2024년 영업이익은 얼마인가요?"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| file_name | string | O | `/files/` 에서 받은 파일명 (확장자 제외) |
| message | string | O | 사용자 질문 |

### Response `200 OK`

```json
{
  "answer": "2024년 삼성전자의 영업이익은 **32.7조 원**입니다.\n\n..."
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| answer | string | 마크다운 형식의 답변. 이미지 참조 시 `<img src=\"...\">` 포함 가능 |

### Response — VectorDB 없을 때

```json
{
  "error": "VectorDB를 찾을 수 없습니다: 삼성전자"
}
```

---

## 상태값 정의

| status | 설명 |
|--------|------|
| `processing` | 전처리 진행 중 |
| `completed` | 전처리 완료, 채팅 가능 |
| `failed` | 전처리 중 오류 발생 |

---

## 전처리 파이프라인 단계 (progress 참고용)

| 단계 | 내용 |
|------|------|
| 1/6 | Unstructured — 텍스트/테이블/figure 추출 |
| 2/6 | PyMuPDF — 페이지별 PNG 이미지 추출 |
| 3/6 | PyMuPDF — 페이지별 TXT 추출 |
| 4/6 | LLM (gpt-4.1) — 페이지 이미지+텍스트 정제 |
| 5/6 | LLM (gpt-4.1) — QA 합성 데이터 생성 |
| 6/6 | Chroma — VectorDB 저장 |
