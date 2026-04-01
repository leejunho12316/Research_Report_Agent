"""
refined_pages.json (이미지 + 텍스트 LLM 정제 데이터) 사용해 Golden Set 만들기

refined_pages.json은 pdf가 가지고 있는 텍스트를 pdf 구조와 형식을 참고해 평문으로 전환한 정제된 텍스트 데이터이다.
이 데이터를 한 페이지씩 LLM에 context로 집어넣으면서 QA 셋을 만든다.

QA셋은 사람이 pdf 파일을 보고서 agent에게 물어볼만한 질문과 그에 대한 답 셋이다.
Answer 칼럼은 정답이 정해져 있는 단답형 질문이어야 한다.

칼럼:
  Question : 사람이 할 만한 말투로 pdf 파일에 대한 질문
  Answer   : 정답이 있는 단답형 답변 (예: 45, 1, 상승)

과정:
  1. refined_pages.json 불러오기
  2. 한 페이지씩 iteration하며 LLM에 prompt 전달
  3. 해당 context 기반 QA set 5개 생성 요청
  4. 답변 파싱
  5. iteration 종료 후 csv 파일로 저장 (최소 100개)
"""

import os
import json
import csv
import re

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

REFINED_PAGES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "temp_res", "json/refined_pages_OCI.json"
)

OUTPUT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden_set.csv"
)

QA_PER_PAGE = 10

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""당신은 주어진 문서로부터 사람이 실제로 물어볼법한 질문과 단답형 정답 쌍을 생성하는 전문가입니다.
주어진 문서는 미래에셋 증권 회사에서 발생한 기업 분석 리서치 리포트 입니다.

규칙:
1. 반드시 주어진 문서에 근거한 질문과 답변만 작성하십시오.
2. 답변(Answer)은 반드시 단답형이어야 합니다. 숫자, 단어, 짧은 구절 수준으로 작성하십시오. (예: 270,000원, 36.2%, 매수, 상승, 3개월)
3. 질문(Question)은 일반 사용자가 자연스럽게 물어볼 수 있는 말투로 작성하십시오.
4. 정확한 수치, 비율, 등급, 방향성 등 객관적으로 정답이 하나로 정해지는 질문을 만드십시오.
5. 추론이 필요하거나 답이 여러 개일 수 있는 질문은 만들지 마십시오.
6. 답이 여러 개가 될 수 있고 선택지가 많은 경우 답으로 '전체 중 택1'이라고 작성하세요.
7. 반드시 아래 형식을 정확히 지켜 {QA_PER_PAGE}개를 출력하십시오.

출력 형식:
Q1: (질문)
A1: (단답형 답변)
Q2: (질문)
A2: (단답형 답변)
Q3: (질문)
A3: (단답형 답변)
...
(생략)
"""


def build_user_prompt(page_text: str, page_num: int) -> str:
    return f"[페이지 {page_num}]\n\n{page_text}\n\n"


def parse_qa(response_text: str) -> list[dict]:
    """LLM 응답에서 Q/A 쌍을 파싱해 [{'Question': ..., 'Answer': ...}, ...] 반환"""
    pairs = []
    for i in range(1, QA_PER_PAGE + 1):
        q_pattern = rf"Q{i}:\s*(.+?)(?=A{i}:)"
        a_pattern = rf"A{i}:\s*(.+?)(?=Q{i + 1}:|$)"
        q_match = re.search(q_pattern, response_text, re.DOTALL)
        a_match = re.search(a_pattern, response_text, re.DOTALL)
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            if question and answer:
                pairs.append({"Question": question, "Answer": answer})
    return pairs


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    # 1. refined_pages.json 불러오기
    with open(REFINED_PAGES_PATH, encoding='utf-8') as f:
        refined_pages: list[str] = json.load(f)
    print(f"총 {len(refined_pages)}페이지 로드 완료.")

    client = openai.OpenAI()
    all_qa: list[dict] = []

    # 2~4. 페이지별 LLM 호출 및 파싱
    for page_num, page_text in enumerate(tqdm(refined_pages, desc="QA 생성"), start=1):
        if not page_text.strip():
            continue

        user_prompt = build_user_prompt(page_text, page_num)

        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content or ""
        qa_pairs = parse_qa(response_text)
        for qa in qa_pairs:
            qa["Page"] = page_num
        all_qa.extend(qa_pairs)

        print(f"  페이지 {page_num}: {len(qa_pairs)}개 파싱 완료 (누적 {len(all_qa)}개)")

    print(f"\n총 QA 쌍: {len(all_qa)}개")


    # 5. CSV 저장
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Page", "Question", "Answer"])
        writer.writeheader()
        writer.writerows(all_qa)

    print(f"저장 완료: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
