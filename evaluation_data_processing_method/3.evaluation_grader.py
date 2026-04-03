# evaluation_result.csv를 사용.
# LLM에게 evaluation_result.csv의 Question, Answer를 주고 정답이 입력된 칼럼이 제대로 답했는지 O, X로 채점.
# 채점 대상 칼럼 : answer_plain, answer_refined, answer_qa_only, answer_refined_qa
# 채점 결과는 각각 answer_plain_result, answer_refined_result, answer_qa_only_result, answer_refined_qa_result 칼럼으로 evaluation_result.csv에 저장됨.
#
# 채점 조건
#숫자에 ,는 신경쓰지 않기
#단위 (십억, GW 등) 작성 안하면 틀린것
#원, 달러같은건 안적어도 됨.
#답이 여러개라면 그 중 하나만 적어도 정답.
#

#채점 결과 evaluation_result.csv에 같이 저장되도록 수정

import os
from typing import cast

import openai
import pandas as pd
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm
import time

load_dotenv()

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_RESULT_PATH = os.path.join(EVAL_DIR, "evaluation_result.csv")

LLM_MODEL = "gpt-4.1-mini"

# ---------------------------------------------------------------------------
# 채점 프롬프트
# ---------------------------------------------------------------------------

GRADER_SYSTEM_PROMPT = """당신은 객관식/단답형 답변을 채점하는 채점관입니다.
정답(Answer)과 학생 답변(Student Answer)을 비교하여 O 또는 X 중 하나만 출력하세요.

채점 규칙:
1. 숫자의 쉼표(,)는 무시합니다. (예: 1,000 = 1000)
2. 십억, GW, % 등 단위는 반드시 포함되어야 정답입니다. 단위를 빠뜨리면 X입니다.
3. 원(KRW), 달러(USD) 등 통화 단위는 없어도 정답으로 인정합니다.
4. 정답이 여러 개인 경우(예: "말레이시아, 베트남, 캄보디아, 태국 중 택1") 그 중 하나만 맞아도 O입니다.
5. 의미가 같으면 표현이 달라도 정답입니다. (예: "매수" = "BUY")
6. 대소문자는 구분하지 않습니다.
7. 정답과 학생 답변의 핵심 수치·단어가 일치하면 O, 다르면 X입니다.
8. 답이 범위인 경우 범위의 시작과 끝을 정확히 적어야 정답으로 인정해주세요.

반드시 O 또는 X 한 글자만 출력하세요."""


def grade_answer(client: openai.OpenAI, question: str, correct_answer: str, student_answer: str) -> str:
    """LLM으로 단일 답변을 채점하여 O 또는 X 반환"""
    user_prompt = (
        f"질문: {question}\n"
        f"정답: {correct_answer}\n"
        f"학생 답변: {student_answer}"
    )

    messages = cast(list[ChatCompletionMessageParam], cast(object, [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=5,
        stream=False,
    )

    result = (response.choices[0].message.content or "").strip()
    # 혹시 앞뒤 공백이나 설명이 붙으면 O/X만 추출
    if "O" in result:
        return "O"
    if "X" in result:
        return "X"
    return "X"


def grade_column(
    client: openai.OpenAI,
    df: pd.DataFrame,
    answer_col: str,
) -> list[str]:
    """answer_col 전체를 채점하여 O/X 리스트 반환"""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"채점 중: {answer_col}"):
        grade = grade_answer(
            client,
            question=str(row["Question"]),
            correct_answer=str(row["Answer"]),
            student_answer=str(row[answer_col]),
        )
        results.append(grade)

        time.sleep(1)

    return results


def print_summary(df: pd.DataFrame, result_cols: list[str]) -> None:
    """채점 결과 요약 출력"""
    print("\n===== 채점 결과 요약 =====")
    for col in result_cols:
        total = len(df)
        correct = (df[col] == "O").sum()
        accuracy = correct / total * 100
        print(f"  {col}: {correct}/{total} ({accuracy:.1f}%)")
    print("==========================")


def main():
    df = pd.read_csv(EVAL_RESULT_PATH)
    assert isinstance(df, pd.DataFrame)

    # answer_로 시작하는 칼럼 자동 감지 (채점 결과 칼럼 제외)
    answer_cols = [
        col for col in df.columns
        if col.startswith("answer_") and not col.endswith("_result")
    ]
    print(f"채점 대상 칼럼: {answer_cols}")

    client = openai.OpenAI()

    result_cols = []
    for col in answer_cols:
        result_col = f"{col}_result"

        # 이미 채점된 칼럼은 건너뜀
        if result_col in df.columns and df[result_col].notna().all():
            print(f"건너뜀 (이미 채점 완료): {result_col}")
            result_cols.append(result_col)
            continue

        df[result_col] = grade_column(client, df, col)
        result_cols.append(result_col)

        # 칼럼 채점 완료 시마다 중간 저장
        df.to_csv(EVAL_RESULT_PATH, index=False, encoding="utf-8-sig")
        print(f"중간 저장 완료: {result_col}")

    print_summary(df, result_cols)


if __name__ == "__main__":
    main()
