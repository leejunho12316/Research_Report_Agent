#evaluation_result.csv를 사용해 result 별 정확도 시각화 이미지를 만들어 저장한다.
#전체 문제 개수(100개) 별 맞춘 개수 (O)의 개수와 틀린 개수 (X)를 막대그래프로 나타낸다.
#현재 폴더에 evaluation_visualization.png로 저장한다.

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_RESULT_PATH = os.path.join(EVAL_DIR, "evaluation_result.csv")
OUTPUT_PATH = os.path.join(EVAL_DIR, "evaluation_visualization.png")

LABEL_MAP = {
    "answer_plain_result": "Plain",
    "answer_refined_result": "Refined",
    "answer_qa_only_result": "QA Only",
    "answer_refined_qa_result": "Refined + QA",
}


def main():
    df = pd.read_csv(EVAL_RESULT_PATH)
    assert isinstance(df, pd.DataFrame)

    result_cols = [col for col in LABEL_MAP if col in df.columns]
    if not result_cols:
        raise ValueError("채점 결과 칼럼이 없습니다. 먼저 3.evaluation_grader.py를 실행하세요.")

    total = len(df)
    labels = [LABEL_MAP[col] for col in result_cols]
    correct_counts = [(df[col] == "O").sum() for col in result_cols]
    accuracies = [c / total * 100 for c in correct_counts]

    plot_df = pd.DataFrame({"방식": labels, "정답 수": correct_counts})

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=min(correct_counts) * 0.5, vmax=max(correct_counts) * 1.2)
    palette = [cmap(norm(c)) for c in correct_counts]
    sns.barplot(data=plot_df, x="방식", y="정답 수", palette=palette, width=0.6, ax=ax)

    # 막대 위에 개수 표시
    for patch, count, acc in zip(ax.patches, correct_counts, accuracies):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.3,
            str(int(count)),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    # x축 레이블에 정확도 추가
    ax.set_xticklabels(
        [f"{lbl}\n({acc:.1f}%)" for lbl, acc in zip(labels, accuracies)],
        fontsize=11,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy by Data Processing Method (Total {total} Questions)", fontsize=14, fontweight="bold")
    ax.set_ylim(50, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"시각화 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()