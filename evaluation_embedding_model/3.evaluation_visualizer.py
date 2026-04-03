
# data : retriever_comparison_results.csv
# 칼럼과 행 전치.
#
# k값에 따라 dataframe 분리
# df1 (칼럼 : Accuracy@1, Precision@1 ,,, )
# df3 (칼럼 : Accuracy@3, Precision@3 ,,, )
# df5 ...
# df10 ...
#
# 각 dataframe seaborn heatmap 으로 시각화해 png 저장

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RES_DIR = os.path.join(os.path.dirname(__file__))
CSV_PATH = os.path.join(RES_DIR, "heatmaps/second_result/retriever_comparison_results.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. CSV 로드 및 전치 ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, index_col=0)  # index: 메트릭명, columns: 리트리버명
df = df.T                                # 전치 → index: 리트리버명, columns: 메트릭명

# ── 2. k값에 따라 DataFrame 분리 ──────────────────────────────────────────────
k_values = [1, 3, 5, 10]
dfs = {k: df[[c for c in df.columns if c.endswith(f"@{k}")]] for k in k_values}

# ── 3. 각 DataFrame을 seaborn heatmap으로 시각화 후 PNG 저장 ──────────────────
for k, df_k in dfs.items():
    fig, ax = plt.subplots(figsize=(len(df_k.columns) * 1.2, len(df_k) * 1.0 + 0.6))

    sns.heatmap(
        df_k.astype(float).round(4),
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(f"Retriever 성능 비교 (k={k})", fontsize=14, pad=12)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Retriever", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"heatmap_k{k}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {out_path}")