import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# --------------------------
# 定義各構面的欄位
# --------------------------
ENG_COLS = ['Total_View_Duration', 'Avg_Finish_Rate', 'Full_Completion_Count',
            'Total_Prac_Attempts', 'Total_Interaction_Count']

STR_COLS = ['Total_Rewinds', 'Total_Forwards', 'Total_Speed_Changes',
            'Total_Post_Exam_Reviews', 'Avg_Exam_Ans_Time', 'Std_Exam_Ans_Time']

PROF_COLS = ['Avg_Prac_Score_Rate', 'Avg_Prac_Duration', 'Avg_Exam_Correctness']

# --------------------------
# 計算 PCA 函式
# --------------------------
def compute_pcs(df, cols):
    X = df[cols].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)
    return pcs

# --------------------------
# 生成雷達圖函式
# --------------------------
def plot_radar_cn(df, user_id, baseline=0):
    if user_id not in df['user_sn'].values:
        raise ValueError(f"找不到學生編號 {user_id}")

    eng_pcs = compute_pcs(df, ENG_COLS)
    str_pcs = compute_pcs(df, STR_COLS)
    prof_pcs = compute_pcs(df, PROF_COLS)

    idx = df.index[df['user_sn'] == user_id][0]
    values = np.concatenate([eng_pcs[idx], str_pcs[idx], prof_pcs[idx]])
    values = values.tolist()
    values += values[:1]

    labels = [
    '總投入度', '完課率', '互動型(+)或練習型(-)',
    '作答時間', '影片操作(+)或檢討(-)', '影片觀看節奏',
    '快速熟練', '穩扎穩打', '練習型(+)或考試型(-)'
    ]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # 畫圖
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2, label=f'學生 {user_id}')
    ax.fill(angles, values, color='blue', alpha=0.25)

    # 基準線
    baseline_values = [baseline] * len(labels)
    baseline_values += baseline_values[:1]
    ax.plot(angles, baseline_values, color='red', linestyle='--', linewidth=1.5, label=f'基準值={baseline}')
    ax.scatter(angles[:-1], baseline_values[:-1], color='red')

    # 軸標籤
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f'PCA 雷達圖', pad=20)
    ax.legend(loc='upper right')

    return fig
