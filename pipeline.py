import joblib
import shap
import pandas as pd

stage1_scaler = joblib.load("models/scaler_stage1.pkl")
stage2_scaler = joblib.load("models/scaler_stage2.pkl")
stage1 = joblib.load("models/model_stage1.pkl")
stage2 = joblib.load("models/model_stage2.pkl")

shap_explainer = shap.TreeExplainer(stage1)

FEATURES = [
    "Total_View_Duration",
        "Avg_Finish_Rate",
        "Full_Completion_Count",
        "Total_Prac_Attempts",
        "Total_Interaction_Count",
        "Total_Rewinds",
        "Total_Forwards",
        "Total_Speed_Changes",
        "Total_Post_Exam_Reviews",
        "Avg_Exam_Ans_Time",
        "Std_Exam_Ans_Time",
        "Avg_Prac_Score_Rate",
        "Avg_Prac_Duration",
        "Avg_Exam_Correctness"
]

FEATURE_NAME_MAP = {
    "Total_View_Duration": "總觀看時長",
    "Avg_Finish_Rate": "平均影片完成率",
    "Full_Completion_Count": "完整觀看次數",
    "Total_Prac_Attempts": "練習題作答次數",
    "Total_Interaction_Count": "互動次數（含操作行為）",
    "Total_Rewinds": "倒轉次數",
    "Total_Forwards": "快轉次數",
    "Total_Speed_Changes": "播放速度切換次數",
    "Total_Post_Exam_Reviews": "考後檢討次數",
    "Avg_Exam_Ans_Time": "平均考題作答時間",
    "Std_Exam_Ans_Time": "作答時間",
    "Avg_Prac_Score_Rate": "練習題平均得分率",
    "Avg_Prac_Duration": "練習題平均作答時長",
    "Avg_Exam_Correctness": "檢核點平均正確率"
}

CLUSTER_NAME_MAP = {
    0: "精準型",
    1: "低投入型",
    2: "努力型"
}

    
def analyze_file(df):
    """
    df: DataFrame (多筆學生資料)
    回傳：df_result, detail_list
    df_result → 用於顯示與下載的 DataFrame
    detail_list → 包含每位學生的 shap / cluster（詳細資訊）
    """
    identity_cols = df.columns.difference(FEATURES)
    identity_df = df[identity_cols]
    X = df[FEATURES]

    # -------- Stage 1：所有資料先做 scaler1 --------
    X_scaled_stage1 = stage1_scaler.transform(X.values)
    outlier_preds = stage1.predict(X_scaled_stage1)

    normal_groups = {c: [] for c in CLUSTER_NAME_MAP.keys()}
    outlier_records = []

    # -------- Stage 2 & SHAP：逐筆處理 --------
    for idx, is_outlier in enumerate(outlier_preds):
        student_identity = identity_df.iloc[idx].to_dict()
        if is_outlier == 1:
            shap_values = shap_explainer.shap_values(X_scaled_stage1[idx:idx+1])

            max_idx = shap_values.argmax()
            reason_feature = FEATURE_NAME_MAP[FEATURES[max_idx]]

            record = student_identity.copy()
            record.update({
                "status": "Outlier",
                "biggest_influencing_factor": reason_feature
            })
            outlier_records.append(record)

        else:
            X_scaled_stage2 = stage2_scaler.transform(X.values[idx:idx+1])
            cluster = int(stage2.predict(X_scaled_stage2)[0])
            cluster_name = CLUSTER_NAME_MAP[cluster]

            record = student_identity.copy()
            record.update({
                "status": "Normal",
                "cluster": cluster_name
            })
            normal_groups[cluster].append(record)

    normal_group_dfs = {}
    for cluster_id, records in normal_groups.items():
        if len(records) > 0:
            normal_group_dfs[cluster_id] = pd.DataFrame(records)
    outlier_df = pd.DataFrame(outlier_records)
    return normal_group_dfs, outlier_df
