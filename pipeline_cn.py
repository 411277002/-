import joblib
import shap
import pandas as pd

# ===== 載入模型 =====
stage1_scaler = joblib.load("models/scaler_stage1.pkl")
stage2_scaler = joblib.load("models/scaler_stage2.pkl")
stage1 = joblib.load("models/model_stage1.pkl")
stage2 = joblib.load("models/model_stage2.pkl")
shap_explainer = shap.TreeExplainer(stage1)

# ===== 特徵 =====
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
FEATURES_MISSING = [f"{c}_missing" for c in FEATURES]
STAGE1_FEATURES_ALL = FEATURES + FEATURES_MISSING

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

# ===== 國文 cluster 名稱直接寫在 pipeline =====
CLUSTER_NAME_MAP = {
    0: "高效學霸型",
    1: "勤奮影音型",
    2: "潛在放棄型",
    3: "盲忙型"
}

def analyze_cn(df):
    identity_cols = df.columns.difference(FEATURES)
    identity_df = df[identity_cols]

    X = df.copy()
    for col in FEATURES:
        X[col + "_missing"] = X[col].isna().astype(int)
    X[FEATURES] = X[FEATURES].fillna(0)

    X_stage1 = X[STAGE1_FEATURES_ALL]
    X_stage1_scaled = stage1_scaler.transform(X_stage1)
    outlier_preds = stage1.predict(X_stage1_scaled)

    normal_groups = {name: [] for name in CLUSTER_NAME_MAP.values()}
    outlier_records = []

    for idx, is_outlier in enumerate(outlier_preds):
        student_identity = identity_df.iloc[idx].to_dict()

        if is_outlier == 1:
            shap_vals = shap_explainer.shap_values(X_stage1_scaled[idx:idx+1])[0]
            max_idx = abs(shap_vals).argmax()
            reason_feature = STAGE1_FEATURES_ALL[max_idx]

            record = student_identity.copy()
            record.update({
                "status": "離群學生",
                "biggest_influencing_factor": FEATURE_NAME_MAP.get(reason_feature.replace("_missing",""), reason_feature)
            })
            outlier_records.append(record)
        else:
            X_stage2 = X.loc[idx:idx, FEATURES]
            X_stage2_scaled = stage2_scaler.transform(X_stage2)
            cluster_idx = int(stage2.predict(X_stage2_scaled)[0])
            cluster_name = CLUSTER_NAME_MAP[cluster_idx]

            record = student_identity.copy()
            record.update({
                "cluster": cluster_name
            })
            normal_groups[cluster_name].append(record)

    # 轉成 DataFrame
    normal_group_dfs = {name: pd.DataFrame(records) for name, records in normal_groups.items() if records}
    outlier_df = pd.DataFrame(outlier_records)

    return normal_group_dfs, outlier_df
