import joblib
import pandas as pd
import numpy as np
import shap

scaler_stage1 = joblib.load("models/math_models/scaler_stage1.pkl")
model_stage1 = joblib.load("models/math_models/model_stage1.pkl")
scaler_stage2 = joblib.load("models/math_models/scaler_stage2.pkl")
model_stage2 = joblib.load("models/math_models/model_stage2.pkl")
cluster_centers = joblib.load("models/math_models/cluster_centers.pkl")
explainer_stage1 = shap.TreeExplainer(model_stage1)
explainer_stage2 = shap.TreeExplainer(model_stage2)

STAGE1_FEATURES_ALL = [
    'Total_View_Duration', 'Avg_Finish_Rate', 'Full_Completion_Count',
    'Total_Prac_Attempts', 'Total_Interaction_Count', 'Active_Days',
    'Review_Ratio', 'Completion_View_Efficiency', 'View_Efficiency',
    'Prac_Efficiency', 'Composite_Engagement_Index', 'Avg_Daily_View_Time',
    'Avg_Daily_Prac', 'Avg_Daily_Interaction', 'Total_Notes', 'Total_Chkpts',
    'Total_Continue', 'Rewind_Ratio', 'Forward_Ratio', 'SpeedChange_Ratio',
    'PostReview_Ratio', 'Note_Ratio', 'Chkpt_Ratio', 'Continue_Ratio',
    'Revisit_Intensity', 'Total_Adjustment_Ratio', 'Focus_Play_Active',
    'Focus_Pause_Control', 'Focus_Speed_Adjust', 'Focus_Seek',
    'Focus_Review_Checkpoints', 'Focus_Continue', 'Focus_Notes',
    'Focus_Browse_UI', 'Focus_Questions', 'Speed_Correctness_Index',
    'Median_Exam_Ans_Time', 'Total_Correct', 'Avg_Prac_Score_Rate',
    'Avg_Prac_Efficiency', 'Prac_Score_StdDev', 'Avg_Exam_Correctness',
    'Exam_Efficiency', 'Total_Exam_Count', 'Prac_Improvement',
    'Practice_Exam_Consistency', 'Avg_Item_Error_Rate', 'Error_Streak_Max'
] + [c + "_missing" for c in [
    'Total_View_Duration', 'Avg_Finish_Rate', 'Full_Completion_Count',
    'Total_Prac_Attempts', 'Total_Interaction_Count', 'Active_Days',
    'Review_Ratio', 'Completion_View_Efficiency', 'View_Efficiency',
    'Prac_Efficiency', 'Composite_Engagement_Index', 'Avg_Daily_View_Time',
    'Avg_Daily_Prac', 'Avg_Daily_Interaction', 'Total_Notes', 'Total_Chkpts',
    'Total_Continue', 'Rewind_Ratio', 'Forward_Ratio', 'SpeedChange_Ratio',
    'PostReview_Ratio', 'Note_Ratio', 'Chkpt_Ratio', 'Continue_Ratio',
    'Revisit_Intensity', 'Total_Adjustment_Ratio', 'Focus_Play_Active',
    'Focus_Pause_Control', 'Focus_Speed_Adjust', 'Focus_Seek',
    'Focus_Review_Checkpoints', 'Focus_Continue', 'Focus_Notes',
    'Focus_Browse_UI', 'Focus_Questions', 'Speed_Correctness_Index',
    'Median_Exam_Ans_Time', 'Total_Correct', 'Avg_Prac_Score_Rate',
    'Avg_Prac_Efficiency', 'Prac_Score_StdDev', 'Avg_Exam_Correctness',
    'Exam_Efficiency', 'Total_Exam_Count', 'Prac_Improvement',
    'Practice_Exam_Consistency', 'Avg_Item_Error_Rate', 'Error_Streak_Max']
]

FEATURE_NAME_MAP = {
    'Total_View_Duration': '總觀看時長',
    'Avg_Finish_Rate': '平均完成率',
    'Full_Completion_Count': '完整觀看次數',
    'Total_Prac_Attempts': '練習題作答次數',
    'Total_Interaction_Count': '互動次數',
    'Active_Days': '活躍天數',
    'Review_Ratio': '回看比例',
    'Completion_View_Efficiency': '有效觀看效率',
    'View_Efficiency': '觀看效率',
    'Prac_Efficiency': '練習效率',
    'Composite_Engagement_Index': '綜合投入指標',
    'Avg_Daily_View_Time': '每日平均觀看時間',
    'Avg_Daily_Prac': '每日平均練習時間',
    'Avg_Daily_Interaction': '每日平均互動次數',
    'Total_Notes': '筆記總數',
    'Total_Chkpts': '檢核點總數',
    'Total_Continue': '續播總次數',
    'Rewind_Ratio': '倒轉比例',
    'Forward_Ratio': '快轉比例',
    'SpeedChange_Ratio': '倍速切換比例',
    'PostReview_Ratio': '考後複習比例',
    'Note_Ratio': '筆記比例',
    'Chkpt_Ratio': '檢核點比例',
    'Continue_Ratio': '續播比例',
    'Revisit_Intensity': '回訪強度',
    'Total_Adjustment_Ratio': '操作調整比例',
    'Focus_Play_Active': '專注-播放活躍',
    'Focus_Pause_Control': '專注-暫停控制',
    'Focus_Speed_Adjust': '專注-變速操作',
    'Focus_Seek': '專注-跳轉',
    'Focus_Review_Checkpoints': '專注-檢核點瀏覽',
    'Focus_Continue': '專注-續看行為',
    'Focus_Notes': '專注-筆記行為',
    'Focus_Browse_UI': '專注-介面瀏覽',
    'Focus_Questions': '專注-題目操作',
    'Speed_Correctness_Index': '速度-正確率關聯指標',
    'Median_Exam_Ans_Time': '考題作答中位數時間',
    'Total_Correct': '總答對題數',
    'Avg_Prac_Score_Rate': '練習題平均得分率',
    'Avg_Prac_Efficiency': '練習題平均效率',
    'Prac_Score_StdDev': '練習得分標準差',
    'Avg_Exam_Correctness': '考試平均正確率',
    'Exam_Efficiency': '考試效率',
    'Total_Exam_Count': '考試次數',
    'Prac_Improvement': '練習進步幅度',
    'Practice_Exam_Consistency': '練習與測驗一致性',
    'Avg_Item_Error_Rate': '題目平均錯誤率',
    'Error_Streak_Max': '連錯最大次數'
}



STAGE2_FEATURES_ALL = [
    'Total_View_Duration', 'Avg_Finish_Rate', 'Full_Completion_Count',
    'Total_Prac_Attempts', 'Total_Interaction_Count', 'Active_Days',
    'Review_Ratio', 'Completion_View_Efficiency', 'View_Efficiency',
    'Prac_Efficiency', 'Composite_Engagement_Index', 'Avg_Daily_View_Time',
    'Avg_Daily_Prac', 'Avg_Daily_Interaction', 'Total_Notes', 'Total_Chkpts',
    'Total_Continue', 'Rewind_Ratio', 'Forward_Ratio', 'SpeedChange_Ratio',
    'PostReview_Ratio', 'Note_Ratio', 'Chkpt_Ratio', 'Continue_Ratio',
    'Revisit_Intensity', 'Total_Adjustment_Ratio', 'Focus_Play_Active',
    'Focus_Pause_Control', 'Focus_Speed_Adjust', 'Focus_Seek',
    'Focus_Review_Checkpoints', 'Focus_Continue', 'Focus_Notes',
    'Focus_Browse_UI', 'Focus_Questions', 'Speed_Correctness_Index',
    'Median_Exam_Ans_Time', 'Total_Correct', 'Avg_Prac_Score_Rate',
    'Avg_Prac_Efficiency', 'Prac_Score_StdDev', 'Avg_Exam_Correctness',
    'Exam_Efficiency', 'Total_Exam_Count', 'Prac_Improvement',
    'Practice_Exam_Consistency', 'Avg_Item_Error_Rate', 'Error_Streak_Max'
] + [c + "_missing" for c in [
    'Total_View_Duration', 'Avg_Finish_Rate', 'Full_Completion_Count',
    'Total_Prac_Attempts', 'Total_Interaction_Count', 'Active_Days',
    'Review_Ratio', 'Completion_View_Efficiency', 'View_Efficiency',
    'Prac_Efficiency', 'Composite_Engagement_Index', 'Avg_Daily_View_Time',
    'Avg_Daily_Prac', 'Avg_Daily_Interaction', 'Total_Notes', 'Total_Chkpts',
    'Total_Continue', 'Rewind_Ratio', 'Forward_Ratio', 'SpeedChange_Ratio',
    'PostReview_Ratio', 'Note_Ratio', 'Chkpt_Ratio', 'Continue_Ratio',
    'Revisit_Intensity', 'Total_Adjustment_Ratio', 'Focus_Play_Active',
    'Focus_Pause_Control', 'Focus_Speed_Adjust', 'Focus_Seek',
    'Focus_Review_Checkpoints', 'Focus_Continue', 'Focus_Notes',
    'Focus_Browse_UI', 'Focus_Questions', 'Speed_Correctness_Index',
    'Median_Exam_Ans_Time', 'Total_Correct', 'Avg_Prac_Score_Rate',
    'Avg_Prac_Efficiency', 'Prac_Score_StdDev', 'Avg_Exam_Correctness',
    'Exam_Efficiency', 'Total_Exam_Count', 'Prac_Improvement',
    'Practice_Exam_Consistency', 'Avg_Item_Error_Rate', 'Error_Streak_Max']
]

CLUSTER_NAME_MAP = {
    0: "高投入策略型",
    1: "低投入被動型"
}

def analyze_math(df):
    identity_cols = df.columns.difference(STAGE1_FEATURES_ALL)
    identity_df = df[identity_cols]

    X = df.copy()

    for col in STAGE1_FEATURES_ALL:
        X[col + "_missing"] = X[col].isna().astype(int)
    X[STAGE1_FEATURES_ALL] = X[STAGE1_FEATURES_ALL].fillna(0)

    X_stage1 = X[STAGE1_FEATURES_ALL]
    X_stage1_scaled = scaler_stage1.transform(X_stage1)

    outlier_preds = model_stage1.predict(X_stage1_scaled)

    normal_groups = {
        "高投入策略型-正常": [],
        "高投入策略型-高風險": [],
        "低投入被動型-正常": [],
        "低投入被動型-高風險": []
    }
    outlier_records = []

    # Stage2: 高風險用全特徵
    X_stage2 = X[STAGE2_FEATURES_ALL].fillna(0)
    X_stage2_scaled = scaler_stage2.transform(X_stage2)

    # 只取 PCA 特徵計算群中心距離
    n_pcs = cluster_centers.shape[1]
    X_for_cluster = X_stage1_scaled[:, :n_pcs]

    for idx, is_outlier in enumerate(outlier_preds):
        student_identity = identity_df.iloc[idx].to_dict()

        if is_outlier == 1:
            shap_vals = explainer_stage1.shap_values(X_stage1_scaled[idx:idx+1])[0]
            shap_df = pd.DataFrame({
                "Feature": [FEATURE_NAME_MAP.get(f, f) for f in STAGE1_FEATURES_ALL],
                "SHAP_Value": shap_vals
            }).sort_values(by="SHAP_Value", key=np.abs, ascending=False)

            record = student_identity.copy()
            record.update({
                "status": "離群學生",
                "SHAP_Stage1": shap_df.head(1)
            })
            outlier_records.append(record)
        else:
            dists = np.linalg.norm(cluster_centers - X_for_cluster[idx], axis=1)
            cluster_label = int(np.argmin(dists))
            cluster_name = CLUSTER_NAME_MAP[cluster_label]

            is_risk = model_stage2.predict(X_stage2_scaled[idx:idx+1])[0]
            risk_prob = float(model_stage2.predict_proba(X_stage2_scaled[idx:idx+1])[0, 1])

            record = student_identity.copy()
            record.update({
                "status": "正常學生",
                "cluster": cluster_name,
                "Risk": "高風險" if is_risk == 1 else "正常",
                "Risk_Probability": risk_prob
            })

            if is_risk == 1:
                shap_vals2 = explainer_stage2.shap_values(X_stage2_scaled[idx:idx+1])[0]
                shap_df2 = pd.DataFrame({
                    "Feature": [FEATURE_NAME_MAP.get(f, f) for f in STAGE2_FEATURES_ALL],
                    "SHAP_Value": shap_vals2
                }).sort_values(by="SHAP_Value", key=np.abs, ascending=False)
                record["SHAP_Stage2"] = shap_df2.head(1)

            key = f"{cluster_name}-高風險" if is_risk == 1 else f"{cluster_name}-正常"
            normal_groups[key].append(record)

    normal_group_dfs = {cid: pd.DataFrame(records) for cid, records in normal_groups.items() if records}
    outlier_df = pd.DataFrame(outlier_records)

    return normal_group_dfs, outlier_df
