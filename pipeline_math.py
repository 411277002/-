import joblib
import pandas as pd

stage1_scaler = joblib.load("models/math_scaler_stage1.pkl")
stage2_scaler = joblib.load("models/math_scaler_stage2.pkl")
stage1 = joblib.load("models/math_model_stage1.pkl")
stage2 = joblib.load("models/math_model_stage2.pkl")
explainer = joblib.load("models/importance_stage1.pkl")

FEATURES = [
    'Total_View_Duration', 'Avg_Finish_Rate', 'Full_Completion_Count',
    'Total_Prac_Attempts', 'Total_Interaction_Count', 'Active_Days',
    'Review_Ratio', 'Completion_View_Efficiency', 'View_Efficiency',
    'Prac_Efficiency', 'Composite_Engagement_Index', 'Avg_Daily_View_Time',
    'Avg_Daily_Prac', 'Avg_Daily_Interaction', 'Total_Notes', 'Total_Chkpts',
    'Total_Continue', 'Rewind_Ratio',
    'Forward_Ratio', 'SpeedChange_Ratio', 'PostReview_Ratio', 'Note_Ratio',
    'Chkpt_Ratio', 'Continue_Ratio', 'Revisit_Intensity',
    'Total_Adjustment_Ratio', 'Focus_Play_Active', 'Focus_Pause_Control',
    'Focus_Speed_Adjust', 'Focus_Seek', 'Focus_Review_Checkpoints',
    'Focus_Continue', 'Focus_Notes', 'Focus_Browse_UI', 'Focus_Questions',
    'Speed_Correctness_Index', 'Median_Exam_Ans_Time', 'Total_Correct',
    'Avg_Prac_Score_Rate', 'Avg_Prac_Efficiency',
    'Prac_Score_StdDev', 'Avg_Exam_Correctness', 'Exam_Efficiency',
    'Total_Exam_Count', 'Prac_Improvement', 'Practice_Exam_Consistency',
    'Avg_Item_Error_Rate', 'Error_Streak_Max'
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

CLUSTER_NAME_MAP = {
    0: "穩定型",
    1: "待觀察型",
    2: "努力型"
}

feature_rank_map = {row['feature']: idx for idx, row in explainer.iterrows()}

def analyze_math(df):
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
            max_feature = explainer.iloc[0]['feature']
            reason_feature = FEATURE_NAME_MAP[max_feature]

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