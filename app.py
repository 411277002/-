import streamlit as st
import pandas as pd
from pipeline import analyze_file
import shap
import chardet
import io
st.title("行為分析測試")


def read_csv_safely(uploaded_file):
    raw_data = uploaded_file.read()

    detect_result = chardet.detect(raw_data)
    encoding = detect_result["encoding"]

    try:
        # 嘗試使用新版 pandas 參數
        df = pd.read_csv(
            io.BytesIO(raw_data),
            encoding=encoding,
            engine="python",
            errors="ignore"
        )
    except TypeError:
        # 舊版 pandas fallback
        df = pd.read_csv(
            io.BytesIO(raw_data),
            encoding=encoding,
            engine="python"
        )
    return df


CLUSTER_NAME_MAP = {
    0: "精準型",
    1: "低投入型",
    2: "努力型"
}


uploaded_file = st.file_uploader("請上傳 CSV 檔", type=["csv"])

if uploaded_file:
    df = read_csv_safely(uploaded_file)
    st.dataframe(df)

    if st.button("開始分析"):
        normal_group_dfs, outlier_df = analyze_file(df)
        st.success("分析完成！")
    

        for cluster_id, group_df in normal_group_dfs.items():
            cluster_name = CLUSTER_NAME_MAP[cluster_id]

            st.write(f"### {cluster_name}")

            st.dataframe(group_df)

            csv_data = group_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=f"下載 {cluster_name} CSV",
                data=csv_data,
                file_name=f"cluster_{cluster_id}.csv",
                mime="text/csv",
                key=f"download_cluster_{cluster_id}"
            )
        

        if not outlier_df.empty:
            st.subheader("離群學生")
            st.dataframe(outlier_df)

            csv_outlier = outlier_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下載離群學生 CSV",
                data=csv_outlier,
                file_name="outlier.csv",
                mime="text/csv",
                key="download_outlier"
            )