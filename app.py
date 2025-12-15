import streamlit as st
import pandas as pd
import chardet
import io

from pipeline_cn import analyze_cn
from pipeline_math import analyze_math
from radar_cn import plot_radar_cn
from radar_math import plot_radar_math

# =============================
# 基本設定
# =============================
st.set_page_config(
    page_title="學生學習行為分析系統",
    layout="wide"
)

# Session State
if "page" not in st.session_state:
    st.session_state.page = "home"
if "subject" not in st.session_state:
    st.session_state.subject = None
if "df_uploaded" not in st.session_state:
    st.session_state.df_uploaded = None

# =============================
# CSS 美化
# =============================
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 10px;
}
.sub-title {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}
.card {
    padding: 24px;
    border-radius: 14px;
    background-color: #f9fafb;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("## 📌 功能選單")

    if st.button("學科選擇", use_container_width=True):
        st.session_state.page = "home"

    if st.button("上傳檔案", use_container_width=True):
        st.session_state.page = "upload"

    if st.button("行為評估", use_container_width=True):
        if st.session_state.df_uploaded is None:
            st.warning("請先上傳檔案")
        else:
            st.session_state.page = "behavior"

    if st.button("雷達圖", use_container_width=True):
        if st.session_state.df_uploaded is None:
            st.warning("請先上傳檔案")
        else:
            st.session_state.page = "radar"

# =============================
# 首頁：學科選擇
# =============================
if st.session_state.page == "home":
    st.markdown('<div class="main-title">學生學習行為分析系統</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">請選擇欲分析的學科</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("國語文", use_container_width=True):
            st.session_state.subject = "國語文"
            st.session_state.page = "upload"

    with col2:
        if st.button("數學", use_container_width=True):
            st.session_state.subject = "數學"
            st.session_state.page = "upload"

# =============================
# 上傳頁
# =============================
elif st.session_state.page == "upload":
    st.markdown(
        f'<div class="main-title">{st.session_state.subject}｜資料上傳</div>',
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("請上傳檔案", type=["csv", "xlsx", "xls"])

    def read_file_safely(uploaded_file): 
        uploaded_file.seek(0)
        filename = uploaded_file.name.lower()

        # Excel 檔
        if filename.endswith((".xlsx", ".xls")):
            try:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
                return df
            except Exception as e:
                st.error(f"Excel 讀取錯誤：{e}")
                return None

        # CSV 檔
        try:
            uploaded_file.seek(0)
            raw_data = uploaded_file.read()
            detect_result = chardet.detect(raw_data)
            encoding = detect_result["encoding"] or "utf-8"

            uploaded_file.seek(0)
            df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding, engine="python")
            return df
        except Exception as e:
            st.error(f"CSV 讀取錯誤：{e}")
            return None

    if uploaded_file:
        df = read_file_safely(uploaded_file)
        if df is not None:
            st.session_state.df_uploaded = df  # 存到 session_state 方便其他頁面使用
            st.success("檔案上傳完成！")
            st.dataframe(df)

# =============================
# 行為評估
# =============================
elif st.session_state.page == "behavior":
    st.markdown('<div class="main-title">行為評估結果</div>', unsafe_allow_html=True)

    df = st.session_state.df_uploaded

    if st.session_state.subject == "國語文":
        normal_groups, outlier_df = analyze_cn(df)
    else:
        normal_groups, outlier_df = analyze_math(df)

    col1, col2 = st.columns(2)
    col1.metric("總學生數", len(df))
    col2.metric("分析科目", st.session_state.subject)

    tab_names = [str(k) for k in normal_groups.keys()] + ["離群學生"]
    tabs = st.tabs(tab_names)

    for tab, (cid, gdf) in zip(tabs, normal_groups.items()):
        with tab:
            st.subheader(str(cid))
            st.write(f"人數：{len(gdf)}")
            st.dataframe(gdf, use_container_width=True)

            st.download_button(
                label="下載 CSV",
                data=gdf.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{str(cid)}.csv",
                mime="text/csv",
                key=f"download_{st.session_state.subject}_cluster_{cid}"
            )

    # ===== 離群學生 =====
    if not outlier_df.empty:
        with tabs[-1]:
            st.subheader("離群學生")
            st.write(f"人數：{len(outlier_df)}")
            st.dataframe(outlier_df, use_container_width=True)

            st.download_button(
                label="下載 CSV",
                data=outlier_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="outlier_students.csv",
                mime="text/csv",
                key=f"download_{st.session_state.subject}_outlier"
            )

# =============================
# 雷達圖
# =============================
elif st.session_state.page == "radar":
    st.markdown('<div class="main-title">學生學習雷達圖</div>', unsafe_allow_html=True)

    df = st.session_state.df_uploaded
    left, right = st.columns([1, 2])

    with left:
        user_id = st.number_input(
            "🎯 請輸入學生編號 (user_sn)",
            min_value=int(df["user_sn"].min()),
            max_value=int(df["user_sn"].max()),
            step=1
        )
        generate = st.button("生成雷達圖")

    with right:
        if generate:
            try:
                fig = (
                    plot_radar_cn(df, user_id)
                    if st.session_state.subject == "國語文"
                    else plot_radar_math(df, user_id)
                )
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))
