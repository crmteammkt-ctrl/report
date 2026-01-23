import streamlit as st
from load_data import get_active_data, set_active_data
from pivot_builder import render_pivot_builder
import pandas as pd

st.set_page_config(page_title="Pivot Builder (Parquet)", layout="wide")
st.title("📊 Pivot Builder (App mới - dùng Parquet)")

with st.sidebar:
    st.header("⚙️ Data source")
    st.caption("Dùng data/data.parquet (giống app cũ), hoặc upload parquet khác.")

    uploaded = st.file_uploader("Upload parquet (optional)", type=["parquet"])
    if uploaded is not None:
        try:
            up_df = pd.read_parquet(uploaded)
            set_active_data(up_df, source="upload")
            st.success("Đã load parquet upload vào session.")
        except Exception as e:
            st.error(f"Không đọc được parquet upload: {e}")

    if st.button("🧹 Clear cache + reload", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        for k in ["active_df", "active_source"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

df = get_active_data()

st.success(f"Source: {st.session_state.get('active_source','?')} | Rows: {len(df):,} | Cols: {len(df.columns)}")
render_pivot_builder(df)
