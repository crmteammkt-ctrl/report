# load_data.py
import os
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILE = os.path.join(BASE_DIR, "data", "data.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize date
    if "Ngày" in df.columns:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df = df.dropna(subset=["Ngày"])

        # derived columns for pivot
        df["Year"] = df["Ngày"].dt.year
        df["Month"] = df["Ngày"].dt.month
        df["YearMonth"] = df["Ngày"].dt.to_period("M").astype(str)

    # numeric columns you already use
    for c in ["Tổng_Gross", "Tổng_Net"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_resource
def _load_parquet_cached(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _normalize(df)


def get_active_data() -> pd.DataFrame:
    """
    - Nếu đã có st.session_state["active_df"] => trả luôn (KHÔNG load lại)
    - Nếu chưa có => load từ cache_resource 1 lần rồi gán vào session_state
    """
    if "active_df" in st.session_state and isinstance(st.session_state["active_df"], pd.DataFrame):
        return st.session_state["active_df"]

    if not os.path.exists(PARQUET_FILE):
        st.error(f"Không thấy file dữ liệu: {PARQUET_FILE}")
        st.stop()

    df = _load_parquet_cached(PARQUET_FILE)
    st.session_state["active_df"] = df
    st.session_state["active_source"] = "default"
    return df


def set_active_data(df: pd.DataFrame, source: str = "upload"):
    """
    Khi upload parquet mới:
    - Gán vào session_state để toàn bộ pages dùng chung ngay
    - Không cache (vì file upload khác nhau)
    """
    if df is None or df.empty:
        return

    df = _normalize(df)
    st.session_state["active_df"] = df
    st.session_state["active_source"] = source


def first_purchase(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = get_active_data()

    if "Số_điện_thoại" not in df.columns or "Ngày" not in df.columns:
        return pd.DataFrame(columns=["Số_điện_thoại", "First_Date"])

    fp = (
        df.groupby("Số_điện_thoại", as_index=False)["Ngày"]
        .min()
        .rename(columns={"Ngày": "First_Date"})
    )
    return fp
