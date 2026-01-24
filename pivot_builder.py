import streamlit as st
import pandas as pd
import numpy as np
from utils_excel import to_excel_bytes

AGG_CHOICES = ["Sum", "Mean", "Count", "Nunique", "Min", "Max"]


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _format_numeric_for_display(
    df: pd.DataFrame, thousand_sep: bool = True, decimals: int = 0
) -> pd.DataFrame:
    out = df.copy()

    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return f"{x:,}" if thousand_sep else str(x)
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return ""
            if decimals <= 0:
                return f"{x:,.0f}" if thousand_sep else f"{x:.0f}"
            return f"{x:,.{decimals}f}" if thousand_sep else f"{x:.{decimals}f}"
        return str(x)

    for c in out.columns:
        if _is_numeric(out[c]):
            out[c] = out[c].map(fmt)
    return out


def _safe_category_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if _is_numeric(df[c])]

    dims = []

    # 🔹 Cho phép dùng cột Ngày (datetime) làm dimension
    if "Ngày" in df.columns and _is_datetime(df["Ngày"]):
        dims.append("Ngày")

    # 🔹 Các cột text/category khác
    for c in df.columns:
        if c in numeric_cols:
            continue
        if c == "Ngày":
            continue
        dims.append(c)

    # 🔹 Ưu tiên các cột thời gian dẫn xuất
    for c in ["YearMonth", "Year", "Month"]:
        if c in df.columns and c not in dims:
            dims.insert(0, c)

    # remove duplicates, giữ thứ tự
    seen, out = set(), []
    for c in dims:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out



def _aggfunc(name: str):
    if name == "Sum":
        return np.sum
    if name == "Mean":
        return np.mean
    if name == "Count":
        return "count"
    if name == "Nunique":
        return pd.Series.nunique
    if name == "Min":
        return np.min
    if name == "Max":
        return np.max
    return np.sum


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    # -----------------------
    # Date filter
    # -----------------------
    if "Ngày" in dff.columns and _is_datetime(dff["Ngày"]):
        with st.sidebar:
            st.subheader("📅 Date filter")
            min_d = dff["Ngày"].min().date()
            max_d = dff["Ngày"].max().date()

            # mặc định 30 ngày gần nhất
            default_start = max(min_d, (max_d - pd.Timedelta(days=30)))
            start, end = st.date_input(
                "Range",
                value=(default_start, max_d),
                min_value=min_d,
                max_value=max_d,
            )

        dff = dff[(dff["Ngày"].dt.date >= start) & (dff["Ngày"].dt.date <= end)]

    dims = _safe_category_columns(dff)

    # -----------------------
    # Choose which columns to filter
    # -----------------------
    with st.sidebar:
        st.subheader("🧰 Filters")
        st.caption("Chọn cột cần filter (tránh lag).")
        filter_cols = st.multiselect(
            "Filter columns",
            options=dims,
            default=[c for c in ["Brand", "Region", "LoaiCT"] if c in dims],
        )

    # -----------------------
    # Apply filters
    # -----------------------
    for c in filter_cols:
        nunq = dff[c].nunique(dropna=True)
        if nunq <= 0:
            continue

        # === NHIỀU GIÁ TRỊ: search rồi chọn nhiều ===
        if nunq > 200:
            with st.sidebar:
                st.write(f"🔎 Search & select: **{c}** (unique: {nunq:,})")

                keyword = st.text_input(
                    f"Tìm {c}",
                    value="",
                    placeholder="VD: NB12 (sẽ hiện danh sách khớp để chọn nhiều)",
                    key=f"kw_{c}",
                )

                match_mode = st.selectbox(
                    f"Match mode - {c}",
                    options=["contains", "starts_with", "equals"],
                    index=0,
                    key=f"mm_{c}",
                )

                limit = st.number_input(
                    f"Max results - {c}",
                    min_value=20,
                    max_value=500,
                    value=200,
                    step=20,
                    key=f"lim_{c}",
                )

            uniques = dff[c].dropna().astype(str).unique()
            s = pd.Series(uniques, dtype="string")

            if keyword.strip():
                kw = keyword.strip()
                if match_mode == "contains":
                    s2 = s[s.str.contains(kw, case=False, na=False)]
                elif match_mode == "starts_with":
                    s2 = s[s.str.startswith(kw, na=False)]
                else:  # equals
                    s2 = s[s.str.lower() == kw.lower()]
            else:
                # chưa gõ thì không show hàng nghìn options
                s2 = pd.Series([], dtype="string")

            options = s2.sort_values().head(int(limit)).tolist()

            with st.sidebar:
                picked = st.multiselect(
                    f"Chọn nhiều {c} (từ kết quả tìm kiếm)",
                    options=options,
                    default=[],
                    key=f"pick_{c}",
                )
                st.caption(f"Kết quả khớp: {len(s2):,} | đang hiển thị: {len(options):,}")

            if picked:
                picked_set = set(str(x) for x in picked)
                dff = dff[dff[c].astype(str).isin(picked_set)]

            continue

        # === ÍT GIÁ TRỊ: dropdown bình thường ===
        options = dff[c].dropna().unique().tolist()
        options = sorted(options, key=lambda x: str(x))

        with st.sidebar:
            picked = st.multiselect(c, options=options, default=[])

        if picked:
            dff = dff[dff[c].isin(picked)]

    return dff


def render_pivot_builder(df: pd.DataFrame):
    st.header("🧩 Pivot Builder")

    # Apply filters first
    dff = _apply_filters(df)
    if dff.empty:
        st.warning("Sau filter không còn dữ liệu.")
        return

    dims = _safe_category_columns(dff)
    numeric_cols = [c for c in dff.columns if _is_numeric(dff[c])]

    if not numeric_cols:
        st.error("Không có cột numeric để làm Values. Kiểm tra cột Tổng_Gross/Tổng_Net hoặc convert số.")
        return

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        rows = st.multiselect("Rows (Group by)", options=dims, default=dims[:1] if dims else [])
        cols = st.multiselect("Columns (Pivot)", options=["(None)"] + dims, default=["(None)"])
        cols = [c for c in cols if c != "(None)"]

    with col2:
        values = st.multiselect(
            "Values",
            options=numeric_cols,
            default=[c for c in ["Tổng_Net"] if c in numeric_cols] or numeric_cols[:1],
        )
        agg_name = st.selectbox("Aggregation", options=AGG_CHOICES, index=0)
        fillna0 = st.checkbox("Fill NaN = 0", value=True)

    optA, optB, optC, optD = st.columns([1, 1, 1, 1])
    with optA:
        show_total = st.checkbox("Show totals", value=False)
    with optB:
        pct_mode = st.selectbox("Percent mode", options=["None", "% of row", "% of column"], index=0)
    with optC:
        thousand_sep = st.checkbox("Thousand separator", value=True)
    with optD:
        decimals = st.number_input("Decimals", min_value=0, max_value=4, value=0, step=1)

    if not rows:
        st.warning("Chọn ít nhất 1 cột Rows.")
        return
    if not values:
        st.warning("Chọn ít nhất 1 cột Values.")
        return

    try:
        pv = pd.pivot_table(
            dff,
            index=rows,
            columns=cols if cols else None,
            values=values,
            aggfunc=_aggfunc(agg_name),
            fill_value=0 if fillna0 else None,
            margins=show_total,
            margins_name="Total",
            dropna=False,
        )
    except Exception as e:
        st.error(f"Pivot error: {e}")
        return

    # flatten columns if MultiIndex
    if isinstance(pv.columns, pd.MultiIndex):
        pv.columns = [
            " | ".join([str(x) for x in tup if x is not None and str(x) != ""])
            for tup in pv.columns
        ]
    else:
        pv.columns = pv.columns.map(str)

    pv = pv.reset_index()

    # Percent mode
    if pct_mode != "None":
        num_cols = [c for c in pv.columns if c not in rows and _is_numeric(pv[c])]
        if num_cols:
            arr = pv[num_cols].astype(float)
            if pct_mode == "% of row":
                denom = arr.sum(axis=1).replace(0, np.nan)
                pv[num_cols] = (arr.div(denom, axis=0) * 100).round(2)
            elif pct_mode == "% of column":
                denom = arr.sum(axis=0).replace(0, np.nan)
                pv[num_cols] = (arr.div(denom, axis=1) * 100).round(2)

    st.subheader("✅ Kết quả")
    st.caption(f"Filtered rows: {len(dff):,} | Pivot rows: {len(pv):,}")

    st.dataframe(
        _format_numeric_for_display(pv, thousand_sep=thousand_sep, decimals=int(decimals)),
        use_container_width=True,
    )

    # Download
    xlsx = to_excel_bytes(pv, sheet_name="Pivot")
    st.download_button(
        "⬇️ Download Pivot Excel",
        data=xlsx,
        file_name="pivot_table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("📈 Quick chart", expanded=False):
        num_cols = [c for c in pv.columns if c not in rows and _is_numeric(pv[c])]
        if not num_cols:
            st.info("Không có cột numeric để vẽ chart.")
        else:
            ycol = st.selectbox("Y column", options=num_cols)
            xcol = rows[0]
            chart_df = pv[[xcol, ycol]].dropna().copy()
            chart_df = chart_df.sort_values(by=ycol, ascending=False).head(30)
            st.bar_chart(chart_df.set_index(xcol)[ycol])
