import streamlit as st
import pandas as pd
import numpy as np
from utils_excel import to_excel_bytes

# =========================================================
# MEASURE BUILDER
# =========================================================
def measure_builder(df: pd.DataFrame) -> dict:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if "user_measures" not in st.session_state:
        st.session_state["user_measures"] = {}

    with st.sidebar:
        st.subheader("🧮 Measures (tự tạo)")
        enable = st.checkbox("Bật tạo measure", value=True, key="me_enable")

        # ---- Add nhanh measure tỷ lệ CK đúng ----
        can_add_ck = ("Tổng_Gross" in df.columns) and ("Tổng_Net" in df.columns)
        if st.button(
            "➕ Add nhanh: Tỷ lệ CK = (Gross - Net) / Gross",
            use_container_width=True,
            disabled=not can_add_ck,
            key="me_quick_ck",
        ):
            st.session_state["user_measures"]["Tỷ lệ CK (Gross-Net)/Gross"] = {
                "type": "ck_rate",
                "gross": "Tổng_Gross",
                "net": "Tổng_Net",
            }
            st.success("Đã thêm measure Tỷ lệ CK!")

        if not can_add_ck:
            st.caption("⚠ Không thấy 'Tổng_Gross' hoặc 'Tổng_Net' nên nút add nhanh bị tắt.")

    if not enable:
        return st.session_state["user_measures"]

    with st.sidebar:
        st.markdown("### ➕ Tạo measure mới")
        name = st.text_input("Tên measure", placeholder="VD: Tỷ lệ CK đúng", key="me_name")

        mtype = st.selectbox(
            "Loại measure",
            ["SUM", "COUNT", "NUNIQUE", "RATIO(SUM/SUM)", "WEIGHTED_AVG", "CK_RATE (Gross-Net)/Gross"],
            index=0,
            key="me_type",
        )

        spec = None

        if mtype == "SUM":
            if numeric_cols:
                col = st.selectbox("Cột", numeric_cols, index=0, key="me_sum_col")
                spec = {"type": "sum", "col": col}
            else:
                st.warning("Không có cột numeric.")

        elif mtype == "COUNT":
            col = st.selectbox("Cột (count non-null)", df.columns.tolist(), index=0, key="me_cnt_col")
            spec = {"type": "count", "col": col}

        elif mtype == "NUNIQUE":
            col = st.selectbox("Cột (unique)", df.columns.tolist(), index=0, key="me_nu_col")
            spec = {"type": "nunique", "col": col}

        elif mtype == "RATIO(SUM/SUM)":
            if numeric_cols:
                num = st.selectbox("Tử số (SUM)", numeric_cols, index=0, key="me_ratio_num")
                den = st.selectbox("Mẫu số (SUM)", numeric_cols, index=0, key="me_ratio_den")
                spec = {"type": "ratio", "num": num, "den": den}
            else:
                st.warning("Không có cột numeric.")

        elif mtype == "WEIGHTED_AVG":
            if numeric_cols:
                x = st.selectbox("Giá trị X", numeric_cols, index=0, key="me_wavg_x")
                w = st.selectbox("Trọng số W", numeric_cols, index=0, key="me_wavg_w")
                spec = {"type": "wavg", "val": x, "w": w}
            else:
                st.warning("Không có cột numeric.")

        elif mtype == "CK_RATE (Gross-Net)/Gross":
            if numeric_cols:
                gross = st.selectbox(
                    "Gross (SUM)",
                    options=numeric_cols,
                    index=numeric_cols.index("Tổng_Gross") if "Tổng_Gross" in numeric_cols else 0,
                    key="me_ck_gross",
                )
                net = st.selectbox(
                    "Net (SUM)",
                    options=numeric_cols,
                    index=numeric_cols.index("Tổng_Net") if "Tổng_Net" in numeric_cols else 0,
                    key="me_ck_net",
                )
                spec = {"type": "ck_rate", "gross": gross, "net": net}
            else:
                st.warning("Không có cột numeric.")

        add = st.button("➕ Add measure", use_container_width=True, disabled=(not name or spec is None), key="me_add")

    if add and name and spec:
        st.session_state["user_measures"][name] = spec
        st.success(f"Đã thêm measure: {name}")

    with st.sidebar:
        if st.session_state["user_measures"]:
            st.markdown("### 📌 Measures đã tạo")
            for k, v in st.session_state["user_measures"].items():
                st.write(f"- **{k}**: `{v}`")

            if st.button("🗑️ Xoá tất cả measures", use_container_width=True, key="me_clear"):
                st.session_state["user_measures"] = {}
                st.rerun()

    return st.session_state["user_measures"]


def compute_measures(dff: pd.DataFrame, group_keys: list[str], measures: dict) -> pd.DataFrame:
    g = dff.groupby(group_keys, dropna=False)
    out = pd.DataFrame(index=g.size().index)

    for name, spec in measures.items():
        t = spec["type"]

        if t == "sum":
            out[name] = g[spec["col"]].sum(min_count=1)

        elif t == "count":
            out[name] = g[spec["col"]].count()

        elif t == "nunique":
            out[name] = g[spec["col"]].nunique(dropna=True)

        elif t == "ratio":
            num = g[spec["num"]].sum(min_count=1)
            den = g[spec["den"]].sum(min_count=1)
            out[name] = (num / den).replace([np.inf, -np.inf], np.nan)

        elif t == "wavg":
            x = spec["val"]
            w = spec["w"]
            num = (dff[x] * dff[w]).groupby(group_keys).sum(min_count=1)
            den = dff[w].groupby(group_keys).sum(min_count=1)
            out[name] = (num / den).replace([np.inf, -np.inf], np.nan)

        elif t == "ck_rate":
            gross_col = spec["gross"]
            net_col = spec["net"]
            gross_sum = g[gross_col].sum(min_count=1)
            net_sum = g[net_col].sum(min_count=1)
            out[name] = ((gross_sum - net_sum) / gross_sum).replace([np.inf, -np.inf], np.nan)

    return out.reset_index()


# =========================================================
# PIVOT UTILS
# =========================================================
AGG_CHOICES = ["Sum", "Mean", "Count", "Nunique", "Min", "Max"]

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _fmt_number(x, decimals=0, thousand_sep=True):
    if pd.isna(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return ""
    if decimals <= 0:
        return f"{v:,.0f}" if thousand_sep else f"{v:.0f}"
    return f"{v:,.{decimals}f}" if thousand_sep else f"{v:.{decimals}f}"

def apply_format_rules(
    pv: pd.DataFrame,
    rules: dict,
    thousand_sep_default=True,
    decimals_default=0
) -> pd.DataFrame:
    """
    rules = {
      "colname": {"type":"number|percent|currency|raw", "decimals":2, "thousand":True, "prefix":"₫ ", "suffix":""}
    }
    """
    out = pv.copy()

    for c in out.columns:
        r = rules.get(c)

        # default: format numeric as number
        if not r:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda x: _fmt_number(x, decimals_default, thousand_sep_default))
            continue

        ftype = r.get("type", "number")
        dec = int(r.get("decimals", decimals_default))
        thou = bool(r.get("thousand", thousand_sep_default))
        prefix = r.get("prefix", "")
        suffix = r.get("suffix", "")

        if ftype == "raw":
            out[c] = out[c].astype(str)
            continue

        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(str)
            continue

        if ftype == "percent":
            out[c] = out[c].map(
                lambda x: "" if pd.isna(x) else (f"{float(x)*100:,.{dec}f}%" if thou else f"{float(x)*100:.{dec}f}%")
            )
        elif ftype == "currency":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{prefix}{_fmt_number(x, dec, thou)}{suffix}")
        else:  # number
            out[c] = out[c].map(lambda x: _fmt_number(x, dec, thou))

    return out

def _safe_category_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if _is_numeric(df[c])]
    dims = []

    # Cho phép dùng "Ngày" làm dimension
    if "Ngày" in df.columns and _is_datetime(df["Ngày"]):
        dims.append("Ngày")

    # Các cột không phải numeric
    for c in df.columns:
        if c in numeric_cols:
            continue
        if c == "Ngày":
            continue
        dims.append(c)

    # Ưu tiên các cột thời gian dẫn xuất nếu có
    for c in ["YearMonth", "Year", "Month"]:
        if c in df.columns and c not in dims:
            dims.insert(0, c)

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


# =========================================================
# FILTERS
# =========================================================
def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    # Date filter
    if "Ngày" in dff.columns and _is_datetime(dff["Ngày"]):
        with st.sidebar:
            st.subheader("📅 Date filter")
            min_d = dff["Ngày"].min().date()
            max_d = dff["Ngày"].max().date()
            default_start = max(min_d, (max_d - pd.Timedelta(days=30)))

            start, end = st.date_input(
                "Range",
                value=(default_start, max_d),
                min_value=min_d,
                max_value=max_d,
                key="date_range",
            )

        dff = dff[(dff["Ngày"].dt.date >= start) & (dff["Ngày"].dt.date <= end)]

    dims = _safe_category_columns(dff)

    with st.sidebar:
        st.subheader("🧰 Filters")
        st.caption("Chọn cột cần filter (tránh lag).")
        filter_cols = st.multiselect(
            "Filter columns",
            options=dims,
            default=[c for c in ["Brand", "Region", "LoaiCT"] if c in dims],
            key="filter_cols",
        )

    for c in filter_cols:
        nunq = dff[c].nunique(dropna=True)
        if nunq <= 0:
            continue

        # nhiều giá trị -> search rồi chọn nhiều
        if nunq > 200:
            with st.sidebar:
                st.write(f"🔎 Search & select: **{c}** (unique: {nunq:,})")

                keyword = st.text_input(
                    f"Tìm {c}",
                    value="",
                    placeholder="VD: NB12",
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
                else:
                    s2 = s[s.str.lower() == kw.lower()]
            else:
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
                dff = dff[dff[c].astype(str).isin(set(map(str, picked)))]

            continue

        # ít giá trị -> dropdown thường
        options = sorted(dff[c].dropna().unique().tolist(), key=lambda x: str(x))
        with st.sidebar:
            picked = st.multiselect(c, options=options, default=[], key=f"pick_small_{c}")
        if picked:
            dff = dff[dff[c].isin(picked)]

    return dff


# =========================================================
# MAIN RENDER
# =========================================================
def render_pivot_builder(df: pd.DataFrame):
    st.header("🧩 Pivot Builder")

    # 1) filter
    dff = _apply_filters(df)
    if dff.empty:
        st.warning("Sau filter không còn dữ liệu.")
        return

    # 2) measures UI (sidebar)
    user_measures = measure_builder(dff)

    dims = _safe_category_columns(dff)
    numeric_cols = [c for c in dff.columns if _is_numeric(dff[c])]

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        rows = st.multiselect("Rows (Group by)", options=dims, default=dims[:1] if dims else [], key="rows")
        cols = st.multiselect("Columns (Pivot)", options=["(None)"] + dims, default=["(None)"], key="cols")
        cols = [c for c in cols if c != "(None)"]

    with col2:
        use_measures = st.checkbox("✅ Dùng Measures (khuyên dùng cho tỷ lệ)", value=True, key="use_measures")

        if use_measures:
            measure_names = list(user_measures.keys())
            selected_measures = st.multiselect(
                "Measures",
                options=measure_names,
                default=measure_names[:1] if measure_names else [],
                key="selected_measures",
            )

            add_sum_cols = st.multiselect(
                "➕ Add columns (SUM)",
                options=numeric_cols,
                default=[c for c in ["Tổng_Gross", "Tổng_Net"] if c in numeric_cols],
                key="add_sum_cols",
            )

            # placeholders to avoid UnboundLocalError
            values, agg_name, fillna0 = [], "Sum", True
        else:
            selected_measures = []
            add_sum_cols = []

            values = st.multiselect(
                "Values",
                options=numeric_cols,
                default=[c for c in ["Tổng_Net"] if c in numeric_cols] or (numeric_cols[:1] if numeric_cols else []),
                key="values",
            )
            agg_name = st.selectbox("Aggregation", options=AGG_CHOICES, index=0, key="agg")
            fillna0 = st.checkbox("Fill NaN = 0", value=True, key="fillna0")

    optA, optB, optC, optD = st.columns([1, 1, 1, 1])
    with optA:
        show_total = st.checkbox("Show totals", value=False, key="show_total")
    with optB:
        pct_mode = st.selectbox("Percent mode", options=["None", "% of row", "% of column"], index=0, key="pct")
    with optC:
        thousand_sep = st.checkbox("Thousand separator", value=True, key="thou")
    with optD:
        decimals = st.number_input("Decimals", min_value=0, max_value=6, value=0, step=1, key="decimals")

    if not rows:
        st.warning("Chọn ít nhất 1 cột Rows.")
        return

    # 3) compute pivot
    if use_measures:
        if (not selected_measures) and (not add_sum_cols):
            st.warning("Bạn đang bật Measures nhưng chưa chọn Measure nào hoặc chưa chọn Add columns (SUM).")
            return

        group_keys = rows + (cols if cols else [])

        # gộp measures + SUM columns
        selected_specs = {m: user_measures[m] for m in selected_measures}

        for col in add_sum_cols:
            name = f"SUM({col})"
            if name not in selected_specs:
                selected_specs[name] = {"type": "sum", "col": col}

        out = compute_measures(dff, group_keys=group_keys, measures=selected_specs)
        value_cols = list(selected_specs.keys())

        if cols:
            pv = out.pivot_table(
                index=rows,
                columns=cols,
                values=value_cols,
                aggfunc="first",
                margins=show_total,
            )
            if isinstance(pv.columns, pd.MultiIndex):
                pv.columns = [" | ".join(map(str, t)) for t in pv.columns]
            pv = pv.reset_index()
        else:
            pv = out

    else:
        if not values:
            st.warning("Chọn ít nhất 1 cột Values.")
            return

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

        if isinstance(pv.columns, pd.MultiIndex):
            pv.columns = [
                " | ".join([str(x) for x in tup if x is not None and str(x) != ""])
                for tup in pv.columns
            ]
        else:
            pv.columns = pv.columns.map(str)

        pv = pv.reset_index()

    # 4) Percent mode (áp dụng cho numeric result)
    if pct_mode != "None":
        id_cols = set(rows)
        num_cols = [c for c in pv.columns if c not in id_cols and _is_numeric(pv[c])]
        if num_cols:
            arr = pv[num_cols].astype(float)
            if pct_mode == "% of row":
                denom = arr.sum(axis=1).replace(0, np.nan)
                pv[num_cols] = (arr.div(denom, axis=0) * 100).round(6)
            elif pct_mode == "% of column":
                denom = arr.sum(axis=0).replace(0, np.nan)
                pv[num_cols] = (arr.div(denom, axis=1) * 100).round(6)

    # =========================
    # FORMAT UI (per-column)
    # =========================
    with st.expander("🎨 Format cột (tuỳ chọn)", expanded=False):
        st.caption("Chọn format cho từng cột output. File Excel download vẫn giữ số thật (raw).")

        id_cols = set(rows)
        value_like_cols = [c for c in pv.columns if c not in id_cols]

        if "pivot_format_rules" not in st.session_state:
            st.session_state["pivot_format_rules"] = {}

        # default chung
        thou_default = st.checkbox("Default: Thousand separator", value=thousand_sep, key="fmt_default_thou")
        dec_default = st.number_input("Default: Decimals", 0, 6, int(decimals), 1, key="fmt_default_dec")

        # gợi ý chọn sẵn cột có chữ CK/Tỷ lệ
        suggest_default = [c for c in value_like_cols if ("ck" in str(c).lower() or "tỷ" in str(c).lower() or "ty" in str(c).lower())]
        target_cols = st.multiselect(
            "Chọn cột cần format riêng",
            options=value_like_cols,
            default=suggest_default,
            key="fmt_target_cols",
        )

        for col in target_cols:
            st.markdown(f"**{col}**")
            c1, c2, c3, c4 = st.columns([2, 1, 1, 2])

            # default type heuristic
            default_type = "percent" if ("ck" in str(col).lower() or "tỷ" in str(col).lower() or "ty" in str(col).lower()) else "number"

            with c1:
                ftype = st.selectbox(
                    f"Type - {col}",
                    options=["number", "percent", "currency", "raw"],
                    index=["number", "percent", "currency", "raw"].index(default_type),
                    key=f"fmt_type_{col}",
                )
            with c2:
                dec = st.number_input(f"Dec - {col}", 0, 6, dec_default, 1, key=f"fmt_dec_{col}")
            with c3:
                thou = st.checkbox(f"Thou - {col}", value=thou_default, key=f"fmt_thou_{col}")
            with c4:
                prefix_default = "₫ " if ftype == "currency" else ""
                prefix = st.text_input(f"Prefix - {col}", value=prefix_default, key=f"fmt_pre_{col}")
                suffix = st.text_input(f"Suffix - {col}", value="", key=f"fmt_suf_{col}")

            st.session_state["pivot_format_rules"][col] = {
                "type": ftype,
                "decimals": int(dec),
                "thousand": bool(thou),
                "prefix": prefix,
                "suffix": suffix,
            }

        if st.button("Reset format rules", use_container_width=True, key="fmt_reset"):
            st.session_state["pivot_format_rules"] = {}
            st.rerun()

    pv_display = apply_format_rules(
        pv,
        rules=st.session_state.get("pivot_format_rules", {}),
        thousand_sep_default=thousand_sep,
        decimals_default=int(decimals),
    )

    # 5) display + export
    st.subheader("✅ Kết quả")
    st.caption(f"Filtered rows: {len(dff):,} | Pivot rows: {len(pv):,}")

    st.dataframe(pv_display, use_container_width=True)

    # Download (raw pv, không format thành text)
    xlsx = to_excel_bytes(pv, sheet_name="Pivot")
    st.download_button(
        "⬇️ Download Pivot Excel",
        data=xlsx,
        file_name="pivot_table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("📈 Quick chart", expanded=False):
        id_cols = set(rows)
        # chart dùng số raw (pv) để đúng
        num_cols = [c for c in pv.columns if c not in id_cols and _is_numeric(pv[c])]
        if not num_cols:
            st.info("Không có cột numeric để vẽ chart.")
        else:
            ycol = st.selectbox("Y column", options=num_cols, key="chart_y")
            xcol = rows[0]
            chart_df = pv[[xcol, ycol]].dropna().copy()
            chart_df = chart_df.sort_values(by=ycol, ascending=False).head(30)
            st.bar_chart(chart_df.set_index(xcol)[ycol])
