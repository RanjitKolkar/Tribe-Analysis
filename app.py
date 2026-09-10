from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Tribe Data Analyzer", layout="wide")


def _norm(text: object) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip().upper()


def _clean_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).strip().upper().startswith("UNNAMED")]
    return df[keep_cols].copy()


def _find_sheet(sheet_names: List[str], target: str) -> str | None:
    target_n = _norm(target)
    exact = {(_norm(s), s) for s in sheet_names}
    for normalized, original in exact:
        if normalized == target_n:
            return original
    for s in sheet_names:
        if target_n in _norm(s):
            return s
    return None


def _tribe_name_from_file(path: Path) -> str:
    name = path.stem.replace("Tribe List", "").replace("tribe list", "").strip(" -_")
    return name or path.stem


def _safe_read_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(how="all")
    return df


@st.cache_data(show_spinner=False)
def load_tribe_workbook(path: str) -> Dict[str, pd.DataFrame]:
    file_path = Path(path)
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    tribe = _tribe_name_from_file(file_path)
    result: Dict[str, pd.DataFrame] = {}

    # Sheet1: base roster
    base_sheet = _find_sheet(sheets, "Sheet1")
    if base_sheet:
        base = _clean_unnamed_columns(_safe_read_excel(file_path, base_sheet))
        rename_map = {c: str(c).strip() for c in base.columns}
        base = base.rename(columns=rename_map)

        age_col = "age" if "age" in base.columns else "Age" if "Age" in base.columns else None
        sex_col = "Sex" if "Sex" in base.columns else "SEX" if "SEX" in base.columns else None

        slim = pd.DataFrame(index=base.index)
        slim["tribe"] = tribe
        slim["name"] = base.get("Full Name", pd.Series(dtype=object))
        slim["sex"] = base[sex_col] if sex_col else np.nan
        slim["age"] = pd.to_numeric(base[age_col], errors="coerce") if age_col else np.nan
        result["roster"] = slim

    # Males and Females listing for data completeness checks
    for gender_sheet in ["Males", "Females"]:
        actual = _find_sheet(sheets, gender_sheet)
        if actual:
            df = _clean_unnamed_columns(_safe_read_excel(file_path, actual))
            df["tribe"] = tribe
            result[gender_sheet.lower()] = df

    # Measurements
    measurement_sheet = _find_sheet(sheets, "Measurements")
    if measurement_sheet:
        m = _clean_unnamed_columns(_safe_read_excel(file_path, measurement_sheet))
        m["tribe"] = tribe
        for col in m.columns:
            if col not in {"ID Assigned", "tribe"}:
                m[col] = pd.to_numeric(m[col], errors="coerce")
        result["measurements"] = m

    # Questionnaire (possible trailing spaces in name)
    q_sheet = _find_sheet(sheets, "Questionnaire")
    if q_sheet:
        q = _clean_unnamed_columns(_safe_read_excel(file_path, q_sheet))
        q["tribe"] = tribe
        result["questionnaire"] = q

    # Fingerprint/Hair data
    f_sheet = _find_sheet(sheets, "Fingerprint and Hair")
    if f_sheet:
        f = _clean_unnamed_columns(_safe_read_excel(file_path, f_sheet))
        f["tribe"] = tribe
        result["fingerprint_hair"] = f

    return result


@st.cache_data(show_spinner=False)
def load_all_data(folder: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    root = Path(folder)
    files = sorted(root.glob("*.xlsx"))
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for file in files:
        tribe = _tribe_name_from_file(file)
        all_data[tribe] = load_tribe_workbook(str(file))
    return all_data


def _combine(data: Dict[str, Dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    pieces = [v[key] for v in data.values() if key in v]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _categorical_rate(df: pd.DataFrame, col: str, top_n: int = 8) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["tribe", "category", "count", "pct"])
    working = df[["tribe", col]].copy()
    working[col] = working[col].map(_norm)
    working = working[working[col] != ""]
    if working.empty:
        return pd.DataFrame(columns=["tribe", "category", "count", "pct"])

    counts = (
        working.groupby(["tribe", col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={col: "category"})
    )
    totals = counts.groupby("tribe")["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals) * 100
    counts = counts.sort_values(["tribe", "count"], ascending=[True, False])
    return counts.groupby("tribe").head(top_n)


def _non_never_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["tribe", "rate_non_never"])
    temp = df[["tribe", col]].copy()
    temp[col] = temp[col].map(_norm)
    temp = temp[temp[col] != ""]
    if temp.empty:
        return pd.DataFrame(columns=["tribe", "rate_non_never"])

    temp["is_non_never"] = temp[col] != "NEVER"
    out = temp.groupby("tribe", as_index=False)["is_non_never"].mean()
    out["rate_non_never"] = out["is_non_never"] * 100
    return out[["tribe", "rate_non_never"]]


def generate_insights(roster: pd.DataFrame, questionnaire: pd.DataFrame, measurements: pd.DataFrame) -> Tuple[List[str], List[str]]:
    commonalities: List[str] = []
    differences: List[str] = []

    if not roster.empty:
        sex = roster.copy()
        sex["sex"] = sex["sex"].map(_norm)
        pop = sex.groupby("tribe", as_index=False).size().rename(columns={"size": "population"})

        if len(pop) >= 2:
            commonalities.append("Both tribes have records for both sexes and age information for most members.")
            max_pop = pop.loc[pop["population"].idxmax()]
            min_pop = pop.loc[pop["population"].idxmin()]
            differences.append(
                f"Population coverage differs: {max_pop['tribe']} has {int(max_pop['population'])} roster entries, while {min_pop['tribe']} has {int(min_pop['population'])}."
            )

    if not questionnaire.empty:
        edu = _categorical_rate(questionnaire, "Education", top_n=3)
        if not edu.empty:
            top_edu = edu.sort_values(["tribe", "count"], ascending=[True, False]).groupby("tribe").head(1)
            if len(top_edu) >= 2:
                if (top_edu["category"] == "CLASS 1-5 (PRIMARY)").all() or (top_edu["category"] == "NAN").all():
                    commonalities.append("Education responses in both tribes are concentrated in lower schooling buckets and missing values.")

        tobacco = _non_never_rate(questionnaire, "Tobacco Smoke Use")
        alcohol = _non_never_rate(questionnaire, "Alcohol USe Frequency")

        if len(tobacco) >= 2:
            tob_sorted = tobacco.sort_values("rate_non_never", ascending=False)
            hi = tob_sorted.iloc[0]
            lo = tob_sorted.iloc[-1]
            differences.append(
                f"Tobacco-use prevalence differs: non-'never' responses are {hi['rate_non_never']:.1f}% in {hi['tribe']} versus {lo['rate_non_never']:.1f}% in {lo['tribe']}."
            )

        if len(alcohol) >= 2:
            alc_sorted = alcohol.sort_values("rate_non_never", ascending=False)
            hi = alc_sorted.iloc[0]
            lo = alc_sorted.iloc[-1]
            differences.append(
                f"Alcohol-use prevalence differs: non-'never' responses are {hi['rate_non_never']:.1f}% in {hi['tribe']} versus {lo['rate_non_never']:.1f}% in {lo['tribe']}."
            )
            commonalities.append("Most respondents in both tribes report 'never' for alcohol use.")

    if not measurements.empty and "BMI" in measurements.columns:
        bmi = measurements[["tribe", "BMI"]].copy()
        bmi = bmi.dropna()
        if not bmi.empty:
            med = bmi.groupby("tribe", as_index=False)["BMI"].median().rename(columns={"BMI": "median_bmi"})
            if len(med) >= 2:
                hi = med.sort_values("median_bmi", ascending=False).iloc[0]
                lo = med.sort_values("median_bmi", ascending=True).iloc[0]
                differences.append(
                    f"Median BMI differs: {hi['tribe']} is {hi['median_bmi']:.2f} vs {lo['tribe']} at {lo['median_bmi']:.2f}."
                )

    # De-duplicate while preserving order
    commonalities = list(dict.fromkeys(commonalities))
    differences = list(dict.fromkeys(differences))
    return commonalities, differences


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled_n = len(a) + len(b) - 2
    if pooled_n <= 0:
        return np.nan
    pooled_sd = np.sqrt((((len(a) - 1) * (s1**2)) + ((len(b) - 1) * (s2**2))) / pooled_n)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return (b.mean() - a.mean()) / pooled_sd


def _fingerprint_code_profile(fp_hair: pd.DataFrame) -> pd.DataFrame:
    if fp_hair.empty:
        return pd.DataFrame(columns=["tribe", "code", "count", "pct"])
    finger_cols = [
        "R Thumb",
        "R Index",
        "R Middle ",
        "R Ring",
        "R Little",
        "L Thumb",
        "L Index",
        "L Middle",
        "L Ring",
        "L Little",
    ]
    available = [c for c in finger_cols if c in fp_hair.columns]
    if not available:
        return pd.DataFrame(columns=["tribe", "code", "count", "pct"])

    long = fp_hair[["tribe", *available]].melt(id_vars=["tribe"], value_name="code")
    long["code"] = long["code"].map(_norm)
    long = long[long["code"] != ""]
    if long.empty:
        return pd.DataFrame(columns=["tribe", "code", "count", "pct"])

    out = long.groupby(["tribe", "code"], as_index=False).size().rename(columns={"size": "count"})
    out["pct"] = out["count"] / out.groupby("tribe")["count"].transform("sum") * 100
    return out.sort_values(["tribe", "count"], ascending=[True, False])


def _fingerprint_symmetry(fp_hair: pd.DataFrame) -> pd.DataFrame:
    if fp_hair.empty:
        return pd.DataFrame(columns=["tribe", "finger", "match_pct", "n_pairs"])
    pairs = [
        ("R Thumb", "L Thumb", "Thumb"),
        ("R Index", "L Index", "Index"),
        ("R Middle ", "L Middle", "Middle"),
        ("R Ring", "L Ring", "Ring"),
        ("R Little", "L Little", "Little"),
    ]
    rows = []
    for tribe, sub in fp_hair.groupby("tribe"):
        for right_col, left_col, label in pairs:
            if right_col not in sub.columns or left_col not in sub.columns:
                continue
            pair_df = sub[[right_col, left_col]].copy()
            pair_df[right_col] = pair_df[right_col].map(_norm)
            pair_df[left_col] = pair_df[left_col].map(_norm)
            pair_df = pair_df[(pair_df[right_col] != "") & (pair_df[left_col] != "")]
            if pair_df.empty:
                continue
            rows.append(
                {
                    "tribe": tribe,
                    "finger": label,
                    "match_pct": (pair_df[right_col] == pair_df[left_col]).mean() * 100,
                    "n_pairs": len(pair_df),
                }
            )
    return pd.DataFrame(rows)


def _relationship_numeric_columns(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    numeric_candidates = [
        c
        for c in df.columns
        if c not in {"ID Assigned", "tribe"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    preferred = [
        "Height (cm)",
        "Weight (kg)",
        "BMI",
        "Waist Cir (cm)",
        "Hip Cir (cm)",
        "Chest Cir (cm)",
        "Head Cir. (cm)",
    ]
    ordered = [c for c in preferred if c in numeric_candidates]
    ordered.extend([c for c in numeric_candidates if c not in ordered])
    return ordered


def _tribe_correlation_matrix(df: pd.DataFrame, tribe_name: str, cols: List[str]) -> pd.DataFrame:
    if df.empty or not cols:
        return pd.DataFrame()
    tribe_subset = df.loc[df["tribe"] == tribe_name, cols].copy()
    if tribe_subset.empty:
        return pd.DataFrame()
    corr = tribe_subset.corr(numeric_only=True)
    corr = corr.fillna(0)
    return corr


def main() -> None:
    st.title("Tribe Data Analyzer")
    st.caption("Cross-tribe comparison for roster, demographics, anthropometry, and lifestyle questionnaire data.")

    folder = Path(__file__).resolve().parent
    all_data = load_all_data(str(folder))

    if not all_data:
        st.error("No .xlsx files found in the app folder.")
        st.stop()

    roster = _combine(all_data, "roster")
    males = _combine(all_data, "males")
    females = _combine(all_data, "females")
    measurements = _combine(all_data, "measurements")
    questionnaire = _combine(all_data, "questionnaire")
    fp_hair = _combine(all_data, "fingerprint_hair")

    st.sidebar.header("Loaded Tribe Files")
    for tribe in sorted(all_data):
        st.sidebar.write(f"- {tribe}")

    if roster.empty:
        st.warning("Roster data could not be loaded. Some charts will be unavailable.")

    commonalities, differences = generate_insights(roster, questionnaire, measurements)

    top1, top2 = st.columns(2)
    with top1:
        st.subheader("Commonalities")
        if commonalities:
            for item in commonalities:
                st.write(f"- {item}")
        else:
            st.write("No strong common patterns detected with current non-missing fields.")

    with top2:
        st.subheader("Differences")
        if differences:
            for item in differences:
                st.write(f"- {item}")
        else:
            st.write("No strong differences detected with current non-missing fields.")

    st.divider()

    tab_overview, tab_demo, tab_measure, tab_q, tab_deep, tab_methods, tab_quality = st.tabs(
        ["Overview", "Demographics", "Measurements", "Questionnaire", "Deep Patterns", "Methods & Relationships", "Data Quality"]
    )

    with tab_overview:
        if not roster.empty:
            pop = roster.groupby("tribe", as_index=False).size().rename(columns={"size": "Population"})
            sex = roster.copy()
            sex["sex"] = sex["sex"].map(_norm).replace({"M": "MALE", "F": "FEMALE"})
            sex = sex[sex["sex"].isin(["MALE", "FEMALE"])]
            sex_ct = sex.groupby(["tribe", "sex"], as_index=False).size().rename(columns={"size": "Count"})

            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(pop, use_container_width=True)
            with c2:
                fig = px.bar(sex_ct, x="tribe", y="Count", color="sex", barmode="group", title="Sex Distribution")
                st.plotly_chart(fig, use_container_width=True)

        c3, c4, c5 = st.columns(3)
        c3.metric("Roster Rows", f"{len(roster):,}")
        c4.metric("Measurement Rows", f"{len(measurements):,}")
        c5.metric("Questionnaire Rows", f"{len(questionnaire):,}")

    with tab_demo:
        if roster.empty or roster["age"].dropna().empty:
            st.info("Age data is unavailable.")
        else:
            bins = list(range(0, 91, 10)) + [120]
            demo = roster[["tribe", "age"]].dropna().copy()
            demo["Age Group"] = pd.cut(demo["age"], bins=bins, right=False)
            age_group = (
                demo.groupby(["tribe", "Age Group"], as_index=False, observed=False)
                .size()
                .rename(columns={"size": "Count"})
            )
            age_group["Age Group"] = age_group["Age Group"].astype(str)

            fig_hist = px.histogram(demo, x="age", color="tribe", marginal="box", nbins=20, barmode="overlay", opacity=0.6)
            fig_hist.update_layout(title="Age Distribution by Tribe")
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_age_group = px.bar(age_group, x="Age Group", y="Count", color="tribe", barmode="group", title="Age Group Comparison")
            st.plotly_chart(fig_age_group, use_container_width=True)

    with tab_measure:
        if measurements.empty:
            st.info("Measurements sheet not available.")
        else:
            numeric_cols = [
                c
                for c in measurements.columns
                if c not in {"ID Assigned", "tribe"} and pd.api.types.is_numeric_dtype(measurements[c])
            ]
            if not numeric_cols:
                st.info("No numeric measurement columns found.")
            else:
                default_metric = "BMI" if "BMI" in numeric_cols else numeric_cols[0]
                selected_metric = st.selectbox("Select a body measurement", numeric_cols, index=numeric_cols.index(default_metric))

                metric_df = measurements[["tribe", selected_metric]].dropna().copy()
                if metric_df.empty:
                    st.info("No non-missing values for selected metric.")
                else:
                    fig_box = px.box(metric_df, x="tribe", y=selected_metric, points="all", title=f"{selected_metric} by Tribe")
                    st.plotly_chart(fig_box, use_container_width=True)

                    summary = (
                        metric_df.groupby("tribe")[selected_metric]
                        .agg(["count", "mean", "median", "std", "min", "max"])
                        .reset_index()
                    )
                    st.dataframe(summary, use_container_width=True)

    with tab_q:
        if questionnaire.empty:
            st.info("Questionnaire sheet not available.")
        else:
            candidate_cols = [
                "Gender",
                "Marital Status",
                "Education",
                "Main Work",
                "Alcohol USe Frequency",
                "Tobacco Smoke Use",
                "Gutkha Use",
                "Home type",
            ]
            available = [c for c in candidate_cols if c in questionnaire.columns]
            if not available:
                st.info("No comparable questionnaire fields available.")
            else:
                selected_col = st.selectbox("Select a lifestyle/socioeconomic field", available, index=available.index("Education") if "Education" in available else 0)
                dist = _categorical_rate(questionnaire, selected_col, top_n=10)
                if dist.empty:
                    st.info("No non-missing values for selected field.")
                else:
                    fig = px.bar(
                        dist,
                        x="category",
                        y="pct",
                        color="tribe",
                        barmode="group",
                        title=f"{selected_col}: Relative Distribution (%)",
                    )
                    fig.update_layout(xaxis_title="Category", yaxis_title="Percent within tribe")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dist.sort_values(["tribe", "count"], ascending=[True, False]), use_container_width=True)

    with tab_deep:
        st.subheader("Deeper Cross-Tribe Patterns")

        if not roster.empty and not roster["age"].dropna().empty:
            demo = roster[["tribe", "sex", "age"]].dropna(subset=["age"]).copy()
            demo["sex"] = demo["sex"].map(_norm).replace({"M": "MALE", "F": "FEMALE"})
            demo = demo[demo["sex"].isin(["MALE", "FEMALE"])]
            demo["age_band"] = pd.cut(demo["age"], bins=[0, 10, 20, 30, 40, 50, 60, 120], right=False)
            age_sex = (
                demo.groupby(["tribe", "sex", "age_band"], as_index=False, observed=False)
                .size()
                .rename(columns={"size": "count"})
            )
            age_sex["age_band"] = age_sex["age_band"].astype(str)
            fig_age_sex = px.bar(
                age_sex,
                x="age_band",
                y="count",
                color="sex",
                barmode="group",
                facet_col="tribe",
                title="Age-Sex Structure by Tribe",
            )
            st.plotly_chart(fig_age_sex, use_container_width=True)

        tribes = sorted(measurements["tribe"].dropna().unique().tolist()) if not measurements.empty else []
        if len(tribes) == 2 and not measurements.empty:
            t1, t2 = tribes
            candidate = [
                "Height (cm)",
                "Weight (kg)",
                "BMI",
                "Waist Cir (cm)",
                "Hip Cir (cm)",
                "Chest Cir (cm)",
                "Head Cir. (cm)",
            ]
            rows = []
            for col in candidate:
                if col not in measurements.columns:
                    continue
                a = measurements.loc[measurements["tribe"] == t1, col]
                b = measurements.loc[measurements["tribe"] == t2, col]
                if a.dropna().shape[0] < 10 or b.dropna().shape[0] < 10:
                    continue
                d = _cohens_d(a, b)
                rows.append(
                    {
                        "metric": col,
                        f"mean_{t1}": round(a.mean(), 2),
                        f"mean_{t2}": round(b.mean(), 2),
                        "effect_size_d": round(d, 2) if not np.isnan(d) else np.nan,
                        "magnitude": "small" if abs(d) < 0.3 else "moderate" if abs(d) < 0.8 else "large",
                    }
                )
            effect_df = pd.DataFrame(rows)
            if not effect_df.empty:
                st.markdown("**Anthropometric Effect Sizes (Cohen d)**")
                st.dataframe(effect_df.sort_values("effect_size_d", ascending=False), use_container_width=True)

                fig_d = px.bar(
                    effect_df,
                    x="metric",
                    y="effect_size_d",
                    color="magnitude",
                    title=f"Standardized Difference ({t2} - {t1})",
                )
                fig_d.update_layout(yaxis_title="Cohen d")
                st.plotly_chart(fig_d, use_container_width=True)

        fp_profile = _fingerprint_code_profile(fp_hair)
        if not fp_profile.empty:
            top_fp = fp_profile.groupby("tribe", as_index=False).head(6)
            st.markdown("**Fingerprint Pattern Composition (Top 6 Codes per Tribe)**")
            fig_fp = px.bar(
                top_fp,
                x="code",
                y="pct",
                color="tribe",
                barmode="group",
                title="Fingerprint Code Mix by Tribe",
            )
            fig_fp.update_layout(yaxis_title="Percent within tribe")
            st.plotly_chart(fig_fp, use_container_width=True)

            entropy_rows = []
            for tribe, sub in fp_profile.groupby("tribe"):
                p = (sub["count"] / sub["count"].sum()).values
                entropy = float(-(p * np.log2(p)).sum()) if len(p) else np.nan
                entropy_rows.append(
                    {
                        "tribe": tribe,
                        "unique_codes": int(sub["code"].nunique()),
                        "shannon_entropy": round(entropy, 3),
                    }
                )
            st.dataframe(pd.DataFrame(entropy_rows), use_container_width=True)

        fp_sym = _fingerprint_symmetry(fp_hair)
        if not fp_sym.empty:
            st.markdown("**Left-Right Finger Symmetry**")
            fig_sym = px.bar(fp_sym, x="finger", y="match_pct", color="tribe", barmode="group", title="Same-Finger Left/Right Match Rate")
            fig_sym.update_layout(yaxis_title="Match %")
            st.plotly_chart(fig_sym, use_container_width=True)

        if not questionnaire.empty:
            behavior_cols = ["Tobacco Smoke Use", "Alcohol USe Frequency"]
            behavior_rows = []
            for col in behavior_cols:
                if col not in questionnaire.columns:
                    continue
                temp = questionnaire[["tribe", col]].copy()
                temp[col] = temp[col].map(_norm)
                temp = temp[temp[col] != ""]
                if temp.empty:
                    continue
                temp["is_user"] = temp[col] != "NEVER"
                rates = temp.groupby("tribe", as_index=False)["is_user"].mean()
                for _, row in rates.iterrows():
                    behavior_rows.append(
                        {
                            "tribe": row["tribe"],
                            "behavior": col,
                            "non_never_pct": row["is_user"] * 100,
                        }
                    )
            if behavior_rows:
                behavior_df = pd.DataFrame(behavior_rows)
                fig_b = px.bar(
                    behavior_df,
                    x="behavior",
                    y="non_never_pct",
                    color="tribe",
                    barmode="group",
                    title="Substance Exposure Signal (Non-'Never' Rate)",
                )
                fig_b.update_layout(yaxis_title="Percent of respondents")
                st.plotly_chart(fig_b, use_container_width=True)

    with tab_methods:
        st.subheader("Analytical methods used")
        st.write("This dashboard uses descriptive and comparative statistical techniques rather than a supervised predictive model.")
        methods = [
            "- Data harmonisation and cleaning to align roster, measurements, questionnaire, and fingerprint sheets across tribes.",
            "- Descriptive summaries such as counts, means, medians, ranges, age-sex structure, and completeness checks.",
            "- Comparative statistics including effect sizes (Cohen's d) for anthropometric differences between tribes.",
            "- Frequency and percentage comparisons for categorical questionnaire fields and fingerprint-code composition.",
            "- Correlation-based feature relationship analysis to test whether anthropometric variables behave similarly or differently within and between tribes.",
        ]
        for item in methods:
            st.write(item)

        numeric_cols = _relationship_numeric_columns(measurements)
        tribes = sorted(measurements["tribe"].dropna().unique().tolist()) if not measurements.empty else []

        if numeric_cols and tribes:
            st.divider()
            st.subheader("Intra-tribe feature relationships")
            selected_intra_tribe = st.selectbox("Select a tribe", tribes, key="intra_tribe")
            corr = _tribe_correlation_matrix(measurements, selected_intra_tribe, numeric_cols)
            if corr.empty or corr.shape[0] < 2:
                st.info("Insufficient numeric data for correlation analysis in the selected tribe.")
            else:
                fig_intra = px.imshow(
                    corr,
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"Pearson correlation matrix for {selected_intra_tribe}",
                )
                fig_intra.update_xaxes(side="top")
                st.plotly_chart(fig_intra, use_container_width=True)
                st.dataframe(corr.round(3), use_container_width=True)

            if len(tribes) >= 2:
                st.divider()
                st.subheader("Inter-tribe feature relationship differences")
                tribe_a = st.selectbox("Reference tribe", tribes, key="rel_tribe_a")
                comparison_options = [t for t in tribes if t != tribe_a]
                tribe_b = st.selectbox("Comparison tribe", comparison_options, key="rel_tribe_b")
                corr_a = _tribe_correlation_matrix(measurements, tribe_a, numeric_cols)
                corr_b = _tribe_correlation_matrix(measurements, tribe_b, numeric_cols)
                if corr_a.empty or corr_b.empty:
                    st.info("Correlation matrices could not be built for both selected tribes.")
                else:
                    shared_cols = [c for c in corr_a.columns if c in corr_b.columns]
                    if len(shared_cols) < 2:
                        st.info("Not enough shared numeric features for a tribe-to-tribe comparison.")
                    else:
                        corr_a = corr_a.loc[shared_cols, shared_cols]
                        corr_b = corr_b.loc[shared_cols, shared_cols]
                        diff = (corr_a - corr_b).abs()
                        fig_inter = px.imshow(
                            diff,
                            labels=dict(x="Feature", y="Feature", color="Absolute difference"),
                            color_continuous_scale="Blues",
                            title=f"Absolute correlation difference between {tribe_a} and {tribe_b}",
                        )
                        fig_inter.update_xaxes(side="top")
                        st.plotly_chart(fig_inter, use_container_width=True)
                        diff_long = diff.where(np.triu(np.ones(diff.shape), k=1).astype(bool)).stack().reset_index()
                        diff_long.columns = ["feature_a", "feature_b", "abs_diff"]
                        diff_long = diff_long.sort_values("abs_diff", ascending=False).head(10)
                        st.caption("Large values mean the feature association is much stronger or weaker in one tribe than the other; smaller values suggest broadly similar relationships.")
                        st.dataframe(diff_long, use_container_width=True)
        else:
            st.info("Anthropometric measurement columns are not available for relationship analysis.")

    with tab_quality:
        quality_rows = []

        def add_quality_row(sheet_key: str, sheet_df: pd.DataFrame) -> None:
            if sheet_df.empty:
                return
            for tribe, sub in sheet_df.groupby("tribe"):
                completeness = (1 - (sub.isna().sum().sum() / (sub.shape[0] * max(sub.shape[1], 1)))) * 100
                quality_rows.append(
                    {
                        "tribe": tribe,
                        "sheet": sheet_key,
                        "rows": int(sub.shape[0]),
                        "columns": int(sub.shape[1]),
                        "completeness_pct": round(completeness, 2),
                    }
                )

        add_quality_row("roster", roster)
        add_quality_row("males", males)
        add_quality_row("females", females)
        add_quality_row("measurements", measurements)
        add_quality_row("questionnaire", questionnaire)
        add_quality_row("fingerprint_hair", fp_hair)

        quality_df = pd.DataFrame(quality_rows)
        if quality_df.empty:
            st.info("No quality information available.")
        else:
            st.dataframe(quality_df.sort_values(["sheet", "tribe"]), use_container_width=True)
            fig_q = px.bar(
                quality_df,
                x="sheet",
                y="completeness_pct",
                color="tribe",
                barmode="group",
                title="Approximate Completeness by Sheet",
            )
            st.plotly_chart(fig_q, use_container_width=True)


if __name__ == "__main__":
    main()
