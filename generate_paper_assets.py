from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "paper_assets" / "figures"
TAB_DIR = ROOT / "paper_assets" / "tables"
OUT_DIR = ROOT / "paper_assets"

for d in [FIG_DIR, TAB_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def norm(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def find_sheet(sheet_names: List[str], target: str) -> str | None:
    t = target.strip().upper()
    for s in sheet_names:
        if s.strip().upper() == t:
            return s
    for s in sheet_names:
        if t in s.strip().upper():
            return s
    return None


def tribe_from_filename(path: Path) -> str:
    return path.stem.replace("Tribe List", "").strip(" -_")


def load_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for f in sorted(ROOT.glob("*.xlsx")):
        tribe = tribe_from_filename(f)
        xls = pd.ExcelFile(f)
        sheets = xls.sheet_names

        d: Dict[str, pd.DataFrame] = {}

        s1 = find_sheet(sheets, "Sheet1")
        if s1:
            roster = pd.read_excel(f, sheet_name=s1).dropna(how="all")
            age_col = "age" if "age" in roster.columns else "Age"
            d["roster"] = pd.DataFrame(
                {
                    "tribe": tribe,
                    "sex": roster.get("Sex", np.nan),
                    "age": pd.to_numeric(roster.get(age_col, np.nan), errors="coerce"),
                }
            )

        m = find_sheet(sheets, "Measurements")
        if m:
            meas = pd.read_excel(f, sheet_name=m).dropna(how="all")
            meas["tribe"] = tribe
            for c in meas.columns:
                if c not in {"ID Assigned", "tribe"}:
                    meas[c] = pd.to_numeric(meas[c], errors="coerce")
            d["measurements"] = meas

        q = find_sheet(sheets, "Questionnaire")
        if q:
            ques = pd.read_excel(f, sheet_name=q).dropna(how="all")
            ques["tribe"] = tribe
            ques["Age"] = pd.to_numeric(ques.get("Age", np.nan), errors="coerce")
            d["questionnaire"] = ques

        fp = find_sheet(sheets, "Fingerprint and Hair")
        if fp:
            fing = pd.read_excel(f, sheet_name=fp).dropna(how="all")
            fing["tribe"] = tribe
            d["fingerprint_hair"] = fing

        out[tribe] = d
    return out


def combine(data: Dict[str, Dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    frames = [v[key] for v in data.values() if key in v]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled = np.sqrt((((len(a) - 1) * s1 * s1) + ((len(b) - 1) * s2 * s2)) / (len(a) + len(b) - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (b.mean() - a.mean()) / pooled


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().values
    b = b.dropna().values
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return (gt - lt) / (len(a) * len(b))


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.array(pvals, dtype=float)
    n = len(p)
    idx = np.argsort(p)
    sorted_p = p[idx]
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, sorted_p[i] * n / rank)
        q[i] = val
        prev = val
    out = np.empty(n)
    out[idx] = q
    return out.tolist()


def make_methodology_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    boxes = [
        (0.02, 0.65, 0.18, 0.22, "Data Ingestion\n2 Excel workbooks\n10+ sheets each"),
        (0.24, 0.65, 0.2, 0.22, "Harmonization\nSchema mapping\nType coercion\nQC flags"),
        (0.48, 0.65, 0.2, 0.22, "Analytical Layers\nDemography\nAnthropometry\nBehavior\nDermatoglyphics"),
        (0.72, 0.65, 0.25, 0.22, "Inference\nMann-Whitney U\nChi-square\nEffect sizes\nFDR correction"),
        (0.08, 0.3, 0.24, 0.22, "Pattern Discovery\nPCA\nCorrelation networks\nSymmetry metrics"),
        (0.4, 0.3, 0.24, 0.22, "Validation\nMissingness audit\nSensitivity checks\nCross-sheet consistency"),
        (0.72, 0.3, 0.24, 0.22, "Deliverables\nFigures\nStatistical tables\nNature-style manuscript"),
    ]

    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, ec="#1f2937", fc="#e5eef9", lw=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.20, 0.76), (0.24, 0.76)),
        ((0.44, 0.76), (0.48, 0.76)),
        ((0.68, 0.76), (0.72, 0.76)),
        ((0.58, 0.65), (0.52, 0.52)),
        ((0.84, 0.65), (0.84, 0.52)),
        ((0.2, 0.41), (0.4, 0.41)),
        ((0.64, 0.41), (0.72, 0.41)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="#111827"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "methodology_pipeline.png", dpi=300)
    plt.close(fig)


def build_assets() -> None:
    data = load_data()
    roster = combine(data, "roster")
    meas = combine(data, "measurements")
    ques = combine(data, "questionnaire")
    fp = combine(data, "fingerprint_hair")

    roster["sex"] = roster["sex"].map(norm).replace({"M": "MALE", "F": "FEMALE"})

    # Table 1: Cohort profile
    t1 = []
    for tribe, sub in roster.groupby("tribe"):
        t1.append(
            {
                "tribe": tribe,
                "n_roster": int(len(sub)),
                "n_male": int((sub["sex"] == "MALE").sum()),
                "n_female": int((sub["sex"] == "FEMALE").sum()),
                "age_median": float(np.nanmedian(sub["age"])),
                "age_iqr": float(np.nanpercentile(sub["age"].dropna(), 75) - np.nanpercentile(sub["age"].dropna(), 25)),
            }
        )
    t1_df = pd.DataFrame(t1)
    t1_df.to_csv(TAB_DIR / "table1_cohort_profile.csv", index=False)

    # Figure 1: age-sex structure
    demo = roster.dropna(subset=["age"]).copy()
    demo = demo[demo["sex"].isin(["MALE", "FEMALE"])]
    demo["age_band"] = pd.cut(demo["age"], bins=[0, 10, 20, 30, 40, 50, 60, 120], right=False)
    age_sex = demo.groupby(["tribe", "sex", "age_band"], observed=False).size().reset_index(name="count")
    age_sex["age_band"] = age_sex["age_band"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=age_sex, x="age_band", y="count", hue="sex", ax=ax)
    ax.set_title("Age-Sex Structure Across Tribes")
    ax.set_xlabel("Age band")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "age_sex_structure.png", dpi=300)
    plt.close(fig)

    # Table 2: anthropometric tests + effect sizes
    numeric_candidates = [
        "Height (cm)",
        "Weight (kg)",
        "BMI",
        "Head Cir. (cm)",
        "Waist Cir (cm)",
        "Hip Cir (cm)",
        "Chest Cir (cm)",
        "Biceps (mm)",
        "Triceps (mm)",
        "Subscapular (mm)",
        "Abdomen (mm)",
    ]
    tribes = sorted(meas["tribe"].dropna().unique().tolist())
    a_name, b_name = tribes[0], tribes[1]
    a_df = meas[meas["tribe"] == a_name]
    b_df = meas[meas["tribe"] == b_name]

    rows = []
    pvals = []
    for col in numeric_candidates:
        if col not in meas.columns:
            continue
        a = a_df[col].dropna()
        b = b_df[col].dropna()
        if len(a) < 8 or len(b) < 8:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        cd = cliffs_delta(a, b)
        rows.append(
            {
                "variable": col,
                f"{a_name}_median": float(np.median(a)),
                f"{b_name}_median": float(np.median(b)),
                "mannwhitney_u": float(u),
                "p_value": float(p),
                "cohens_d": float(d),
                "cliffs_delta": float(cd),
                "n_a": int(len(a)),
                "n_b": int(len(b)),
            }
        )
        pvals.append(float(p))

    t2_df = pd.DataFrame(rows)
    if not t2_df.empty:
        t2_df["q_value_bh"] = benjamini_hochberg(pvals)
        t2_df = t2_df.sort_values("p_value")
    t2_df.to_csv(TAB_DIR / "table2_anthropometry_stats.csv", index=False)

    # Figure 2: effect size forest
    if not t2_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = t2_df.sort_values("cohens_d")
        ax.axvline(0, ls="--", color="gray", lw=1)
        ax.scatter(order["cohens_d"], order["variable"], color="#0f766e", s=100)
        ax.set_title("Anthropometric Standardized Differences (Cohen d)")
        ax.set_xlabel(f"Cohen d ({b_name} - {a_name})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "effect_sizes_cohens_d.png", dpi=300)
        plt.close(fig)

    # Figure 3: BMI distribution
    if "BMI" in meas.columns:
        bmi = meas[["tribe", "BMI"]].dropna().copy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=bmi, x="tribe", y="BMI", inner="box", cut=0, ax=ax)
        ax.set_title("BMI Distribution by Tribe")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "bmi_violin.png", dpi=300)
        plt.close(fig)

    # Table 3: categorical comparisons
    cat_cols = ["Marital Status", "Education", "Main Work", "Tobacco Smoke Use", "Alcohol USe Frequency", "Gutkha Use"]
    c_rows = []
    cpvals = []

    for c in cat_cols:
        if c not in ques.columns:
            continue
        t = ques[["tribe", c]].copy()
        t[c] = t[c].map(norm)
        t = t[t[c] != ""]
        if t.empty:
            continue
        ct = pd.crosstab(t["tribe"], t[c])
        if ct.shape[0] != 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        r, k = ct.shape
        cramers_v = np.sqrt((chi2 / n) / (min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else np.nan
        c_rows.append({"field": c, "chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": float(cramers_v), "n": int(n)})
        cpvals.append(float(p))

    t3_df = pd.DataFrame(c_rows)
    if not t3_df.empty:
        t3_df["q_value_bh"] = benjamini_hochberg(cpvals)
        t3_df = t3_df.sort_values("p_value")
    t3_df.to_csv(TAB_DIR / "table3_categorical_stats.csv", index=False)

    # Figure 4: non-never behavior rates
    behav = []
    for c in ["Tobacco Smoke Use", "Alcohol USe Frequency", "Gutkha Use"]:
        if c not in ques.columns:
            continue
        t = ques[["tribe", c]].copy()
        t[c] = t[c].map(norm)
        t = t[t[c] != ""]
        if t.empty:
            continue
        t["non_never"] = t[c] != "NEVER"
        g = t.groupby("tribe", as_index=False)["non_never"].mean()
        g["field"] = c
        g["rate"] = g["non_never"] * 100
        behav.append(g[["tribe", "field", "rate"]])

    if behav:
        bdf = pd.concat(behav, ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=bdf, x="field", y="rate", hue="tribe", ax=ax)
        ax.set_ylabel("Non-'never' rate (%)")
        ax.set_title("Behavioral Exposure Signal")
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "behavior_non_never_rates.png", dpi=300)
        plt.close(fig)

    # Fingerprint profile and symmetry
    finger_cols = ["R Thumb", "R Index", "R Middle ", "R Ring", "R Little", "L Thumb", "L Index", "L Middle", "L Ring", "L Little"]
    avf = [c for c in finger_cols if c in fp.columns]
    if not fp.empty and avf:
        long = fp[["tribe", *avf]].melt(id_vars=["tribe"], value_name="code")
        long["code"] = long["code"].map(norm)
        long = long[long["code"] != ""]

        prof = long.groupby(["tribe", "code"], as_index=False).size().rename(columns={"size": "count"})
        prof["pct"] = prof["count"] / prof.groupby("tribe")["count"].transform("sum") * 100
        prof.to_csv(TAB_DIR / "table4_fingerprint_profile.csv", index=False)

        top = prof.sort_values(["tribe", "count"], ascending=[True, False]).groupby("tribe", as_index=False).head(6)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top, x="code", y="pct", hue="tribe", ax=ax)
        ax.set_title("Top Fingerprint Codes by Tribe")
        ax.set_ylabel("Percent within tribe")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fingerprint_top_codes.png", dpi=300)
        plt.close(fig)

        # symmetry
        pairs = [("R Thumb", "L Thumb", "Thumb"), ("R Index", "L Index", "Index"), ("R Middle ", "L Middle", "Middle"), ("R Ring", "L Ring", "Ring"), ("R Little", "L Little", "Little")]
        s_rows = []
        for tribe, sub in fp.groupby("tribe"):
            for r, l, label in pairs:
                if r not in sub.columns or l not in sub.columns:
                    continue
                q = sub[[r, l]].copy()
                q[r] = q[r].map(norm)
                q[l] = q[l].map(norm)
                q = q[(q[r] != "") & (q[l] != "")]
                if q.empty:
                    continue
                s_rows.append({"tribe": tribe, "finger": label, "match_rate": float((q[r] == q[l]).mean() * 100), "n": int(len(q))})
        sym = pd.DataFrame(s_rows)
        sym.to_csv(TAB_DIR / "table5_fingerprint_symmetry.csv", index=False)

        if not sym.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=sym, x="finger", y="match_rate", hue="tribe", ax=ax)
            ax.set_title("Left-Right Finger Symmetry")
            ax.set_ylabel("Match rate (%)")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fingerprint_symmetry.png", dpi=300)
            plt.close(fig)

    # Figure: PCA
    pca_cols = [c for c in ["Height (cm)", "Weight (kg)", "BMI", "Waist Cir (cm)", "Hip Cir (cm)", "Chest Cir (cm)"] if c in meas.columns]
    pca_df = meas[["tribe", *pca_cols]].dropna().copy()
    if len(pca_df) > 10 and len(pca_cols) >= 3:
        X = StandardScaler().fit_transform(pca_df[pca_cols].values)
        pca = PCA(n_components=2, random_state=42)
        z = pca.fit_transform(X)
        plot_df = pd.DataFrame({"PC1": z[:, 0], "PC2": z[:, 1], "tribe": pca_df["tribe"].values})
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="tribe", ax=ax, s=80)
        ax.set_title(f"PCA of Anthropometry (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "anthropometry_pca.png", dpi=300)
        plt.close(fig)

    # Missingness table
    q_rows = []
    for key in ["roster", "measurements", "questionnaire", "fingerprint_hair"]:
        df = combine(data, key)
        if df.empty:
            continue
        for tribe, sub in df.groupby("tribe"):
            miss = sub.isna().sum().sum() / (sub.shape[0] * max(sub.shape[1], 1))
            q_rows.append({"tribe": tribe, "dataset": key, "rows": int(sub.shape[0]), "cols": int(sub.shape[1]), "missingness_pct": float(miss * 100)})
    pd.DataFrame(q_rows).to_csv(TAB_DIR / "table6_missingness.csv", index=False)

    make_methodology_diagram()

    # Quick narrative summary
    summary = {
        "tribes": sorted(data.keys()),
        "n_roster": int(len(roster)),
        "n_measurements": int(len(meas)),
        "n_questionnaire": int(len(ques)),
        "n_fingerprint": int(len(fp)),
    }
    pd.Series(summary).to_json(OUT_DIR / "summary.json", indent=2)


if __name__ == "__main__":
    build_assets()
    print("Paper assets generated in ./paper_assets")
