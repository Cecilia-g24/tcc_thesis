"""
liwc_de_en_results_comparison.py

Investigates differences between LIWC-22 analysis results run on the
German (DE) originals and their English (EN) translations.

Run from the repo root:
    python utils/liwc_de_en_results_comparison.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; remove if running in a Jupyter/interactive env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data", "data_clean", "02_results_liwc_dict")
DE_PATH  = os.path.join(DATA_DIR, "LIWC-22 Results - full_dataset_de - LIWC Analysis.csv")
EN_PATH  = os.path.join(DATA_DIR, "LIWC-22 Results - full_dataset_en - LIWC Analysis.csv")
OUT_DIR  = os.path.join(BASE, "figures", "liwc_de_en_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
sns.set_theme(style="whitegrid", palette="Set2")
COLOR_DE = "#4C72B0"   # blue   → German
COLOR_EN = "#DD8452"   # orange → English

SEP  = "=" * 72
SEP2 = "─" * 40

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
de = pd.read_csv(DE_PATH)
en = pd.read_csv(EN_PATH)

# Metadata columns that are NOT LIWC features
DE_META = {"id", "dimension", "rater_one", "rater_two", "rater_three",
           "average_score", "ColumnID", "Text", "Segment"}
EN_META = {"id", "dimension", "text", "rater_one", "rater_two", "rater_three",
           "average_score", "ColumnID", "Text", "Segment"}

de_liwc_cols = [c for c in de.columns if c not in DE_META]
en_liwc_cols = [c for c in en.columns if c not in EN_META]

# Case-insensitive matching maps
de_lower_map = {c.lower(): c for c in de_liwc_cols}
en_lower_map = {c.lower(): c for c in en_liwc_cols}
de_set = set(de_lower_map)
en_set = set(en_lower_map)

common_lower  = sorted(de_set & en_set)
de_only_lower = sorted(de_set - en_set)
en_only_lower = sorted(en_set - de_set)

common_de_cols = [de_lower_map[k] for k in common_lower]
common_en_cols = [en_lower_map[k] for k in common_lower]


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC STATS
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("  LIWC-22  DE vs EN  –  RESULTS COMPARISON")
print(SEP)

for label, df, meta_set, liwc_cols in [
    ("DE  (German original)", de, DE_META, de_liwc_cols),
    ("EN  (English translation)", en, EN_META, en_liwc_cols),
]:
    num_liwc = sum(pd.api.types.is_numeric_dtype(df[c]) for c in liwc_cols)
    print(f"\n{SEP2}")
    print(f"  {label}")
    print(SEP2)
    print(f"  Shape                  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Metadata columns       : {len(meta_set)}")
    print(f"  LIWC feature cols total: {len(liwc_cols)}")
    print(f"    of which numeric     : {num_liwc}")

    print(f"\n  Rows per dimension:")
    for dim, cnt in df["dimension"].value_counts().items():
        print(f"    {dim:<38} {cnt:>4}")

    print(f"\n  average_score  (human rating 1–5):")
    desc = df["average_score"].describe()
    for stat, val in desc.items():
        print(f"    {stat:<10}: {val:.3f}")

    print(f"\n  WC  (word count per utterance):")
    desc_wc = df["WC"].describe()
    for stat, val in desc_wc.items():
        print(f"    {stat:<10}: {val:.2f}")

    print(f"\n  First 6 LIWC cols : {liwc_cols[:6]}")
    print(f"  Last  6 LIWC cols : {liwc_cols[-6:]}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. COLUMN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  COLUMN COMPARISON  (case-insensitive matching)")
print(SEP)
print(f"\n  LIWC cols in DE          : {len(de_liwc_cols)}")
print(f"  LIWC cols in EN          : {len(en_liwc_cols)}")
print(f"  Common (matched)         : {len(common_lower)}")
print(f"  DE-only                  : {len(de_only_lower)}")
print(f"  EN-only                  : {len(en_only_lower)}")

print(f"\n  Common LIWC columns ({len(common_lower)}):")
print("    " + ",  ".join(common_de_cols))

print(f"\n  DE-only columns ({len(de_only_lower)}):")
print("    " + ",  ".join(de_lower_map[k] for k in de_only_lower))

print(f"\n  EN-only columns ({len(en_only_lower)}):")
print("    " + ",  ".join(en_lower_map[k] for k in en_only_lower))


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROW / ID COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  ROW / ID COMPARISON")
print(SEP)

de_ids     = set(de["id"])
en_ids     = set(en["id"])
both_ids   = de_ids & en_ids
de_only_ids = de_ids - en_ids
en_only_ids = en_ids - de_ids

print(f"\n  IDs in DE       : {len(de_ids)}")
print(f"  IDs in EN       : {len(en_ids)}")
print(f"  IDs in BOTH     : {len(both_ids)}")
print(f"  DE-only IDs     : {len(de_only_ids)}")
print(f"  EN-only IDs     : {len(en_only_ids)}")

if de_only_ids:
    print(f"  DE-only sample  : {sorted(de_only_ids)[:5]}")
if en_only_ids:
    print(f"  EN-only sample  : {sorted(en_only_ids)[:5]}")

same_dims = sorted(de["dimension"].unique()) == sorted(en["dimension"].unique())
print(f"\n  Same dimension labels? : {same_dims}")

# Dimensions present in both CSVs per ID check
print(f"\n  Dimensions in DE: {sorted(de['dimension'].unique())}")
print(f"  Dimensions in EN: {sorted(en['dimension'].unique())}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMMON FEATURE STATISTICS (matched rows only)
# ══════════════════════════════════════════════════════════════════════════════
de_m = de[de["id"].isin(both_ids)].set_index("id").sort_index()
en_m = en[en["id"].isin(both_ids)].set_index("id").sort_index()

de_feat = de_m[common_de_cols].apply(pd.to_numeric, errors="coerce")
en_feat = en_m[common_en_cols].apply(pd.to_numeric, errors="coerce")
de_feat.columns = common_lower
en_feat.columns = common_lower

stats = pd.DataFrame({
    "DE_mean"  : de_feat.mean(),
    "EN_mean"  : en_feat.mean(),
    "DE_std"   : de_feat.std(),
    "EN_std"   : en_feat.std(),
    "DE_median": de_feat.median(),
    "EN_median": en_feat.median(),
})
stats["abs_mean_diff"] = (stats["EN_mean"] - stats["DE_mean"]).abs()
stats["rel_diff_%"]    = stats["abs_mean_diff"] / (stats["DE_mean"].abs() + 1e-9) * 100
stats = stats.sort_values("abs_mean_diff", ascending=False)

print(f"\n{SEP}")
print("  COMMON FEATURE STATISTICS  (matched rows, sorted by |DE_mean – EN_mean|)")
print(SEP)
print(stats.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    print(f"[saved] {path}")


# ── Fig 1: Column overlap overview ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# bar chart
labels_bar = ["DE only", "Common", "EN only"]
vals_bar   = [len(de_only_lower), len(common_lower), len(en_only_lower)]
colors_bar = [COLOR_DE, "#4DAF4A", COLOR_EN]
bars = axes[0].bar(labels_bar, vals_bar, color=colors_bar, width=0.5, edgecolor="white")
for b, v in zip(bars, vals_bar):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v),
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[0].set_title("LIWC Feature Column Overlap", fontweight="bold")
axes[0].set_ylabel("Number of columns")
axes[0].set_ylim(0, max(vals_bar) * 1.2)

# summary table
table_data = [
    ["Total rows in DE",      str(len(de))],
    ["Total rows in EN",      str(len(en))],
    ["Matched IDs (both)",    str(len(both_ids))],
    ["",                      ""],
    ["Total LIWC cols  – DE", str(len(de_liwc_cols))],
    ["Total LIWC cols  – EN", str(len(en_liwc_cols))],
    ["Common columns",        str(len(common_lower))],
    ["DE-only columns",       str(len(de_only_lower))],
    ["EN-only columns",       str(len(en_only_lower))],
]
axes[1].axis("off")
tbl = axes[1].table(cellText=table_data, colLabels=["Metric", "Value"],
                    loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.9)
axes[1].set_title("Dataset & Column Summary", fontweight="bold")

plt.suptitle("LIWC-22 DE vs EN  –  Column & Dataset Overview",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
_save("01_column_overlap.png")
plt.show()


# ── Fig 2: DE-only / EN-only column names ────────────────────────────────────
n_de_only = len(de_only_lower)
n_en_only = len(en_only_lower)
height = max(4, max(n_de_only, n_en_only) * 0.32 + 1.5)

fig, axes = plt.subplots(1, 2, figsize=(14, height))
for ax, title, keys, lower_map, color in [
    (axes[0], f"DE-only  ({n_de_only} cols)", de_only_lower, de_lower_map, COLOR_DE),
    (axes[1], f"EN-only  ({n_en_only} cols)", en_only_lower, en_lower_map, COLOR_EN),
]:
    names = [lower_map[k] for k in keys]
    y = np.arange(len(names))
    ax.barh(y, [1] * len(names), color=color, alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks([])
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()

plt.suptitle("Unique LIWC Columns per Language Version", fontsize=12, fontweight="bold")
plt.tight_layout()
_save("02_unique_columns.png")
plt.show()


# ── Fig 3: Mean scatter plot for all common features ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
xs = stats["DE_mean"]
ys = stats["EN_mean"]
ax.scatter(xs, ys, alpha=0.65, s=55, color="#555555", zorder=3)

# y = x reference line
pad = max(xs.max(), ys.max()) * 0.05
lim_max = max(xs.max(), ys.max()) + pad
lim_min = min(min(xs.min(), 0), min(ys.min(), 0)) - pad
ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="y = x  (equal means)")

# Annotate top-12 most divergent
for feat in stats.head(12).index:
    ax.annotate(feat, (stats.loc[feat, "DE_mean"], stats.loc[feat, "EN_mean"]),
                fontsize=7.5, xytext=(5, 4), textcoords="offset points", color="#333333")

ax.set_xlabel("DE mean value", fontsize=11)
ax.set_ylabel("EN mean value", fontsize=11)
ax.set_title(
    "Common LIWC Features: DE vs EN Mean Values\n(top-12 most divergent labelled)",
    fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
_save("03_mean_scatter.png")
plt.show()


# ── Fig 4: Top-15 features by absolute mean difference ───────────────────────
top15 = stats.head(15)
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(top15))
w = 0.38
ax.bar(x - w / 2, top15["DE_mean"], width=w, label="DE mean", color=COLOR_DE, alpha=0.88)
ax.bar(x + w / 2, top15["EN_mean"], width=w, label="EN mean", color=COLOR_EN, alpha=0.88)
ax.set_xticks(x)
ax.set_xticklabels(top15.index, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Mean LIWC value (%)")
ax.set_title("Top-15 Common LIWC Features Ranked by |DE_mean – EN_mean|",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
_save("04_top15_mean_diff.png")
plt.show()


# ── Fig 5: Violin plots for top-8 most-divergent common features ──────────────
top8 = stats.head(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(top8):
    de_vals = de_feat[feat].dropna()
    en_vals = en_feat[feat].dropna()
    data = pd.DataFrame({
        "value": pd.concat([de_vals, en_vals], ignore_index=True),
        "lang" : ["DE"] * len(de_vals) + ["EN"] * len(en_vals),
    })
    sns.violinplot(data=data, x="lang", y="value",
                   palette={"DE": COLOR_DE, "EN": COLOR_EN},
                   ax=axes[i], inner="box", cut=0)
    axes[i].set_title(feat, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

plt.suptitle("Distributions of Top-8 Most Divergent Common LIWC Features  (DE vs EN)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("05_top8_violin.png")
plt.show()


# ── Fig 6: Word count distributions ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, df_src, label, color in [
    (axes[0], de_m, "DE (German)", COLOR_DE),
    (axes[1], en_m, "EN (English)", COLOR_EN),
]:
    wc = pd.to_numeric(df_src["WC"], errors="coerce").dropna()
    ax.hist(wc, bins=40, color=color, alpha=0.85, edgecolor="white")
    ax.axvline(wc.mean(), color="black", linestyle="--", lw=1.5,
               label=f"mean = {wc.mean():.1f}")
    ax.axvline(wc.median(), color="red", linestyle=":", lw=1.5,
               label=f"median = {wc.median():.1f}")
    ax.set_title(f"Word Count – {label}", fontweight="bold")
    ax.set_xlabel("Word count (WC)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

plt.suptitle("Word Count Distribution: German Original vs English Translation",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("06_wc_distribution.png")
plt.show()


# ── Fig 7: Average-score distribution per dimension ──────────────────────────
all_dims = sorted(de["dimension"].unique())
fig, axes = plt.subplots(1, len(all_dims), figsize=(4 * len(all_dims), 5), sharey=True)
if len(all_dims) == 1:
    axes = [axes]

for ax, dim in zip(axes, all_dims):
    de_scores = de.loc[de["dimension"] == dim, "average_score"].dropna()
    en_scores = en.loc[en["dimension"] == dim, "average_score"].dropna()
    data = pd.DataFrame({
        "score": pd.concat([de_scores, en_scores], ignore_index=True),
        "lang" : ["DE"] * len(de_scores) + ["EN"] * len(en_scores),
    })
    sns.boxplot(data=data, x="lang", y="score",
                palette={"DE": COLOR_DE, "EN": COLOR_EN}, ax=ax,
                order=["DE", "EN"])
    ax.set_title(dim.replace("d", "D").replace("_", "\n"), fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("average_score" if ax == axes[0] else "")
    ax.set_ylim(0.5, 5.5)

plt.suptitle("Human Rating (average_score) per Dimension: DE vs EN",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("07_scores_per_dimension.png")
plt.show()


# ── Fig 8: LIWC profile heatmap per dimension (DE and EN side by side) ────────
PROFILE_KEYS = [k for k in common_lower
                if k in {"analytic", "clout", "authentic", "tone",
                         "cogproc", "insight", "cause", "tentat",
                         "function", "pronoun", "ppron",
                         "affect", "social", "family",
                         "focuspast", "focuspresent", "focusfuture",
                         "verb", "adj", "negate",
                         "power", "reward", "risk", "health", "death"}]
if not PROFILE_KEYS:
    PROFILE_KEYS = common_lower[:20]

de_m2 = de_m.copy()
en_m2 = en_m.copy()
for k in PROFILE_KEYS:
    de_m2[k] = pd.to_numeric(de_m[de_lower_map[k]], errors="coerce")
    en_m2[k] = pd.to_numeric(en_m[en_lower_map[k]], errors="coerce")

de_profile = de_m2.groupby(de_m["dimension"])[PROFILE_KEYS].mean()
en_profile = en_m2.groupby(en_m["dimension"])[PROFILE_KEYS].mean()

# Normalise each feature to [0,1] across both matrices combined for fair colour scale
combined_min = pd.concat([de_profile, en_profile]).min()
combined_max = pd.concat([de_profile, en_profile]).max()
de_norm = (de_profile - combined_min) / (combined_max - combined_min + 1e-9)
en_norm = (en_profile - combined_min) / (combined_max - combined_min + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(all_dims) * 1.2 + 2)),
                         sharey=True)
for ax, matrix, title in [
    (axes[0], de_norm, "DE (German)"),
    (axes[1], en_norm, "EN (English)"),
]:
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                linewidths=0.4, cbar=(ax == axes[1]))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Dimension" if ax == axes[0] else "")
    ax.tick_params(axis="x", rotation=40)

plt.suptitle("Mean LIWC Profile per Dimension  (normalised 0–1 per feature)\nDE vs EN",
             fontsize=12, fontweight="bold")
plt.tight_layout()
_save("08_profile_heatmap.png")
plt.show()


# ── Fig 9: Relative mean difference for ALL common features (ranked) ──────────
fig, ax = plt.subplots(figsize=(9, max(6, len(common_lower) * 0.3)))
ordered = stats["abs_mean_diff"].sort_values()
colors  = [COLOR_DE if stats.loc[f, "DE_mean"] > stats.loc[f, "EN_mean"]
           else COLOR_EN for f in ordered.index]
ax.barh(ordered.index, ordered.values, color=colors, alpha=0.8)
ax.set_xlabel("|DE_mean − EN_mean|", fontsize=10)
ax.set_title("Absolute Mean Difference for All Common LIWC Features\n"
             f"(blue = DE higher, orange = EN higher)", fontsize=11, fontweight="bold")
de_patch = plt.matplotlib.patches.Patch(color=COLOR_DE, label="DE higher")
en_patch = plt.matplotlib.patches.Patch(color=COLOR_EN, label="EN higher")
ax.legend(handles=[de_patch, en_patch], loc="lower right")
plt.tight_layout()
_save("09_abs_mean_diff_all.png")
plt.show()


print(f"\n{SEP}")
print(f"  All figures saved to: {OUT_DIR}")
print(SEP)
