import seaborn as sns
import numpy as np
import pandas as pd
# Seaborn is built on top of Matplotlib and works best with Pandas DataFrames.
# This file is a "mini handbook" with examples and common mistakes to avoid.

import matplotlib.pyplot as plt

# -----------------------------
# 0) Setup: theme + sample data
# -----------------------------
# Syntax:
#   sns.set_theme(style="whitegrid", context="notebook", palette="deep")
# Rules:
#   - Always set a theme early to keep a consistent look across plots.
#   - Use a random seed for reproducible examples.
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

rng = np.random.default_rng(42)

n = 200
df = pd.DataFrame(
    {
        "x": rng.normal(loc=0, scale=1, size=n),
        "y": rng.normal(loc=0, scale=1, size=n),
        "group": rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25]),
    }
)

# Create something that looks like a time series
dates = pd.date_range("2025-01-01", periods=60, freq="D")
ts = pd.DataFrame(
    {
        "date": dates,
        "value": np.cumsum(rng.normal(0, 1, size=len(dates))) + 10,
        "category": rng.choice(["North", "South"], size=len(dates)),
    }
)

# Add a derived column using NumPy (integration example)
# Important rule:
#   - Prefer vectorized operations (NumPy/Pandas) instead of Python loops.
df["radius"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
df["y_noisy"] = df["y"] + rng.normal(0, 0.35, size=n)

# -----------------------------
# 1) Scatter plot
# -----------------------------
# Syntax (object-oriented style recommended):
#   fig, ax = plt.subplots()
#   sns.scatterplot(data=df, x="x", y="y", hue="group", style="group", ax=ax)
# Common mistakes:
#   - Passing arrays of different lengths to x and y.
#   - Using hue for a numeric column without understanding the continuous color scale.
#   - Forgetting to call plt.show() in scripts (some environments require it).
fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(
    data=df,
    x="x",
    y="y_noisy",
    hue="group",         # color encodes category
    style="group",       # marker style encodes category
    s=60,                # marker size
    alpha=0.8,
    ax=ax,
)
ax.set_title("Scatter plot (x vs y_noisy) with hue/style by group")
ax.set_xlabel("x")
ax.set_ylabel("y_noisy")
ax.legend(title="group", loc="best")
sns.despine(ax=ax)
plt.tight_layout()
plt.show()

# -----------------------------
# 2) Line graph (time series)
# -----------------------------
# Syntax:
#   sns.lineplot(data=ts, x="date", y="value", hue="category", ax=ax)
# Important rules:
#   - Ensure x is sorted for time-series. If not, sort_values first.
#   - Missing dates can create visual gaps; decide whether to reindex/forward-fill.
ts_sorted = ts.sort_values("date")

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(
    data=ts_sorted,
    x="date",
    y="value",
    hue="category",
    linewidth=2,
    ax=ax,
)
ax.set_title("Line plot (time series) with hue by category")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Bar chart
# -----------------------------
# Seaborn barplot shows an estimate (mean by default) with uncertainty.
# Syntax:
#   sns.barplot(data=df, x="group", y="radius", estimator="mean", errorbar="sd")
# Important rules:
#   - barplot aggregates data. If you want raw counts, use sns.countplot.
#   - Newer Seaborn uses errorbar=... (not ci=...). Use one or the other depending on version.
# Common mistake:
#   - Expecting barplot to show each row; it summarizes.
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=df,
    x="group",
    y="radius",
    estimator="mean",
    errorbar="sd",  # show standard deviation as error bars
    ax=ax,
)
ax.set_title("Bar plot (mean radius per group) with SD error bars")
ax.set_xlabel("Group")
ax.set_ylabel("Mean radius")
plt.tight_layout()
plt.show()

# If you want counts per category:
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="group", order=["A", "B", "C"], ax=ax)
ax.set_title("Count plot (number of rows per group)")
ax.set_xlabel("Group")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 4) Box-and-whisker plot
# -----------------------------
# Syntax:
#   sns.boxplot(data=df, x="group", y="radius", ax=ax)
# Notes:
#   - Box = IQR (25th to 75th percentile), line = median.
#   - "Whiskers" typically extend to 1.5 * IQR; points outside are outliers.
# Common mistakes:
#   - Comparing groups with very different sample sizes without noticing.
#   - Letting extreme outliers compress the rest of the plot; consider log scale or showfliers=False.
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="group", y="radius", ax=ax)
sns.stripplot(  # show individual points on top (optional)
    data=df,
    x="group",
    y="radius",
    color="black",
    alpha=0.35,
    size=3,
    jitter=0.25,
    ax=ax,
)
ax.set_title("Box plot with overlaid points (radius by group)")
ax.set_xlabel("Group")
ax.set_ylabel("Radius")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Pie chart (not a Seaborn chart)
# -----------------------------
# Seaborn does not provide a pie chart function.
# Use Matplotlib directly if you really need one (many analysts avoid pie charts).
# Syntax (Matplotlib):
#   ax.pie(values, labels=labels, autopct="%.1f%%")
group_counts = df["group"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    group_counts.values,
    labels=group_counts.index,
    autopct="%.1f%%",
    startangle=90,
)
ax.set_title("Pie chart (Matplotlib) - group distribution")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Faceting / small multiples (powerful for comparisons)
# -----------------------------
# Syntax:
#   g = sns.FacetGrid(df, col="group")
#   g.map_dataframe(sns.scatterplot, x="x", y="y_noisy")
# Common mistake:
#   - Forgetting map_dataframe when passing "data=" style arguments.
g = sns.FacetGrid(df, col="group", height=3, aspect=1.1, sharex=True, sharey=True)
g.map_dataframe(sns.scatterplot, x="x", y="y_noisy", alpha=0.8)
g.set_axis_labels("x", "y_noisy")
g.fig.suptitle("FacetGrid: scatter per group", y=1.03)
plt.show()

# -----------------------------
# 7) Quick list of key components you can manipulate
# -----------------------------
# Most seaborn functions accept:
#   - data= (DataFrame), x=, y= (column names)
#   - hue= (color grouping), style= (markers/linestyles), size=
#   - palette= (color palette), order= (category order), hue_order=
#   - ax= (Matplotlib Axes) to control layout precisely
#
# Then use Matplotlib to polish:
#   ax.set_title(...), ax.set_xlabel(...), ax.set_ylabel(...)
#   ax.set_xlim(...), ax.set_ylim(...)
#   ax.legend(...), ax.grid(...)
#   plt.tight_layout()
#
# Common pitfalls / mistakes:
#   1) Wide vs long data:
#      - Seaborn prefers "long/tidy" data: one column per variable, one row per observation.
#   2) Wrong dtypes:
#      - Categorical columns should often be strings or pandas 'category' dtype.
#   3) Missing values:
#      - NaNs are often dropped silently; check df.isna().sum() when plots look odd.
#   4) Overplotting:
#      - Use alpha, smaller markers, or aggregation (e.g., lineplot with estimator).
#   5) Version differences:
#      - errorbar=... is new; older code may use ci=...
#
# Tip: If you are building many charts, wrap plotting code in functions and pass ax=.