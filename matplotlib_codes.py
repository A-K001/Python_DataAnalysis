import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
print("=======================================")

# Matplotlib is the most common plotting library in Python.
# You usually work in this pattern:
#   1) Prepare data (often with NumPy / Pandas)
#   2) Create a figure + axes: fig, ax = plt.subplots()
#   3) Plot on the axes (ax.plot, ax.bar, ax.scatter, ...)
#   4) Customize (title, labels, legend, grid, limits, ticks)
#   5) Show or save: plt.show() / fig.savefig(...)

# -----------------------------
# IMPORTANT RULES / COMMON MISTAKES
# -----------------------------
# 1) Prefer the "object-oriented" API:
#       fig, ax = plt.subplots(); ax.plot(...); plt.show()
#    It avoids confusing global state that happens with plt.plot(...) everywhere.
#    plt.plot(...) is okay for quick interactive work, but not for scripts.
#    Always use fig, ax = plt.subplots() to create figures and axes.
#    ax is the Axes object you plot on (e.g., ax.plot(...), ax.set_title(...)). 
#    its different from plt because plt is a module-level state manager and ax is an instance of Axes class which represents 
#    a single plot area in a figure and provides methods to plot data and customize the plot which is more flexible and 
#    less error-prone because it avoids issues with global state by encapsulating plot-specific settings within the Axes 
#    object but plt is a module that manages global state for plotting.
#
# 2) Always label your axes and add titles. Plots without labels are hard to interpret.
#
# 3) Use plt.tight_layout() if labels overlap.
#
# 4) If you create many figures in a loop, call plt.close(fig) to avoid memory issues.
#
# 5) When using categorical x labels, you can pass strings directly to bar() or
#    use positions (np.arange) and set_xticks / set_xticklabels.
#
# 6) Pie charts: values should be non-negative. Many tiny slices become unreadable.
#
# 7) Pandas has a .plot(...) wrapper that uses matplotlib internally. It’s convenient,
#    but you still can (and should) pass ax=... for consistency.

# -----------------------------
# DATA PREPARATION (NumPy + Pandas)
# -----------------------------
rng = np.random.default_rng(42)

# NumPy arrays (fast numeric operations)
x = np.linspace(0, 10, 200)
y_sin = np.sin(x)
y_cos = np.cos(x)

# A Pandas DataFrame (labeled tabular data)
df = pd.DataFrame(
    {
        "x": x,
        "sin": y_sin,
        "cos": y_cos,
        "noise": rng.normal(0, 0.2, size=x.size),
    }
)

# -----------------------------
# 1) LINE GRAPH (ax.plot)
# Syntax:
#   ax.plot(x, y, label="...", color="...", linestyle="--", marker="o", linewidth=2)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["x"], df["sin"], label="sin(x)", color="C0", linewidth=2)
ax.plot(df["x"], df["cos"], label="cos(x)", color="C1", linestyle="--", linewidth=2)

ax.set_title("Line Graph: sin(x) and cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(df["x"], df["sin"], label="sin(x)", linewidth=2)
plt.plot(df["x"], df["cos"], label="cos(x)", linestyle="--", linewidth=2)

plt.title("Line Graph: sin(x) and cos(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 2) SCATTER PLOT (ax.scatter)
# Syntax:
#   ax.scatter(x, y, s=marker_size, c=color_or_values, alpha=0.7)
#
# Useful components:
#   - s: size of each point
#   - c: single color OR array of values for colormap
#   - cmap: colormap name (e.g., "viridis")
# -----------------------------
y_scatter = df["sin"] + df["noise"]
colors = df["x"]  # color points by x value

fig, ax = plt.subplots(figsize=(8, 4))
sc = ax.scatter(df["x"], y_scatter, s=25, c=colors, cmap="viridis", alpha=0.9, edgecolor="none")
ax.set_title("Scatter Plot: sin(x) with noise (colored by x)")
ax.set_xlabel("x")
ax.set_ylabel("sin(x) + noise")
ax.grid(True, alpha=0.3)

# Colorbar explains the mapping of values to colors
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("x value")

plt.tight_layout()
plt.show()

# -----------------------------
# 3) BAR CHART (ax.bar)
# Syntax:
#   ax.bar(categories, values, width=0.8)
#   ax.bar(x_positions, values); ax.set_xticks(...); ax.set_xticklabels(...)
#
# Common mistakes:
#   - mismatched lengths for categories and values
#   - forgetting to rotate x labels when there are many
# -----------------------------
categories = ["A", "B", "C", "D", "E"]
values = rng.integers(10, 60, size=len(categories))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(categories, values, color="steelblue")
ax.set_title("Bar Chart: Category counts")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Example: grouped bar chart (two series per category)
values_2 = rng.integers(10, 60, size=len(categories))
xpos = np.arange(len(categories))
w = 0.4

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(xpos - w / 2, values, width=w, label="Series 1")
ax.bar(xpos + w / 2, values_2, width=w, label="Series 2")
ax.set_xticks(xpos)
ax.set_xticklabels(categories)
ax.set_title("Grouped Bar Chart")
ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 4) PIE CHART (ax.pie)
# Syntax:
#   ax.pie(values, labels=labels, autopct="%.1f%%", startangle=90)
#
# Notes:
#   - Pie charts become hard to read with many categories.
#   - startangle rotates the chart.
#   - autopct formats percentage labels.
# -----------------------------
pie_values = np.array([35, 25, 20, 15, 5])
pie_labels = ["Rent", "Food", "Transport", "Savings", "Other"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    pie_values,
    labels=pie_labels,
    autopct="%.1f%%",
    startangle=90,
    pctdistance=0.8,
)
ax.set_title("Pie Chart: Monthly budget breakdown")
# Makes the pie a circle (not an oval) when figure isn't square
ax.axis("equal")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) BOX AND WHISKERS (ax.boxplot)
# Syntax:
#   ax.boxplot(data, labels=[...], showmeans=True)
#
# What it shows:
#   - median (line inside the box)
#   - IQR (box: 25th to 75th percentile)
#   - whiskers (range excluding outliers by default)
#   - outliers (points)
#
# Common mistakes:
#   - passing a 2D array with wrong orientation (expects sequences per box)
# -----------------------------
group1 = rng.normal(loc=0, scale=1.0, size=200)
group2 = rng.normal(loc=1.0, scale=1.2, size=200)
group3 = rng.normal(loc=-0.5, scale=0.8, size=200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([group1, group2, group3], labels=["Group 1", "Group 2", "Group 3"], showmeans=True)
ax.set_title("Box and Whisker Plot: Distributions")
ax.set_ylabel("Value")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 6) MATPLOTLIB + PANDAS INTEGRATION
# Pandas uses matplotlib under the hood.
# Syntax:
#   df.plot(x="x", y=["sin", "cos"], ax=ax)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
df.plot(x="x", y=["sin", "cos"], ax=ax, title="Pandas DataFrame .plot() (matplotlib backend)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Another common Pandas plot: histogram
# Syntax:
#   df["col"].plot(kind="hist", bins=20, ax=ax)
fig, ax = plt.subplots(figsize=(7, 4))
df["noise"].plot(kind="hist", bins=25, ax=ax, color="gray", edgecolor="black", title="Histogram (Pandas + Matplotlib)")
ax.set_xlabel("noise")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 7) SAVING FIGURES
# Syntax:
#   fig.savefig("filename.png", dpi=200, bbox_inches="tight")
# bbox_inches="tight" reduces extra whitespace.
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.sin(x), label="sin(x)")
ax.set_title("Saving a figure example")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("saved_plot_example.png", dpi=200, bbox_inches="tight") # Save the figure with high resolution and tight bounding box
plt.show()

print("Finished matplotlib examples. A file 'saved_plot_example.png' should be created in the current working directory.")

# -----------------------------
# 8) MATPLOTLIB INTEGRATION WITH NumPy + Pandas (more examples)
# -----------------------------

# ---- NumPy -> Matplotlib: broadcasting + multiple lines from a 2D array
# Create several sine waves at different frequencies using broadcasting:
freqs = np.array([0.5, 1.0, 2.0, 3.0])
signals = np.sin(freqs[:, None] * x[None, :])  # shape: (n_freqs, n_points)

fig, ax = plt.subplots(figsize=(8, 4))
for i, f in enumerate(freqs):
    ax.plot(x, signals[i], label=f"sin({f}x)")
ax.set_title("NumPy -> Matplotlib: Broadcasting to generate multiple signals")
ax.set_xlabel("x")
ax.set_ylabel("amplitude")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

# ---- NumPy -> Matplotlib: image-like plot from a 2D array
# Heatmap of a simple 2D function z = sin(x) * cos(y)
xx = np.linspace(0, 4 * np.pi, 250)
yy = np.linspace(0, 2 * np.pi, 160)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X) * np.cos(Y)

fig, ax = plt.subplots(figsize=(8, 3.8))
im = ax.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    cmap="coolwarm",
)
ax.set_title("NumPy -> Matplotlib: 2D array as an image (imshow)")
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("sin(x) * cos(y)")
plt.tight_layout()
plt.show()

# ---- Pandas -> Matplotlib: time series + rolling mean + resampling
dates = pd.date_range("2024-01-01", periods=220, freq="D")
ts = pd.DataFrame(
    {
        "sales": 200 + rng.normal(0, 8, size=len(dates)).cumsum(),
        "costs": 120 + rng.normal(0, 6, size=len(dates)).cumsum(),
    },
    index=dates,
)

ts["sales_14d_ma"] = ts["sales"].rolling(14).mean()
ts["costs_14d_ma"] = ts["costs"].rolling(14).mean()

fig, ax = plt.subplots(figsize=(9, 4))
ts[["sales", "costs"]].plot(ax=ax, alpha=0.35, linewidth=1)
ts[["sales_14d_ma", "costs_14d_ma"]].plot(ax=ax, linewidth=2)
ax.set_title("Pandas -> Matplotlib: time series with rolling means")
ax.set_xlabel("date")
ax.set_ylabel("value")
ax.grid(True, alpha=0.3)
# plt.tight_layout()
plt.show()

# Resample to monthly averages (Pandas), then plot with Matplotlib customization
# monthly = ts[["sales", "costs"]].resample("MS").mean()

# fig, ax = plt.subplots(figsize=(8, 4))
# monthly.plot(kind="bar", ax=ax)
# ax.set_title("Pandas -> Matplotlib: monthly averages (resample + bar plot)")
# ax.set_xlabel("month")
# ax.set_ylabel("avg value")
# ax.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()

# ---- Pandas groupby -> Matplotlib: aggregated bar chart with error bars
cats = pd.Categorical(rng.choice(list("ABCDE"), size=300), ordered=True)
vals = rng.normal(loc=50, scale=10, size=300)
gdf = pd.DataFrame({"category": cats, "value": vals})

summary = gdf.groupby("category")["value"].agg(["mean", "std"]).sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(summary.index.astype(str), summary["mean"], yerr=summary["std"], capsize=4, color="tab:blue", alpha=0.85)
ax.set_title("Pandas -> Matplotlib: groupby aggregation (mean ± std)")
ax.set_xlabel("category")
ax.set_ylabel("value")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Mixing Pandas + Matplotlib: plot on one axis, then annotate/customize via Matplotlib
fig, ax = plt.subplots(figsize=(8, 4))
df.plot(x="x", y="sin", ax=ax, label="sin(x)")
df.plot(x="x", y="cos", ax=ax, label="cos(x)")
ax.axhline(0, color="black", linewidth=1, alpha=0.5)
ax.set_title("Pandas .plot() + Matplotlib customization (same ax)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


