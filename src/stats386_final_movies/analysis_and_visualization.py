import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import numpy as np
from matplotlib.ticker import FuncFormatter
millions_formatter = FuncFormatter(lambda x, _: f"{x/1e6:.0f}")
from matplotlib.ticker import MaxNLocator



os.makedirs("figures/analysis_and_visualization", exist_ok=True)
# === Load Cleaned Data ===
df = pd.read_json("dataLoading/cleaned_movies.json")

# Convert release_date to datetime
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["year"] = df["release_date"].dt.year

# Drop rows without year
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# Compute profit
df["profit"] = df["revenue"] - df["budget"]

# === Summary Stats ===
print("=== Summary Stats (USD Millions) ===")
print(f"Total movies: {len(df)}")
print(f"Average budget: {df['budget'].mean() / 1e6:.1f}M")
print(f"Average revenue: {df['revenue'].mean() / 1e6:.1f}M")
print(f"Average profit: {df['profit'].mean() / 1e6:.1f}M")


# === Profit Over Time ===
yearly = (
    df.groupby("year")[["budget", "revenue", "profit"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly, x="year", y="profit", marker="o")
plt.title("Average Profit by Year")
plt.xlabel("Year")
plt.ylabel("Average Profit (USD Millions)")
plt.gca().yaxis.set_major_formatter(millions_formatter)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/analysis_and_visualization/profit_over_time.png")
plt.show()

# === Profit by Genre ===
def extract_genres(g):
    return [x["name"] for x in g] if isinstance(g, list) else []

df["genre_list"] = df["genres"].apply(extract_genres)
df_genres = df.explode("genre_list")

genre_profit = (
    df_genres.groupby("genre_list")["profit"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_profit.values, y=genre_profit.index)
plt.title("Average Profit by Genre")
plt.xlabel("Average Profit(USD Millions)")
plt.ylabel("Genre")
plt.gca().xaxis.set_major_formatter(millions_formatter)
plt.tight_layout()
plt.savefig("figures/analysis_and_visualization/profit_by_genre.png")
plt.show()

# === Profit by Production Company ===
def extract_companies(c):
    return [x["name"] for x in c] if isinstance(c, list) else []

df["company_list"] = df["production_companies"].apply(extract_companies)
df_companies = df.explode("company_list")

company_profit = (
    df_companies.groupby("company_list")["profit"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(12, 6))
sns.barplot(x=company_profit.values, y=company_profit.index)
plt.title("Top 15 Production Companies by Average Profit")
plt.xlabel("Average Profit(USD Millions)")
plt.ylabel("Production Company")
plt.gca().xaxis.set_major_formatter(millions_formatter)
plt.tight_layout()
plt.savefig("figures/analysis_and_visualization/profit_by_production_company.png")
plt.show()

# === Budget vs Revenue Analysis ===
df_br = df[(df["budget"] > 0) & (df["revenue"] > 0)]

corr, pval = pearsonr(df_br["budget"], df_br["revenue"])
print(f"Budget–Revenue correlation: {corr:.3f} (p-value={pval:.3e})")

plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_br,
    x="budget",
    y="revenue",
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "black"}
)
plt.title("Relationship Between Budget and Revenue")
plt.xlabel("Budget (USD Millions)")
plt.ylabel("Revenue (USD Millions)")
ax = plt.gca()
ax.xaxis.set_major_formatter(millions_formatter)
ax.yaxis.set_major_formatter(millions_formatter)
plt.tight_layout()
plt.savefig("figures/analysis_and_visualization/budget_vs_revenue.png")
plt.show()


# === Profit Distribution by Decade ===
df["decade"] = (df["year"] // 10) * 10

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="decade", y="profit")
plt.title("Profit Distribution by Decade")
plt.xlabel("Decade")
plt.ylabel("Profit (USD Millions)")
plt.ylim(-2e8, 1.5e9)
plt.tight_layout()
plt.savefig("figures/analysis_and_visualization/profit_by_decade.png")
plt.show()


# === Genre Profit Comparison: Action vs Comedy ===
action = df_genres[df_genres["genre_list"] == "Action"]["profit"]
comedy = df_genres[df_genres["genre_list"] == "Comedy"]["profit"]

action = action.dropna()
comedy = comedy.dropna()

t_stat, p_val = ttest_ind(action, comedy, equal_var=False)
print(f"Action vs Comedy t-test: t={t_stat:.2f}, p-value={p_val:.4f}")
