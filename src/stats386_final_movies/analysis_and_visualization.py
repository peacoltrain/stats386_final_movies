import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr, ttest_ind
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path

def run_analysis_and_visualization(
    data_path="dataLoading/cleaned_movies.json",
    output_dir="figures/analysis_and_visualization"
):
    """
    Run full exploratory data analysis and generate figures.
    """

    os.makedirs(output_dir, exist_ok=True)

    millions_formatter = FuncFormatter(lambda x, _: f"{x/1e6:.0f}")
    package_dir = Path(__file__).parent
    data_path = package_dir / "dataLoading" / "cleaned_movies.json"
    df = pd.read_json(data_path)

    # === Load Data ===
    df = pd.read_json(data_path)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["profit"] = df["revenue"] - df["budget"]

    # === Summary Stats ===
    print("=== Summary Stats (USD Millions) ===")
    print(f"Total movies: {len(df)}")
    print(f"Average budget: {df['budget'].mean() / 1e6:.1f}M")
    print(f"Average revenue: {df['revenue'].mean() / 1e6:.1f}M")
    print(f"Average profit: {df['profit'].mean() / 1e6:.1f}M")

    # === Profit Over Time ===
    yearly = df.groupby("year")[["budget", "revenue", "profit"]].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=yearly, x="year", y="profit", marker="o")
    plt.title("Average Profit by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Profit (USD Millions)")
    plt.gca().yaxis.set_major_formatter(millions_formatter)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/profit_over_time.png")
    plt.close()

    # === Genre Analysis ===
    df["genre_list"] = df["genres"].apply(
        lambda g: [x["name"] for x in g] if isinstance(g, list) else []
    )
    df_genres = df.explode("genre_list")

    genre_profit = (
        df_genres.groupby("genre_list")["profit"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_profit.values, y=genre_profit.index)
    plt.xlabel("Average Profit (USD Millions)")
    plt.gca().xaxis.set_major_formatter(millions_formatter)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/profit_by_genre.png")
    plt.close()

    # === Budget vs Revenue ===
    df_br = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    corr, pval = pearsonr(df_br["budget"], df_br["revenue"])
    print(f"Budget–Revenue correlation: {corr:.3f} (p-value={pval:.3e})")

    plt.figure(figsize=(8, 6))
    sns.regplot(data=df_br, x="budget", y="revenue", scatter_kws={"alpha": 0.4})
    plt.gca().xaxis.set_major_formatter(millions_formatter)
    plt.gca().yaxis.set_major_formatter(millions_formatter)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/budget_vs_revenue.png")
    plt.close()

    # === Action vs Comedy t-test ===
    action = df_genres[df_genres["genre_list"] == "Action"]["profit"].dropna()
    comedy = df_genres[df_genres["genre_list"] == "Comedy"]["profit"].dropna()

    t_stat, p_val = ttest_ind(action, comedy, equal_var=False)
    print(f"Action vs Comedy t-test: t={t_stat:.2f}, p-value={p_val:.4f}")


if __name__ == "__main__":
    run_analysis_and_visualization()
