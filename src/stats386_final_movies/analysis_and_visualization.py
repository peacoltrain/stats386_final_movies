import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
print("=== Summary Stats ===")
print("Total movies:", len(df))
print("Average budget:", df["budget"].mean())
print("Average revenue:", df["revenue"].mean())
print("Average profit:", df["profit"].mean())

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
plt.ylabel("Average Profit")
plt.grid(True)
plt.tight_layout()
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
plt.xlabel("Average Profit")
plt.ylabel("Genre")
plt.tight_layout()
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
plt.xlabel("Average Profit")
plt.ylabel("Production Company")
plt.tight_layout()
plt.show()
