import json
import pandas as pd

# Load the full JSON file
with open("movies.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Remove rows that are error responses
df = df[df['success'] != False]

# Filter to only U.S. movies
df = df[df['production_countries'].apply(
    lambda countries: any(c.get("iso_3166_1") == "US" for c in countries)
)]

# Filter to only big-budget films (adjust threshold as needed)
df = df[df['budget'] > 1_000_000]

# Keep only relevant fields
columns_to_keep = [
    "budget", "revenue", "release_date", "runtime",
    "genres", "production_companies", "popularity", "vote_average"
]
df = df[columns_to_keep]

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned dataset
df.to_json("cleaned_movies.json", orient="records", indent=2)

print("✅ Cleaned data saved to cleaned_movies.json")
print(f"Remaining entries: {len(df)}")
