import json
import pandas as pd

# Load your JSON file
with open("movies.json", "r") as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Show basic info
print("Total movies:", len(df))
print("\nColumns:", df.columns.tolist())
print("\nSample rows:")
print(df[["title", "release_date", "budget", "revenue"]].head(10))
