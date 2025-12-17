import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the cleaned movie data
data_path = os.path.join(os.path.dirname(__file__), "dataLoading", "cleaned_movies.json")
df = pd.read_json(data_path)

# Process the data
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["year"] = df["release_date"].dt.year
df["release_date"] = df["release_date"].dt.date
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
df["profit"] = df["revenue"] - df["budget"]

# Convert genres and production_companies to readable strings
df["genres"] = df["genres"].apply(lambda x: ", ".join([g.get("name", "") for g in x]) if isinstance(x, list) else "")
df["production_companies"] = df["production_companies"].apply(lambda x: ", ".join([p.get("name", "") for p in x]) if isinstance(x, list) else "")

# Streamlit app
st.title("Movie Data Exploration App")

st.header("Summary Statistics")
st.write("Total movies:", len(df))
st.write("Average budget:", f"${df['budget'].mean():,.2f}")
st.write("Average revenue:", f"${df['revenue'].mean():,.2f}")
st.write("Average profit:", f"${df['profit'].mean():,.2f}")

st.header("Data Preview")
st.dataframe(df.head(10))

st.header("Profit Over Time")
yearly_profit = df.groupby("year")["profit"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=yearly_profit, x="year", y="profit", ax=ax)
ax.set_title("Average Profit by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Average Profit")
st.pyplot(fig)

st.header("Filter by Year")
selected_year = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), (2000, 2020))
filtered_df = df[(df["year"] >= selected_year[0]) & (df["year"] <= selected_year[1])]
st.write(f"Movies in selected years: {len(filtered_df)}")
st.dataframe(filtered_df.head(10))
