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


st.title("Movie Data Exploration App")

st.header("Summary Statistics")
st.write("Total movies:", len(df))
st.write("Average budget (USD Millions):", f"{df['budget'].mean() / 1e6:.1f}")
st.write("Average revenue (USD Millions):", f"{df['revenue'].mean() / 1e6:.1f}")
st.write("Average profit (USD Millions):", f"{df['profit'].mean() / 1e6:.1f}")

st.header("Data Preview")
st.dataframe(df.head(10))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["General Profit", "Year Filter", "Budget vs Revenue Scatter", "Profit by Genre", "Top Studios Profit"])
with tab1:
    st.header("Profit Over Time")
    yearly_profit = df.groupby("year")["profit"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=yearly_profit, x="year", y="profit", ax=ax)
    ax.set_title("Average Profit by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Profit")
    st.pyplot(fig)

with tab2:
    st.header("Filter by Year")
    selected_year = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), (2005, 2020))
    filtered_df = df[(df["year"] >= selected_year[0]) & (df["year"] <= selected_year[1])]
    st.write(f"Movies in selected years: {len(filtered_df)}")
    st.dataframe(filtered_df.head(10))

with tab3:
    st.header("Budget vs Revenue Scatter Plot")
    df['profit_positive'] = df['profit'] > 0
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=df, x="budget", y="revenue", hue="profit_positive", ax=ax, palette={True: 'green', False: 'red'}, alpha=.5)
    sns.regplot(data=df, x="budget", y="revenue", ax=ax, scatter=False, color='black', line_kws={'linewidth':2})
    ax.set_title("Budget vs Revenue Colored by Profit")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Revenue")
    # ax.set(xscale="log")
    st.pyplot(fig)

with tab4:
    st.header("Profit by Genre")
    exploded_genres = df.assign(genres=df['genres'].str.split(', ')).explode('genres')
    genre_profit = exploded_genres.groupby('genres')['profit'].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=genre_profit, x="genres", y="profit", ax=ax)
    ax.set_title("Average Profit by Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Average Profit")
    plt.xticks(rotation=90)
    st.pyplot(fig)

with tab5:
    st.header("Top 5 Production Studios Profit Over Time")
    exploded_companies = df.assign(production_companies=df['production_companies'].str.split(', ')).explode('production_companies')
    top_companies = exploded_companies['production_companies'].value_counts().head(5).index.tolist()
    
    selected_studios = []
    for studio in top_companies:
        if st.checkbox(f"Show {studio}", value=True):
            selected_studios.append(studio)
    
    if selected_studios:
        filtered = exploded_companies[exploded_companies['production_companies'].isin(selected_studios)]
        studio_year_profit = filtered.groupby(['production_companies', 'year'])['profit'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=studio_year_profit, x='year', y='profit', hue='production_companies', ax=ax)
        ax.set_title("Average Profit Over Time for Selected Studios")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Profit")
        st.pyplot(fig)
    else:
        st.write("Please select at least one studio to display.")
