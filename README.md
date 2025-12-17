# Movie Analysis Package (STAT 386 Final Project)

This project analyzes profitability trends in U.S. theatrical films released between 2005 and 2025 using data collected from the TMDB (The Movie Database) API. The project implements a complete data science pipeline including data collection, cleaning, analysis, visualization, documentation, and an interactive Streamlit app.

---

## Project Components

- **Data Collection**
  - Scripts to collect movie IDs and metadata from the TMDB API
  - API rate limits respected during collection
- **Data Cleaning**
  - Filters movies to U.S. theatrical releases with valid budget, revenue, and release dates
    - Final cleaned dataset contains 3,503 movies
- **Analysis & Visualization**
  - Exploratory data analysis and statistical tests
  - Visualizations of profit over time, profit by genre, studio performance, and budget–revenue relationships
- **Interactive App**
  - Streamlit app for interactive exploration of the cleaned dataset
- **Documentation & Report**
  - Quarto website hosted via GitHub Pages
  - Includes package documentation, tutorial, and final written report

---