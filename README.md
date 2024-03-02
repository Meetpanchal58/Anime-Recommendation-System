# Anime Recommendation System

## Introduction
This project implements a recommendation system for anime using cosine similarity. The system suggests similar anime based on user input and provides posters of recommended anime.

## Project Overview
The project is organized into three main steps:

### Step 1: Data Collection
- Scraped anime data from Anime-Planet website using Selenium.
- Extracted necessary details and saved the data as `Anime-planet.csv`.

### Step 2: Model Building
- Performed data cleaning and exploratory data analysis (EDA) on the collected anime data.
- Built a recommendation model using cosine similarity.
- Saved the model as `similarity.pkl` and extracted data as `anime_data.csv`.

### Step 3: Deployment
- Deployed the recommendation system using Streamlit.
- Users can input an anime name and click on the "Recommend" button to receive 10 recommended anime along with their posters.

## File Structure
- `data_collection/`: Contains scripts for data collection.
- `model_building/`: Contains scripts for data cleaning, EDA, and model building.
- `assets/`: Contains background image for the web application.
- `Anime-planet.csv`: Scraped anime data from Anime-Planet website.
- `anime_data.csv`: Extracted data for deployment.
- `similarity.pkl`: Pickled similarity model.
- `app.py`: Main script for deploying the recommendation system using Streamlit.

## Usage
1. Install the necessary dependencies listed in `requirements.txt`.
2. Run the data collection scripts in `data_collection/`.
3. Run the model building scripts in `model_building/`.
4. Execute `app.py` to deploy the recommendation system using Streamlit.
