import streamlit as st
import pandas as pd
import pickle
import asyncio
from aiohttp import ClientSession
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    return pd.read_csv('Model_Building/Anime_Data')

Anime = load_data()

# Load the similarity matrix
model = pickle.load(open('Model_Building/similarity.pkl', 'rb'))

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(Anime['Content'])

# Function to recommend similar anime
def recommend(anime_name):
    index = Anime[Anime['name'] == anime_name].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[index])
    recommended_animes = []
    for idx in indices.flatten()[1:]:
        recommended_animes.append(Anime.iloc[idx]['name'])
    return recommended_animes

# Function to fetch image
async def fetch_image_async(session, image_url):
    try:
        async with session.get(image_url, timeout=10) as response:
            if response.status == 200:
                image_bytes = await response.read()
                image = Image.open(BytesIO(image_bytes))
                return image
            else:
                return None
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None

# Streamlit app
st.set_page_config(layout="wide")

# Display image of anime_Universe with space
st.image("assets/background.jpg", use_column_width=True)
st.write("\n\n")  # Add space between the image and the next content
st.write("\n\n")  # Add space between the image and the next content

st.title('Anime Recommendation System')

# Dropdown to select anime
selected_anime = st.selectbox('Select an Anime:', Anime['name'].values)

# Button to trigger recommendation
if st.button('Recommend'):
    recommendations = recommend(selected_anime)

    # Create an asynchronous event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Fetch images asynchronously
    async def fetch_images():
        async with ClientSession() as session:
            tasks = [fetch_image_async(session, Anime.loc[Anime['name'] == anime_name, 'image_url'].values[0]) for anime_name in recommendations]
            images = await asyncio.gather(*tasks)
            return images
    
    # Display the recommendations with images
    images = loop.run_until_complete(fetch_images())   
    
    # Display the first 5 recommendations
    cols_1 = st.columns(5)
    for i in range(5):
        with cols_1[i]:
            try:
                anime_name = recommendations[i]
                image = images[i]
                if image is not None:
                    st.image(image, caption="", use_column_width=True)
                    st.markdown(f"**{anime_name}**", unsafe_allow_html=True)
            except IndexError:
                st.error(f"Anime {anime_name} not found in the dataset.")

    # Add some space between the first 5 and the remaining 5 recommendations
    st.write("\n\n")  # Add some blank lines

    # Display the remaining 5 recommendations
    cols_2 = st.columns(5)
    for i in range(5, 10):
        with cols_2[i % 5]:
            try:
                anime_name = recommendations[i]
                image = images[i]
                if image is not None:
                    st.image(image, caption="", use_column_width=True)
                    st.markdown(f"**{anime_name}**", unsafe_allow_html=True)
            except IndexError:
                st.error(f"Anime {anime_name} not found in the dataset.")
