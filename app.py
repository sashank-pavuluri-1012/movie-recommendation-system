import streamlit as st
import pickle
import pandas as pd
movies = pickle.load(open('movies.pkl', 'rb'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)
st.title("Movie Recommendation System 🎬")
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies
if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)
