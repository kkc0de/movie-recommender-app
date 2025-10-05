import gradio as gr
import pickle
import pandas as pd

# --- LOAD SAVED FILES ---
# Load the DataFrame and the similarity matrix from the files you saved
try:
    movies_df = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    # This is a fallback in case the app is run without the files
    # In a real deployed app, you'd handle this more robustly
    movies_df = pd.DataFrame()
    similarity = None


# --- DEFINE THE RECOMMENDATION FUNCTION ---
# This is the core logic from your notebook
def recommend(movie_title):
    if movies_df.empty or similarity is None:
        return "Model files not loaded. Please check the setup."
        
    # Find the index of the movie that was selected
    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        return "Movie not found in the dataset."

    # Get the similarity scores for that movie
    distances = similarity[movie_index]
    
    # Get the top 5 most similar movies (excluding the movie itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # Get the titles of the recommended movies
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies_df.iloc[i[0]].title)
    
    # Return the list of movies as a formatted string
    return "\n".join(recommended_movies)


# --- CREATE THE GRADIO INTERFACE ---
# Create a dropdown menu with all the movie titles as choices
movie_titles_list = movies_df['title'].tolist()

iface = gr.Interface(
    fn=recommend,
    inputs=gr.Dropdown(movie_titles_list, label="Select a Movie You Like"),
    outputs=gr.Textbox(label="Top 5 Recommended Movies", lines=5),
    title="🎬 Content-Based Movie Recommender",
    description="Choose a movie from the list to get recommendations for similar movies."
)

# --- LAUNCH THE APP ---
iface.launch()