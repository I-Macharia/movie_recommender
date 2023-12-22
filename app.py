import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    api_key = "53607277a4abba625e13562a61ea99d5"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    return "https://image.tmdb.org/t/p/w500/" + poster_path

# Function to save feedback to a text file
def save_feedback(name, email, phone, feedback_text):
    feedback_file_path = "feedback.txt"
    with open(feedback_file_path, "a", encoding="utf-8") as feedback_file:
        # Write feedback information with clear distinctions
        feedback_file.write(f"Feedback: {feedback_text}\n")
        feedback_file.write(f"Name: {name}\n")
        feedback_file.write(f"Email: {email}\n")
        feedback_file.write(f"Phone: {phone}\n")
        feedback_file.write("=" * 30 + "\n") 

def hybrid_recommendations(title, cosine_sim2, movies_credits):
    # Get the index of the movie that matches the title
    idx = movies_credits[movies_credits['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    ind = [x for x, _ in sim_scores]

    # Grab the title, id, vote_average, and vote_count of the top 10 most similar movies
    tit = []
    movieid = []
    vote_average = []
    vote_count = []
    for x in ind:
        tit.append(movies_credits.iloc[x]['title'])
        movieid.append(movies_credits.iloc[x]['id'])
        vote_average.append(movies_credits.iloc[x]['vote_average'])
        vote_count.append(movies_credits.iloc[x]['vote_count'])

    return pd.DataFrame({
        'index': ind,
        'title': tit,
        'id': movieid,
        'vote_average': vote_average,
        'vote_count': vote_count
    }).set_index('index').sort_values(by='vote_average', ascending=False)


# unzip the archive.zip file
with zipfile.ZipFile('tmdb_5000_credits.zip', 'r') as zip_ref:
  zip_ref.extractall('extracted_files')

# Load and merge the TMDB movies.csv and TMDB credits.csv
tmdb_movies = pd.read_csv(r"tmdb_5000_movies.csv")
tmdb_credits = pd.read_csv(r"tmdb_5000_credits.csv")

# Drop the Title column in Movies Dataset
tmdb_movies.drop(['title'], axis=1, inplace=True)

# Identify the columns that are common and need to be merged
tmdb_credits.columns = ['id', 'title', 'cast', 'crew']

# Adjust column names if necessary
movies_credits = pd.merge(tmdb_credits, tmdb_movies, on='id')

# Create the 'soup' column by combining relevant features
movies_credits['soup'] = movies_credits['overview'] + ' ' + movies_credits['genres'] + ' ' + movies_credits['keywords']

# Fill missing values in 'soup' column with an empty string
movies_credits['soup'].fillna('', inplace=True)

# Calculate cosine similarity
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(movies_credits['soup'])
cosine_sim2 = cosine_similarity(cv_matrix, cv_matrix)

movie_list = movies_credits['title'].values



def main():  # sourcery skip: extract-duplicate-method, extract-method
    menu = ['About', 'Movies', 'Feedback']

    selection = st.sidebar.selectbox("Select Menu", menu)

    # Add other sections using st.markdown()
    st.sidebar.header("MovieFlix!")


    st.sidebar.subheader("About")
    st.sidebar.write("MovieFlix is a recommendation engine that provides suggestions for movies to watch based on a title")
    if selection == "About" : 
        with st.sidebar.expander(""):
            pass

        st.title("About Movie Recommender")

        st.write("""
    Welcome to Movie Recommender, your go-to platform for discovering new and exciting movies tailored just for you! Our recommendation system is designed to provide you with personalized movie suggestions based on your preferences.
    """)

        st.header("How it Works")
        st.write("""
    The Movie Recommender utilizes a sophisticated algorithm that analyzes your selected movie title and considers various factors such as genre, director, cast, and user ratings. Simply choose a movie from the dropdown list, and our system will generate a list of recommended movies that share similar characteristics or themes.
    """)

        st.header("Explore and Enjoy")
        st.write("""
    Whether you're a film enthusiast or just looking for something new to watch, Movie Recommender is here to make your movie-watching experience more enjoyable. Discover hidden gems, explore different genres, and find movies that match your taste.
    """)

        st.header("Feedback and Support")
        st.write("""
    We value your feedback! If you have any suggestions, questions, or issues, feel free to reach out to our support team. Your input helps us improve and enhance the Movie Recommender for everyone.
    """)

        st.header("Happy Watching!")
        st.write("""
    Thank you for choosing Movie Recommender. Sit back, relax, and enjoy the world of cinema with personalized recommendations made just for you.
    """)

        st.markdown("&copy; 2023 Movie Recommender. All rights reserved.")


        if selection == "Movies" : 
            with st.sidebar.expander(""):
                pass
            # Add header
            st.header('Movie Recommender System')
            selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
            recommended_movies = hybrid_recommendations(selected_movie, cosine_sim2, movies_credits)

            if st.button('Show Recommendation'):


                columns = st.columns(5)
                displayed_titles = []
                for _, movie in recommended_movies.iterrows():
                    title = movie['title']
                    if title not in displayed_titles:
                        displayed_titles.append(title)
                    with columns[0]:
                        st.text(title)
                        st.image(fetch_poster(movie['id']))
                    columns = columns[1:] + [columns[0]]

    if selection == "Feedback": 
        with st.sidebar.expander(""):
            pass
        # Add header
        st.title("Feedback for Movie Recommender")
        
        with st.form(key='contact-form'):
                st.markdown("Any queries or suggestions? Please fill out the form below and we will get back to you as soon as possible.")
                # Add a text input for users to provide feedback
                message = st.text_area(label='Enter your message here')
                               
                st.markdown("### Contact Information")
                name = st.text_input(label='Name')
                email = st.text_input(label='Email')
                phone = st.text_input(label='Phone')
                feedback_text = "Feedback: " + message  
                st.markdown("###")
                # Add a button to submit feedback
                if st.form_submit_button("Submit Feedback"):
                    if feedback_text:
                        # Save feedback to a file
                        save_feedback(name, email, phone, feedback_text)
                        st.success("Feedback submitted successfully!")
                    else:
                        st.warning("Please enter your feedback before submitting.") 

        st.write("""
    We value your feedback! Please share your thoughts, suggestions, or any issues you encountered while using Movie Recommender. Your feedback helps us enhance the user experience and improve our services.
    """)

    

    

if __name__ == "__main__":
    main()
