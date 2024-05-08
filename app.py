import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Function to load data
@st.cache_data
def load_data():
    data = pd.read_csv('articles_075040.csv', encoding='latin-1')
    data['Content'] = data['Content'].fillna('')
    return data

df = load_data()

# Preparing the TF-IDF vectorizer and cosine similarity matrix
def vectorize_text(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = vectorize_text(df)

# Recommender function
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = df[df['Title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Top 5 recommendations
        article_indices = [i[0] for i in sim_scores]

        # Return titles along with their URLs
        return [(df.iloc[i]['Title'], df.iloc[i]['URL']) for i in article_indices]
    except IndexError:
        return "Title not found in the dataset."
    except Exception as e:
        return str(e)


# Streamlit user interface
st.title('News Article Recommender')
title = st.selectbox('Select Article Title to Get Recommendations:', df['Title'])

if st.button('Recommend'):
    recommendations = get_recommendations(title)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader('Top 5 Recommended Articles:')
        for title, url in recommendations:
            st.markdown(f"[{title}]({url})", unsafe_allow_html=True)