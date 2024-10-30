# Importing necessary libraries and setting up the initial configurations
import os
import pandas as pd
import numpy as np
from io import BytesIO
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import string
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.spatial.distance import cdist
import streamlit as st

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set the app tab name
st.set_page_config(page_title="XML Data Insight: Clustering and Word Cloud Generation")

# Set up background image using CSS
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    /* Center alignment for all content */
    .center {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    h1 {{
        color: #f5074f;  /* Title color */
    }}
    p {{
        color: #f5074f;  /* Subtitle color */
    }}
    h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set a custom background image by providing the path to the image file
image_base64 = get_base64_image("image.jpg")  # Replace with the path to your chosen image
set_background(image_base64)

# Center the title and subtitle
st.markdown("<h1 class='center'>XML Data Insight</h1>", unsafe_allow_html=True)
st.markdown("<p class='center'>Clustering and Word Cloud Generation Dashboard</p>", unsafe_allow_html=True)

# Center the file uploader
with st.container():
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload XML Files (minimum 15 files)", type=["xml"], accept_multiple_files=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Process uploaded files
if uploaded_files:
    file_count = len(uploaded_files)
    st.success(f"{file_count} file(s) uploaded successfully!")

    if file_count < 15:
        st.warning("Please upload at least 15 XML files to proceed.")
    else:
        st.success("Sufficient files uploaded, proceeding with analysis...")

        # Read and preprocess XML files
        file_contents = [uploaded_file.read() for uploaded_file in uploaded_files]
        
        def parse_xml(file_contents):
            texts = []
            for content in file_contents:
                tree = ET.parse(BytesIO(content))
                root = ET.tostring(tree.getroot(), encoding='utf8').decode('utf8')
                parsed_article = BeautifulSoup(root, features='xml')
                paragraphs = parsed_article.find_all('para')
                text = ' '.join([p.text for p in paragraphs])
                texts.append(text)
            return texts

        def preprocessed_text(text):
            text = re.sub(r'\[[0-9]*\]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            tokens = [WordNetLemmatizer().lemmatize(word.lower()) for word in text.split() 
                      if word.lower() not in stopwords.words('english') and word.isalpha()]
            return ' '.join(tokens)

        raw_data = parse_xml(file_contents)
        cleaned_data = [preprocessed_text(article) for article in raw_data]

        # Vectorization and TF-IDF transformation
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(cleaned_data).toarray()
        X = normalize(TfidfTransformer().fit_transform(X).toarray())

        # Define the elbow method function
        @st.cache_data(show_spinner=False)
        def compute_elbow_method(X, max_k=15):
            distortions = []
            K = range(1, max_k)
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X)
                distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
            return distortions

        # Display the elbow plot
        if "elbow_plot_data" not in st.session_state:
            st.session_state.elbow_plot_data = compute_elbow_method(X)
        
        distortions = st.session_state.elbow_plot_data
        K = range(1, 15)
        
        fig, ax = plt.subplots()
        ax.plot(K, distortions, 'bx-')
        ax.set_xlabel('Values of k')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method for Optimal Number of Clusters')
        st.pyplot(fig)  # Display elbow plot

        # Center the cluster input and word cloud generation button
        with st.container():
            st.markdown("<div class='center'>", unsafe_allow_html=True)
            true_k = st.number_input("Select number of clusters:", min_value=1, max_value=10, value=3, key="cluster_input")
            if st.button("Generate Word Clouds"):
                kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
                kmeans.fit(X)
                cluster_result = pd.DataFrame({'text': cleaned_data, 'group': kmeans.predict(X)})

                def plot_wordclouds(data, n_clusters):
                    for i in range(n_clusters):
                        words = ' '.join(data.loc[data['group'] == i, 'text'])
                        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
                        fig, ax = plt.subplots(figsize=(10, 7))
                        ax.imshow(wordcloud, interpolation="bilinear")
                        ax.axis('off')
                        ax.set_title(f'Cluster {i}')
                        st.pyplot(fig)
                        plt.close(fig)

                plot_wordclouds(cluster_result, true_k)
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Please upload XML files to proceed.")
