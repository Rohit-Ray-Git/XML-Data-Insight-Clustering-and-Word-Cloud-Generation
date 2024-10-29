import os
import pandas as pd
import numpy as np
from io import BytesIO
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import string
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

# Set up file paths and variables
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to parse XML files and extract texts
def parse_xml(file_contents):
    texts = []
    for content in file_contents:
        # Use BytesIO to read XML content
        tree = ET.parse(BytesIO(content))
        root = ET.tostring(tree.getroot(), encoding='utf8').decode('utf8')
        parsed_article = BeautifulSoup(root, features='xml')
        paragraphs = parsed_article.find_all('para')
        text = ' '.join([p.text for p in paragraphs])
        texts.append(text)
    return texts

# Function to clean and preprocess text
def preprocessed_text(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = [lem.lemmatize(word.lower()) for word in text.split() 
              if word.lower() not in stop_words and word.isalpha()]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Streamlit UI
st.title("XML Data Insight: Clustering and Word Cloud Generationn")

uploaded_files = st.file_uploader("Upload XML Files", type=["xml"], accept_multiple_files=True)

if uploaded_files:
    st.success("Files uploaded successfully!")

    # Read uploaded files content
    file_contents = [uploaded_file.read() for uploaded_file in uploaded_files]

    # Parse XML files and clean text
    raw_data = parse_xml(file_contents)
    cleaned_data = [preprocessed_text(article) for article in raw_data]

    # Vectorization and TF-IDF transformation
    vectorizer = CountVectorizer(stop_words=list(stop_words))  # Convert to list
    X = vectorizer.fit_transform(cleaned_data).toarray()
    tfidf_transformer = TfidfTransformer()
    X = normalize(tfidf_transformer.fit_transform(X).toarray())

    # Finding the optimal number of clusters using Elbow Method
    def elbow_method(X, max_k=15):
        distortions = []
        K = range(1, max_k)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method for Optimal k')
        st.pyplot(plt)  # Use Streamlit to display the plot
        plt.clf()  # Clear the current figure for further use

    elbow_method(X)

    # User input for number of clusters
    true_k = st.number_input("Select number of clusters:", min_value=1, max_value=10, value=3)

    # Clustering
    kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(X)
    cluster_result = pd.DataFrame({'text': cleaned_data, 'group': kmeans.predict(X)})

    # Generating WordClouds for each cluster
    def plot_wordclouds(data, n_clusters):
        for i in range(n_clusters):
            words = ' '.join(data.loc[data['group'] == i, 'text'])
            wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)

            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            plt.title(f'Cluster {i}')
            st.pyplot(plt)  # Use Streamlit to display the plot
            plt.clf()  # Clear the current figure for further use

    plot_wordclouds(cluster_result, true_k)
