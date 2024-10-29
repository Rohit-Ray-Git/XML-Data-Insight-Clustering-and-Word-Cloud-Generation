# 🌐 XML Clustering and Word Cloud Generation

## 📊 Project Overview

This project provides a web application that allows users to upload XML files, extract text data, and perform clustering analysis on the content. It utilizes Natural Language Processing (NLP) techniques to preprocess the text, the KMeans algorithm for clustering, and generates word clouds for visual representation of the clusters.

## ✨ Features

- 📁 Upload multiple XML files for processing.
- 📜 Parse and clean text data extracted from the XML files.
- 📊 Perform clustering using the KMeans algorithm.
- 📈 Visualize the optimal number of clusters using the Elbow Method.
- 🌈 Generate word clouds for each cluster to illustrate the most common terms.

## 💻 Technologies Used

- **Python**: Main programming language.
- **Streamlit**: Framework for creating the web application.
- **NLTK**: Library for natural language processing.
- **Scikit-learn**: For implementing the KMeans clustering algorithm and text vectorization.
- **BeautifulSoup**: For parsing XML files.
- **WordCloud**: For generating word clouds.
- **Matplotlib**: For plotting graphs and visualizations.

## 🎉 Usage

1. Launch the application by navigating to the directory where `app.py` is located and executing the Streamlit command.
2. Use the file uploader to upload your XML files.
3. After uploading, the application will display the Elbow plot for selecting the number of clusters.
4. Input the desired number of clusters and click "Generate Word Clouds" to visualize the clusters.

## 🛠️ Known Issues

Currently, there are some limitations with the XML file processing:

- The application does not support running on a single XML file. You must upload multiple files for the application to function correctly. I am actively working on resolving this issue to enhance the user experience.

If you encounter any other issues or have suggestions, please feel free to create an issue or submit a pull request.


## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to create an issue or submit a pull request.

