# YouTube K-means Clustering Analysis 📊🎥

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)

Perform a K-means clustering analysis on a dataset of global YouTube statistics.

<p align="center">
  <img width="460" height="300" src="path_to_some_image_representing_youtube_or_data_analysis">
</p>

## 📌 Table of Contents
- [Data](#data)
- [Analysis](#analysis)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## 📂 Data

The dataset, `Global_YouTube_Statistics.csv`, contains various metrics and attributes related to popular YouTube channels.

## 📈 Analysis

1. **Data Loading**: The dataset is loaded using pandas.
2. **Data Transformation**: Transform selected categorical columns ('Youtuber', 'Title', 'Country') into a numerical format using one-hot encoding.
3. **Elbow Test**: Apply K-means clustering for a range of cluster numbers to find the optimal number using the elbow method.

## 🔧 Requirements

- pandas
- matplotlib
- scikit-learn

## 🚀 Usage

```bash
python youtube_clustering_analysis.py
