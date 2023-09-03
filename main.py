import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# Load the data
def load_data(filepath, encoding='ISO-8859-1'):
    return pd.read_csv(filepath, encoding=encoding)

# One-hot encode categorical columns
def encode_data(data, columns):
    encoder = OneHotEncoder(drop='first', sparse=False)
    return encoder.fit_transform(data[columns])

# Perform the elbow test for optimal k
def elbow_test(encoded_data, max_k=15):
    k_values = range(1, max_k)
    inertia_values = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(encoded_data)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Curve')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

# Main function
if __name__ == '__main__':
    data = load_data('Global_YouTube_Statistics.csv')
    encoded_data = encode_data(data, ['Youtuber', 'Title', 'Country'])
    elbow_test(encoded_data)
