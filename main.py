# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the Data
youtube_data = pd.read_csv('Global YouTube Statistics.csv')

# 2. Data Preprocessing
# Extract the relevant columns for clustering: "video views" and "uploads"
clustering_data = youtube_data[["video views", "uploads"]]

# Scale the data to ensure both features have equal importance in clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# 3. Elbow Test to Determine Optimal Number of Clusters
sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(scaled_data)
    sum_of_squared_distances.append(km.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# 4. Apply k-means clustering with Optimal Clusters (in this case, 4 was chosen based on the elbow method)
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

# 5. Visualize the Clusters
clustering_data['Cluster'] = clusters

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'yellow']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

for i in range(4):
    plt.scatter(clustering_data['video views'][clustering_data.Cluster == i],
                clustering_data['uploads'][clustering_data.Cluster == i], 
                color=colors[i], 
                label=labels[i])

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='black', marker='X', label='Centroids')

plt.title('Clusters of YouTube Channels based on Views and Uploads')
plt.xlabel('Video Views')
plt.ylabel('Uploads')
plt.legend()
plt.grid(True)
plt.show()
