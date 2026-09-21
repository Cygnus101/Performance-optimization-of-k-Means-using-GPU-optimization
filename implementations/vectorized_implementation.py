from sklearn.datasets import fetch_openml
from pathlib import Path
import numpy as np

def load_fashion_mnist():

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "datasets"

    X, y = fetch_openml(
        "Fashion-MNIST",
        version=1,
        return_X_y=True,
        as_frame=False,
        data_home=cache_dir,
        cache=True
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    return X, y 

X, y = load_fashion_mnist()

def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4):

    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], num_clusters, replace=False)
    centroids = X[random_indices]     
    for iteration in range(num_iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(num_clusters)])
        
        if np.linalg.norm(new_centroids - centroids) < threshold:
            centroids = new_centroids
            break
        centroids = new_centroids

    return labels, centroids


kmeans_labels, kmeans_centroids = kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4)
print("K-means clustering completed.")