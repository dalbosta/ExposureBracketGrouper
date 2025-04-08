import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE


def plot_clusters(clusters, max_images_per_cluster=5):
    """
    Visualizes the clustering results by showing a few images from each cluster
    """
    n_clusters = len(clusters)
    fig = plt.figure(figsize=(15, n_clusters * 3))

    for i, (cluster_id, image_paths) in enumerate(clusters.items()):
        # Only show up to max_images_per_cluster per cluster
        paths_to_show = image_paths[:max_images_per_cluster]
        n_images = len(paths_to_show)

        for j, path in enumerate(paths_to_show):
            ax = fig.add_subplot(n_clusters, max_images_per_cluster, i * max_images_per_cluster + j + 1)
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(f"Cluster {cluster_id}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_timing_comparison(model_timing_results):
    """
    Creates a bar chart comparing the runtime of different models or approaches
    """
    models = list(model_timing_results.keys())

    # Prepare data for the bar chart
    cnn_times = [model_timing_results[m]['cnn_feature_extraction'] for m in models]
    dbscan_times = [model_timing_results[m]['dbscan_clustering'] for m in models]
    total_times = [model_timing_results[m]['total'] for m in models]

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, cnn_times, width, label='CNN Feature Extraction')
    rects2 = ax.bar(x, dbscan_times, width, label='DBSCAN Clustering')
    rects3 = ax.bar(x + width, total_times, width, label='Total Pipeline')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()

    # Create a detailed table of timing results
    timing_data = {
        'Method': models,
        'CNN Feature Extraction (s)': [model_timing_results[m]['cnn_feature_extraction'] for m in models],
        'DBSCAN Clustering (s)': [model_timing_results[m]['dbscan_clustering'] for m in models],
        'Model Loading (s)': [model_timing_results[m]['model_loading'] for m in models],
        'Data Loading (s)': [model_timing_results[m]['data_loading'] for m in models],
        'Total Runtime (s)': [model_timing_results[m]['total'] for m in models]
    }

    timing_df = pd.DataFrame(timing_data)
    return timing_df


def visualize_embeddings(embeddings, labels, paths, method='tsne'):
    """
    Visualize the embeddings using dimensionality reduction

    Args:
        embeddings: Feature embeddings
        labels: Cluster labels
        paths: Image paths
        method: Dimensionality reduction method ('tsne', 'pca')
    """
    # Reduce dimensionality to 2D for visualization
    if method == 'tsne':
        print("Computing t-SNE embedding...")
        embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        from sklearn.decomposition import PCA
        print("Computing PCA embedding...")
        embeddings_2d = PCA(n_components=2).fit_transform(embeddings)

    # Get unique labels (excluding noise points with label -1)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    # Create a color map
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    n_clusters = len(unique_labels)
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    cmap = ListedColormap(colors)

    # Create a scatter plot
    plt.figure(figsize=(12, 10))

    # Plot noise points (label -1) in black
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                    c='black', s=50, label='Noise')

    # Plot clusters with different colors
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[colors[i]], s=50, label=f'Cluster {label}')

    plt.legend(loc='upper right')
    plt.title(f'Visualization of image clusters using {method.upper()}')
    plt.tight_layout()
    plt.show()

    # Return the 2D embeddings for further analysis
    return embeddings_2d


def print_cluster_summary(clusters):
    """
    Print a summary of clusters and their sizes
    """
    print(f"Number of clusters found: {len(clusters)}")
    for cluster_id, paths in clusters.items():
        print(f"  Cluster {cluster_id}: {len(paths)} images")