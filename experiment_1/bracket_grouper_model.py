import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


class RealEstateDataset(Dataset):
    """
    Dataset that loads images from a folder and applies transform if necessary.
    """

    def __init__(self, folder_path, transform=None):
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')
        self.image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path


def extract_features(model, dataloader, device='cpu', use_enhanced_features=False):
    """
    Runs all images through the given model in batches and returns:
      - embeddings (NumPy array of shape [N, feat_dim])
      - corresponding list of file paths

    If use_enhanced_features is True, combines CNN features with histogram features
    for better exposure-invariant clustering.
    """
    all_embeddings = []
    all_paths = []

    if use_enhanced_features:
        histogram_features = []

    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            # Convert from Tensor -> NumPy
            outputs = outputs.cpu().numpy()
            all_embeddings.append(outputs)

            # Extract histogram features if enhanced mode is enabled
            if use_enhanced_features:
                for img in batch_imgs:
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    hist_features = []
                    # Extract histogram for each channel
                    for channel in range(3):
                        hist, _ = np.histogram(img_np[:, :, channel], bins=32, range=(0, 1))
                        hist_features.extend(hist / np.sum(hist))  # Normalize
                    histogram_features.append(hist_features)

            all_paths.extend(batch_paths)

    all_embeddings = np.vstack(all_embeddings)  # shape [N, embedding_dim]

    if use_enhanced_features:
        # Combine CNN and histogram features
        hist_features = np.array(histogram_features)

        # Normalize features using StandardScaler
        scaler_cnn = StandardScaler().fit(all_embeddings)
        scaler_hist = StandardScaler().fit(hist_features)

        # Combine with weighting (80% CNN, 20% histogram)
        combined_features = np.hstack([
            scaler_cnn.transform(all_embeddings) * 0.8,
            scaler_hist.transform(hist_features) * 0.2
        ])
        return combined_features, all_paths

    return all_embeddings, all_paths


def cluster_embeddings(embeddings, paths, eps=0.5, min_samples=1, use_two_stage=False, eps2=None):
    """
    Clusters image embeddings using DBSCAN. Returns a dictionary:
      cluster_dict = { cluster_label: [image_paths] }

    If use_two_stage is True, implements a two-stage clustering approach with
    a second refinement step using eps2 as the second epsilon value.
    """
    if use_two_stage and eps2 is None:
        eps2 = eps / 2  # Default second stage eps is half of the first

    if use_two_stage:
        # First stage - broader clusters
        dbscan1 = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels1 = dbscan1.fit_predict(embeddings)

        # Second stage - refine each cluster
        final_labels = np.copy(labels1)
        next_label = np.max(labels1) + 1

        for cluster_id in np.unique(labels1):
            if cluster_id == -1:
                continue

            # Get indices for this cluster
            cluster_indices = np.where(labels1 == cluster_id)[0]
            if len(cluster_indices) <= 1:
                continue

            # Get embeddings for just this cluster
            cluster_embeddings = embeddings[cluster_indices]

            # Run DBSCAN again with tighter parameters
            dbscan2 = DBSCAN(eps=eps2, min_samples=min_samples)
            sub_labels = dbscan2.fit_predict(cluster_embeddings)

            # If any subclusters were found
            if np.any(sub_labels != -1):
                # Map new subcluster labels to unique IDs
                unique_sublabels = np.unique(sub_labels)
                for sub_id in unique_sublabels:
                    if sub_id == -1:
                        continue
                    # Get indices of this subcluster
                    subcluster_indices = np.where(sub_labels == sub_id)[0]
                    # Assign a new unique label
                    for idx in subcluster_indices:
                        final_labels[cluster_indices[idx]] = next_label
                    next_label += 1

        labels = final_labels
    else:
        # Standard single-stage DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(embeddings)

    groupings = defaultdict(list)
    for lbl, path in zip(labels, paths):
        groupings[lbl].append(path)

    # Convert DBSCAN labels (which can be -1) to a nice 1-based index
    final_groups = {}
    idx = 1
    for cluster_label in sorted(groupings.keys()):
        if cluster_label == -1:
            # -1 means 'noise'; skip or handle as you wish.
            continue
        final_groups[idx] = groupings[cluster_label]
        idx += 1

    return final_groups, labels


def create_edge_enhanced_transform(transform):
    """
    Modifies a transform pipeline to apply edge enhancement.
    This increases the focus on structural elements rather than colors/exposure.

    Args:
        transform: The original transform pipeline

    Returns:
        A new transform pipeline with edge enhancement
    """

    def edge_enhance(img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)

        # Convert to grayscale for edge detection
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()

        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Combine edges
        edges = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalize edges to 0-255
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert back to RGB and boost the edges
        if len(img_np.shape) == 3:
            edge_boosted = img_np.copy()
            # Reduce color intensity and boost edge information
            edge_boosted = (edge_boosted * 0.7).astype(np.uint8)
            edge_channel = np.stack([edges] * 3, axis=2)
            edge_boosted = cv2.addWeighted(edge_boosted, 0.7, edge_channel, 0.3, 0)
        else:
            edge_boosted = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)

        # Convert back to PIL Image
        return Image.fromarray(edge_boosted)

    # Create a new transform that applies edge enhancement
    from torchvision import transforms
    return transforms.Compose([
        transforms.Lambda(edge_enhance),
        transform
    ])


def create_perspective_invariant_transform(transform):
    """
    Modifies a transform pipeline to make it more robust to small perspective changes.

    Args:
        transform: The original transform pipeline

    Returns:
        A new transform pipeline with perspective augmentation
    """
    from torchvision import transforms

    # Extract relevant parts of the transform
    resize_crop_part = []
    normalize_part = []
    found_resize = False
    found_normalize = False

    for t in transform.transforms:
        if isinstance(t, (transforms.Resize, transforms.CenterCrop)) and not found_resize:
            resize_crop_part.append(t)
            if isinstance(t, transforms.CenterCrop):
                found_resize = True
        elif isinstance(t, transforms.Normalize):
            normalize_part.append(t)
            found_normalize = True

    # Create a new transform with perspective invariance
    perspective_transforms = []
    perspective_transforms.extend(resize_crop_part)

    # Add perspective augmentation
    perspective_transforms.extend([
        transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
        transforms.ToTensor(),
    ])

    if found_normalize:
        perspective_transforms.extend(normalize_part)

    return transforms.Compose(perspective_transforms)


def apply_histogram_equalization(img):
    """
    Apply histogram equalization to reduce the impact of exposure differences.

    Args:
        img: PIL Image

    Returns:
        PIL Image with histogram equalization applied
    """
    img_np = np.array(img)

    # Check if image is grayscale or color
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        # Grayscale image
        equalized = cv2.equalizeHist(img_np)
        return Image.fromarray(equalized)
    else:
        # Color image - convert to LAB and equalize L channel
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Equalize the L channel
        l_eq = cv2.equalizeHist(l)

        # Merge the channels back
        lab_eq = cv2.merge((l_eq, a, b))

        # Convert back to RGB
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        return Image.fromarray(rgb_eq)


def create_histogram_equalized_transform(transform):
    """
    Modifies a transform pipeline to apply histogram equalization.
    This helps normalize differences in exposure.

    Args:
        transform: The original transform pipeline

    Returns:
        A new transform pipeline with histogram equalization
    """
    from torchvision import transforms

    # Create a new transform that applies histogram equalization
    return transforms.Compose([
        transforms.Lambda(apply_histogram_equalization),
        transform
    ])