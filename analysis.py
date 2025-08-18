import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imagenet import get_x_y_from_data_dict
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time
import warnings
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
import ot
import colorsys
import os
from scipy.interpolate import interp1d


def get_features_gpu(dataloader, model, args):
    features = []
    labels = []
    device = args.device
    # For embedding extraction
    hook_outputs = []
    def hook(model, input, output):
        hook_outputs.append(output)
    hook_handle = model.avgpool.register_forward_hook(hook)
    
    with torch.no_grad():
        for batch in dataloader:
            if args.imagenet_arch:
                images, targets = get_x_y_from_data_dict(batch, device)
            else:
                images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass
            _ = model(images)
            # Get embeddings from hook
            activations = hook_outputs[-1].reshape(len(images), -1)
            features.append(activations.detach().cpu())  # Move to CPU immediately
            hook_outputs.clear()
            # Store labels
            labels.append(targets.cpu())
    
    hook_handle.remove()
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def compute_variance_entanglement(retain_embeddings, forget_embeddings, 
                                  retain_labels=None, forget_labels=None, class_wise=False):
    # Normalize embeddings
    retain_embeddings = F.normalize(retain_embeddings, p=2, dim=1)
    forget_embeddings = F.normalize(forget_embeddings, p=2, dim=1)
    min_val = 0
    max_val = 1300
    if not class_wise:
        # Global entanglement (original implementation)
        mu_R = retain_embeddings.mean(dim=0)
        mu_S = forget_embeddings.mean(dim=0)
        global_embeddings = torch.cat([retain_embeddings, forget_embeddings], dim=0)
        mu = global_embeddings.mean(dim=0)
        # Intra-set variance
        retain_var = ((retain_embeddings - mu_R)**2).sum(dim=1).mean()
        forget_var = ((forget_embeddings - mu_S)**2).sum(dim=1).mean()
        numerator = retain_var + forget_var
        # Inter-set variance
        denom = 0.5 * (torch.sum((mu_R - mu)**2) + torch.sum((mu_S - mu)**2))
        # Avoid division by zero
        if denom.item() == 0:
            return 0.0
        raw_score = numerator.item() / denom.item()
        # Normalize to [0,1]
        normalized_score = (raw_score - min_val) / (max_val - min_val)
        normalized_score = min(1.0, max(0.0, normalized_score))  # Clip to [0,1]
    else:
        # Class-wise entanglement
        if retain_labels is None or forget_labels is None:
            raise ValueError("Class labels must be provided for class-wise entanglement")
        # Get unique classes present in both sets
        retain_classes = set(retain_labels.tolist())
        forget_classes = set(forget_labels.tolist())
        common_classes = retain_classes.intersection(forget_classes)
        if not common_classes:
            return 0.0  # No common classes
        
        class_scores = []
        class_weights = []
        for cls in common_classes:
            # Get class-specific embeddings
            retain_cls_mask = retain_labels == cls
            forget_cls_mask = forget_labels == cls
            retain_cls_embeddings = retain_embeddings[retain_cls_mask]
            forget_cls_embeddings = forget_embeddings[forget_cls_mask]
            # Skip if either set has too few samples
            if len(retain_cls_embeddings) == 0 or len(forget_cls_embeddings) == 0:
                continue
            # Compute class-specific means
            mu_R_cls = retain_cls_embeddings.mean(dim=0)
            mu_S_cls = forget_cls_embeddings.mean(dim=0)
            # Combine and compute global class mean
            cls_embeddings = torch.cat([retain_cls_embeddings, forget_cls_embeddings], dim=0)
            mu_cls = cls_embeddings.mean(dim=0)
            # Intra-class variance
            retain_cls_var = ((retain_cls_embeddings - mu_R_cls)**2).sum(dim=1).mean()
            forget_cls_var = ((forget_cls_embeddings - mu_S_cls)**2).sum(dim=1).mean()
            cls_numerator = retain_cls_var + forget_cls_var
            # Inter-class variance
            cls_denom = 0.5 * (torch.sum((mu_R_cls - mu_cls)**2) + torch.sum((mu_S_cls - mu_cls)**2))
            # Avoid division by zero
            if cls_denom.item() == 0:
                continue
            cls_score = cls_numerator.item() / cls_denom.item()
            cls_weight = len(cls_embeddings)
            class_scores.append(cls_score)
            class_weights.append(cls_weight)
        if not class_scores:
            return 0.0  # No valid classes
        # Compute weighted average
        total_weight = sum(class_weights)
        weighted_score = sum(s * w for s, w in zip(class_scores, class_weights)) / total_weight
        # Normalize the weighted average
        normalized_score = (weighted_score - min_val) / (max_val - min_val)
        normalized_score = min(1.0, max(0.0, normalized_score))
    return normalized_score
    
def compute_silhouette_entanglement(retain_embeddings, forget_embeddings, 
                                    retain_labels=None, forget_labels=None, class_wise=False):
    """
    Silhouette-based entanglement measuring cluster separation quality.
    Lower values indicate better separation (lower entanglement).
    """
    try:
        from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
        HAS_CUML = True
        print("Using GPU-accelerated silhouette score")
    except ImportError:
        from sklearn.metrics import silhouette_score
        HAS_CUML = False
        print("Using CPU silhouette score")
    # Normalize embeddings
    retain_embeddings = F.normalize(retain_embeddings, p=2, dim=1)
    forget_embeddings = F.normalize(forget_embeddings, p=2, dim=1)
    # Convert to numpy arrays
    if isinstance(retain_embeddings, torch.Tensor):
        retain_embeddings = retain_embeddings.cpu().numpy()
    if isinstance(forget_embeddings, torch.Tensor):
        forget_embeddings = forget_embeddings.cpu().numpy()
    if retain_labels is not None and isinstance(retain_labels, torch.Tensor):
        retain_labels = retain_labels.cpu().numpy()
    if forget_labels is not None and isinstance(forget_labels, torch.Tensor):
        forget_labels = forget_labels.cpu().numpy()
    # Check minimum samples requirement
    min_samples_per_group = 2  # Silhouette requires at least 2 samples per group

    if not class_wise:
        # Global entanglement computation
        if len(retain_embeddings) < min_samples_per_group or len(forget_embeddings) < min_samples_per_group:
            print(f"Warning: Not enough samples for silhouette (need at least {min_samples_per_group} per group)")
            raise ValueError
        
        try:
            # Combine embeddings and create labels (0 for retain, 1 for forget)
            X = np.vstack((retain_embeddings, forget_embeddings))
            y = np.concatenate([np.zeros(len(retain_embeddings)), np.ones(len(forget_embeddings))])
            if HAS_CUML:
                # Convert to float32 for cuML
                X = X.astype(np.float32)
                score = cu_silhouette_score(X, y, metric='cosine')
            else:
                score = silhouette_score(X, y, metric='cosine')
            # Convert to entanglement score (inverted and scaled to 0-1)
            # 1 - (score + 1)/2 => transforms from [-1,1] to [0,1] with lower being better separation
            entanglement = 1.0 - (score + 1.0) / 2.0
            return entanglement 
        except Exception as e:
            print(f"Silhouette computation error: {e}")
            raise ValueError
    else:
        # Class-wise entanglement
        if retain_labels is None or forget_labels is None:
            raise ValueError("Class labels must be provided for class-wise entanglement")
        
        # Get unique classes present in both sets
        retain_classes = set(retain_labels.tolist())
        forget_classes = set(forget_labels.tolist())
        common_classes = retain_classes.intersection(forget_classes)
        if not common_classes:
            return 0.0  # No common classes
        class_scores = []
        class_weights = []
        for cls in common_classes:
            # Get class-specific embeddings
            retain_cls_mask = retain_labels == cls
            forget_cls_mask = forget_labels == cls
            retain_cls_embeddings = retain_embeddings[retain_cls_mask]
            forget_cls_embeddings = forget_embeddings[forget_cls_mask]
            # Skip if either set has too few samples
            if len(retain_cls_embeddings) < min_samples_per_group or len(forget_cls_embeddings) < min_samples_per_group:
                continue
            try:
                # Combine embeddings and create labels for this class
                cls_X = np.vstack((retain_cls_embeddings, forget_cls_embeddings))
                cls_y = np.concatenate([np.zeros(len(retain_cls_embeddings)), np.ones(len(forget_cls_embeddings))])
                if HAS_CUML:
                    # Convert to float32 for cuML
                    cls_X = cls_X.astype(np.float32)
                    cls_score = cu_silhouette_score(cls_X, cls_y, metric='cosine')
                else:
                    cls_score = silhouette_score(cls_X, cls_y, metric='cosine')
                # Convert to entanglement score
                cls_entanglement = 1.0 - (cls_score + 1.0) / 2.0
                class_scores.append(cls_entanglement)
                class_weights.append(len(cls_X))
            except Exception as e:
                print(f"Class {cls} silhouette computation error: {e}")
                continue
        
        if not class_scores:
            return 0.0  # No valid classes
        # Compute weighted average
        total_weight = sum(class_weights)
        weighted_score = sum(s * w for s, w in zip(class_scores, class_weights)) / total_weight
        return weighted_score
    
def compute_wasserstein_entanglement(retain_embeddings, forget_embeddings, 
                                     retain_labels=None, forget_labels=None, class_wise=False):
    """
    Wasserstein-based entanglement using optimal transport distance.
    Lower values indicate lower entanglement (better separation).
    """
    # Normalize embeddings
    retain_embeddings = F.normalize(retain_embeddings, p=2, dim=1)
    forget_embeddings = F.normalize(forget_embeddings, p=2, dim=1)
    # Convert to numpy arrays
    if isinstance(retain_embeddings, torch.Tensor):
        retain_embeddings = retain_embeddings.cpu().numpy()
    if isinstance(forget_embeddings, torch.Tensor):
        forget_embeddings = forget_embeddings.cpu().numpy()
    if retain_labels is not None and isinstance(retain_labels, torch.Tensor):
        retain_labels = retain_labels.cpu().numpy()
    if forget_labels is not None and isinstance(forget_labels, torch.Tensor):
        forget_labels = forget_labels.cpu().numpy()
        
    def compute_single_wasserstein(r_embed, f_embed):
        """Helper function to compute Wasserstein distance for a single group"""
        # Cap samples to avoid memory issues with large datasets
        max_samples = 1000
        r_samples = min(len(r_embed), max_samples)
        f_samples = min(len(f_embed), max_samples)
        if len(r_embed) > max_samples:
            r_indices = np.random.choice(len(r_embed), r_samples, replace=False)
            r_subset = r_embed[r_indices]
        else:
            r_subset = r_embed
        if len(f_embed) > max_samples:
            f_indices = np.random.choice(len(f_embed), f_samples, replace=False)
            f_subset = f_embed[f_indices]
        else:
            f_subset = f_embed
        
        # Uniform weights for samples
        a = np.ones(r_samples) / r_samples
        b = np.ones(f_samples) / f_samples
        # Compute cost matrix (cosine distance)
        M = ot.dist(r_subset, f_subset, metric='euclidean')
        M /= M.max()  # Normalize costs
        # Compute optimal transport distance
        ot_distance = ot.emd2(a, b, M)
        entanglement = max(0.0, 1.0 - ot_distance)
        return entanglement
    
    if not class_wise:
        # Global entanglement
        try:
            entanglement = compute_single_wasserstein(retain_embeddings, forget_embeddings)
            return entanglement
        except Exception as e:
            print(f"Wasserstein computation error: {e}")
            raise ValueError
    else:
        # Class-wise entanglement
        if retain_labels is None or forget_labels is None:
            raise ValueError("Class labels must be provided for class-wise entanglement")
        # Get unique classes present in both sets
        retain_classes = set(retain_labels.tolist())
        forget_classes = set(forget_labels.tolist())
        common_classes = retain_classes.intersection(forget_classes)
        if not common_classes:
            return 0.0  # No common classes
    
        class_scores = []
        class_weights = []
        for cls in common_classes:
            # Get class-specific embeddings
            retain_cls_mask = retain_labels == cls
            forget_cls_mask = forget_labels == cls
            retain_cls_embeddings = retain_embeddings[retain_cls_mask]
            forget_cls_embeddings = forget_embeddings[forget_cls_mask]
            # Skip if either set has too few samples
            if len(retain_cls_embeddings) < 2 or len(forget_cls_embeddings) < 2:
                continue
            try:
                # Compute Wasserstein distance for this class
                cls_entanglement = compute_single_wasserstein(retain_cls_embeddings, forget_cls_embeddings)
                class_scores.append(cls_entanglement)
                class_weights.append(len(retain_cls_embeddings) + len(forget_cls_embeddings))
            except Exception as e:
                print(f"Class {cls} Wasserstein computation error: {e}")
                continue
        
        if not class_scores:
            return 0.0  # No valid classes
        # Compute weighted average
        total_weight = sum(class_weights)
        weighted_score = sum(s * w for s, w in zip(class_scores, class_weights)) / total_weight
        return weighted_score
    
def compute_svm_entanglement(retain_embeddings, forget_embeddings, 
                             retain_labels=None, forget_labels=None, class_wise=False):
    """
    SVM-based entanglement measuring prediction errors.
    Higher values indicate higher entanglement (harder to separate).
    """
    try:
        from cuml.svm import SVC as cuSVC
        from cuml.model_selection import train_test_split
        HAS_CUML = True
        print("Using GPU-accelerated SVM")
    except ImportError:
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        HAS_CUML = False
        print("Using CPU SVM")
    # Normalize embeddings
    retain_embeddings = F.normalize(retain_embeddings, p=2, dim=1)
    forget_embeddings = F.normalize(forget_embeddings, p=2, dim=1)
    # Convert to numpy arrays
    if isinstance(retain_embeddings, torch.Tensor):
        retain_embeddings = retain_embeddings.cpu().numpy()
    if isinstance(forget_embeddings, torch.Tensor):
        forget_embeddings = forget_embeddings.cpu().numpy()
    if retain_labels is not None and isinstance(retain_labels, torch.Tensor):
        retain_labels = retain_labels.cpu().numpy()
    if forget_labels is not None and isinstance(forget_labels, torch.Tensor):
        forget_labels = forget_labels.cpu().numpy()
    
    def compute_single_svm(r_embed, f_embed):
        """Helper function to compute SVM separability for a single group"""
        # Handle edge cases
        if len(r_embed) < 5 or len(f_embed) < 5:
            print("Warning: Not enough samples for reliable SVM (need at least 5 per group)")
            raise ValueError
        # Combine embeddings and create labels (0 for retain, 1 for forget)
        X = np.vstack((r_embed, f_embed))
        y = np.concatenate([np.zeros(len(r_embed)), np.ones(len(f_embed))])
        if HAS_CUML:
            try:
                # Convert to float32 for cuML
                X = X.astype(np.float32)
                y = y.astype(np.float32)
                # Split data for validation (cuML doesn't have cross_val_score)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                # Train SVM
                clf = cuSVC(kernel='linear', C=1.0)
                clf.fit(X_train, y_train)
                # Evaluate accuracy
                accuracy = clf.score(X_test, y_test)
            except Exception as e:
                print(f"GPU SVM error: {e}. Falling back to CPU.")
                # Fallback to CPU implementation
                clf = LinearSVC(random_state=42, C=0.01)
                cv_scores = cross_val_score(clf, X, y, cv=min(5, min(np.bincount(y).astype(int))), scoring='accuracy')
                accuracy = np.mean(cv_scores)
        else:
            # CPU implementation with scikit-learn
            clf = LinearSVC(random_state=42, C=0.01)
            cv_scores = cross_val_score(clf, X, y, cv=min(5, min(np.bincount(y).astype(int))), scoring='accuracy')
            accuracy = np.mean(cv_scores)
        entanglement = 1.0 - accuracy
        return entanglement
    
    if not class_wise:
        # Global entanglement
        try:
            entanglement = compute_single_svm(retain_embeddings, forget_embeddings)
            return entanglement
        except Exception as e:
            print(f"SVM computation error: {e}")
            raise ValueError  # Return neutral value on error
    else:
        # Class-wise entanglement
        if retain_labels is None or forget_labels is None:
            raise ValueError("Class labels must be provided for class-wise entanglement")
        # Get unique classes present in both sets
        retain_classes = set(retain_labels.tolist())
        forget_classes = set(forget_labels.tolist())
        common_classes = retain_classes.intersection(forget_classes)
        if not common_classes:
            return 0.0  # No common classes
        
        class_scores = []
        class_weights = []
        for cls in common_classes:
            # Get class-specific embeddings
            retain_cls_mask = retain_labels == cls
            forget_cls_mask = forget_labels == cls
            retain_cls_embeddings = retain_embeddings[retain_cls_mask]
            forget_cls_embeddings = forget_embeddings[forget_cls_mask]
            # Skip if either set has too few samples
            if len(retain_cls_embeddings) < 5 or len(forget_cls_embeddings) < 5:
                continue
            try:
                # Compute SVM separability for this class
                cls_entanglement = compute_single_svm(retain_cls_embeddings, forget_cls_embeddings)
                class_scores.append(cls_entanglement)
                class_weights.append(len(retain_cls_embeddings) + len(forget_cls_embeddings))
            except Exception as e:
                print(f"Class {cls} SVM computation error: {e}")
                continue
        
        if not class_scores:
            return 0.0  # No valid classes
        # Compute weighted average
        total_weight = sum(class_weights)
        weighted_score = sum(s * w for s, w in zip(class_scores, class_weights)) / total_weight
        return weighted_score
    
def compute_knn_entanglement(retain_embeddings, forget_embeddings, 
                            retain_labels=None, forget_labels=None, 
                            class_wise=False, k=10):
    """
    KNN-based entanglement measuring local neighborhood mixing.
    Higher values indicate higher entanglement (more mixed neighborhoods).
    
    Args:
        retain_embeddings: Feature embeddings for retain set
        forget_embeddings: Feature embeddings for forget set
        retain_labels: Class labels for retain set (only needed for class_wise=True)
        forget_labels: Class labels for forget set (only needed for class_wise=True)
        class_wise: If True, computes class-wise entanglement
        k: Number of neighbors to consider (default: 5)
        
    Returns:
        Float: Entanglement score (higher means more entangled, range 0-1)
    """
    try:
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
        HAS_CUML = True
        print("Using GPU-accelerated KNN")
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        HAS_CUML = False
        print("Using CPU KNN")
    
    # Normalize embeddings
    retain_embeddings = F.normalize(retain_embeddings, p=2, dim=1)
    forget_embeddings = F.normalize(forget_embeddings, p=2, dim=1)
    
    # Convert to numpy arrays
    if isinstance(retain_embeddings, torch.Tensor):
        retain_embeddings = retain_embeddings.cpu().numpy()
    if isinstance(forget_embeddings, torch.Tensor):
        forget_embeddings = forget_embeddings.cpu().numpy()
    if retain_labels is not None and isinstance(retain_labels, torch.Tensor):
        retain_labels = retain_labels.cpu().numpy()
    if forget_labels is not None and isinstance(forget_labels, torch.Tensor):
        forget_labels = forget_labels.cpu().numpy()
    
    def compute_single_knn(r_embed, f_embed):
        """Helper function to compute KNN separability for a single group"""
        # Handle edge cases
        if len(r_embed) < k + 1 or len(f_embed) < k + 1:
            print(f"Warning: Not enough samples for reliable KNN with k={k}")
            return 0.5
        # Combine embeddings
        X = np.vstack((r_embed, f_embed))
        # Labels: 0 for retain, 1 for forget
        y = np.concatenate([np.zeros(len(r_embed)), np.ones(len(f_embed))])
        # Initialize KNN
        if HAS_CUML:
            nn = cuNearestNeighbors(n_neighbors=k+1, metric='cosine')
            X = X.astype(np.float32)  # cuML requires float32
        else:
            nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
            
        nn.fit(X)
        # Find k+1 nearest neighbors (including self)
        distances, indices = nn.kneighbors(X)
        # Skip the first neighbor (self) and use the next k
        neighbor_indices = indices[:, 1:k+1]
        # Get the labels of neighbors
        neighbor_labels = y[neighbor_indices]
        # Predict label based on majority vote of neighbors
        predictions = np.mean(neighbor_labels, axis=1) > 0.5
        predictions = predictions.astype(int)
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        # Entanglement = 1 - accuracy
        # Higher entanglement means lower prediction accuracy
        entanglement = 1.0 - accuracy
        return entanglement
    
    if not class_wise:
        try:
            entanglement = compute_single_knn(retain_embeddings, forget_embeddings)
            return entanglement
        except Exception as e:
            print(f"KNN computation error: {e}")
            raise ValueError
    else:
        # Class-wise entanglement
        if retain_labels is None or forget_labels is None:
            raise ValueError("Class labels must be provided for class-wise entanglement")
        # Get unique classes present in both sets
        retain_classes = set(retain_labels.tolist())
        forget_classes = set(forget_labels.tolist())
        common_classes = retain_classes.intersection(forget_classes)
        if not common_classes:
            return 0.0  # No common classes
        
        class_scores = []
        class_weights = []
        for cls in common_classes:
            # Get class-specific embeddings
            retain_cls_mask = retain_labels == cls
            forget_cls_mask = forget_labels == cls
            retain_cls_embeddings = retain_embeddings[retain_cls_mask]
            forget_cls_embeddings = forget_embeddings[forget_cls_mask]
            # Skip if either set has too few samples
            if len(retain_cls_embeddings) < k+1 or len(forget_cls_embeddings) < k+1:
                continue
            try:
                # Compute KNN entanglement for this class
                cls_entanglement = compute_single_knn(
                    retain_cls_embeddings, forget_cls_embeddings
                )
                class_scores.append(cls_entanglement)
                class_weights.append(len(retain_cls_embeddings) + len(forget_cls_embeddings))
            except Exception as e:
                print(f"Class {cls} KNN computation error: {e}")
                continue
        
        if not class_scores:
            raise ValueError
        
        # Compute weighted average
        total_weight = sum(class_weights)
        weighted_score = sum(s * w for s, w in zip(class_scores, class_weights)) / total_weight
        return weighted_score

def interpolate_colors(base_colors, num_colors):
    """
    Interpolate between base colors to create a gradient of colors.
    Base colors can be hex or RGB tuples.
    """
    # Convert hex colors to RGB if needed
    rgb_colors = []
    for color in base_colors:
        if isinstance(color, str) and color.startswith('#'):
            rgb_colors.append(mcolors.hex2color(color))
        else:
            rgb_colors.append(color)
    # Convert to numpy array for easier manipulation
    rgb_colors = np.array(rgb_colors)
    # Create a parameter for interpolation (evenly spaced points)
    t_base = np.linspace(0, 1, len(rgb_colors))
    # Create interpolation functions for each RGB channel
    r_interp = interp1d(t_base, rgb_colors[:, 0], kind='cubic')
    g_interp = interp1d(t_base, rgb_colors[:, 1], kind='cubic')
    b_interp = interp1d(t_base, rgb_colors[:, 2], kind='cubic')
    # Generate new colors
    t_new = np.linspace(0, 1, num_colors)
    r_new = np.clip(r_interp(t_new), 0, 1)
    g_new = np.clip(g_interp(t_new), 0, 1)
    b_new = np.clip(b_interp(t_new), 0, 1)
    # Combine channels
    interpolated_colors = np.column_stack((r_new, g_new, b_new))
    return interpolated_colors

def brighten_color(color, factor=0.3):
    """
    Make a color brighter by blending it with white.
    """
    # Convert hex to RGB if needed
    if isinstance(color, str) and color.startswith('#'):
        rgb = np.array(mcolors.hex2color(color))
    else:
        rgb = np.array(color)
    # Blend with white
    white = np.array([1, 1, 1])
    brighter = rgb + factor * (white - rgb)
    # Ensure values are in range
    brighter = np.clip(brighter, 0, 1)
    return tuple(brighter)

def plot_embeddings_direct(ax, retain_embeddings, forget_embeddings, retain_labels, forget_labels,
                           target_class=None, marker_size_scale=1.0, zoom_factor=1.0, title=None, modelname='0cifar100_resnet50_orig_bs256_lr0.1_seed2_epoch200'):
    """
    Direct plotting of embeddings to a given axes without creating a separate figure.
    This ensures proper marker styles are preserved.
    """
    from cuml.manifold import UMAP as cuUMAP
    # Convert to numpy if tensors
    if isinstance(retain_embeddings, torch.Tensor):
        retain_embeddings = retain_embeddings.cpu().numpy()
    if forget_embeddings is not None and isinstance(forget_embeddings, torch.Tensor):
        forget_embeddings = forget_embeddings.cpu().numpy()
    if retain_labels is not None and isinstance(retain_labels, torch.Tensor):
        retain_labels = retain_labels.cpu().numpy()
    if forget_labels is not None and isinstance(forget_labels, torch.Tensor):
        forget_labels = forget_labels.cpu().numpy()
    
    # If a target class is specified, filter to only include that class
    if target_class is not None:
        retain_mask = retain_labels == target_class
        retain_embeddings = retain_embeddings[retain_mask]
        retain_labels = retain_labels[retain_mask]
        
        if forget_embeddings is not None:
            forget_mask = forget_labels == target_class
            forget_embeddings = forget_embeddings[forget_mask]
            forget_labels = forget_labels[forget_mask]
    
    # If forget embeddings are provided, combine them
    if forget_embeddings is not None:
        combined_embeddings = np.vstack((retain_embeddings, forget_embeddings))
        combined_labels = np.concatenate((retain_labels, forget_labels)) if retain_labels is not None else None
        is_forget = np.concatenate([np.zeros(len(retain_embeddings), dtype=bool), 
                                   np.ones(len(forget_embeddings), dtype=bool)])
    else:
        combined_embeddings = retain_embeddings
        combined_labels = retain_labels
        is_forget = None
    
    # Standardize data
    scaler = StandardScaler()
    combined_embeddings_scaled = scaler.fit_transform(combined_embeddings)
    # Determine UMAP parameters based on zoom factor
    if target_class is not None:
        # Single-class parameters with zoom
        min_dist = 0.05 / zoom_factor
        spread = 1.0 / zoom_factor
        n_neighbors = max(5, int(15 / zoom_factor))
    else:
        # All-classes parameters with zoom
        min_dist = 0.1 / zoom_factor
        spread = 1.2 / zoom_factor
        n_neighbors = 40  # Keep this for global structure
    # Apply UMAP dimensionality reduction
    reducer = cuUMAP(n_components=2,
                   random_state=42,
                   n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   spread=spread,
                   metric='correlation')
    
    reduced_embeddings = reducer.fit_transform(combined_embeddings_scaled)
    # Calculate marker sizes (increased for better visibility)
    if target_class is not None:
        # Single class - larger markers
        base_retain_size = 64
    else:
        # All classes - moderately sized markers
        base_retain_size = 32
    # Forget markers are larger than retain
    base_forget_size = base_retain_size + 1
    # Apply scaling factor
    retain_size = base_retain_size * marker_size_scale
    forget_size = base_forget_size * marker_size_scale
    # For single class visualization
    if target_class is not None:
        # Custom colors for single class
        if '_orig_' in modelname:
            retain_color = mcolors.to_hex((78/255, 101/255, 155/255))
            forget_color = mcolors.to_hex((253/255, 207/255, 158/255))
        else:
            retain_color = '#2f2d54'
            forget_color = '#bd9aad'
        # Different markers for retain and forget
        retain_mask = ~is_forget
        forget_mask = is_forget
        # Plot retain points (circles)
        ax.scatter(
            reduced_embeddings[retain_mask, 0],
            reduced_embeddings[retain_mask, 1],
            color=retain_color,
            marker='o',
            s=retain_size,
            alpha=0.8,
            label="Retain"
        )
        # Plot forget points (stars)
        ax.scatter(
            reduced_embeddings[forget_mask, 0],
            reduced_embeddings[forget_mask, 1],
            color=forget_color,
            marker='X',
            s=forget_size,
            alpha=0.9,
            label="Forget"
        )
        # Add a legend
        ax.legend(fontsize=20)
    # For all classes visualization
    elif combined_labels is not None:
        unique_classes = np.unique(combined_labels)
        n_classes = len(unique_classes)
        # Count samples per class
        class_counts = {}
        for cls in unique_classes:
            class_counts[cls] = np.sum(combined_labels == cls)
        # Sort classes by count (highest first)
        sorted_classes = sorted(unique_classes, key=lambda c: class_counts[c], reverse=True)
        # Create color palette based on provided hex colors
        base_colors = [
            '#2f2d54',
            mcolors.to_hex((78/255, 101/255, 155/255)), 
            mcolors.to_hex((138/255, 140/255, 191/255)),
            mcolors.to_hex((184/255, 168/255, 207/255)),
            mcolors.to_hex((231/255, 188/255, 198/255)),
            mcolors.to_hex((253/255, 207/255, 158/255)),
            mcolors.to_hex((239/255, 164/255, 132/255)),
            mcolors.to_hex((182/255, 118/255, 108/255)),
            '#e5c8a0',
        ]
        # Interpolate to get enough colors
        all_colors = interpolate_colors(base_colors, n_classes)
        # Map colors to classes (largest classes get darker colors)
        class_to_color = {cls: all_colors[i] for i, cls in enumerate(sorted_classes)}
        # Plot each class
        for cls in unique_classes:
            mask = combined_labels == cls
            if is_forget is not None:
                # Split by retain/forget
                retain_mask = mask & ~is_forget
                forget_mask = mask & is_forget
                # Original color for retain points
                retain_color = class_to_color[cls]
                # Brighter color for forget points
                forget_color = brighten_color(retain_color, factor=0.3)
                # Plot retain points (circles)
                if np.any(retain_mask):
                    ax.scatter(
                        reduced_embeddings[retain_mask, 0],
                        reduced_embeddings[retain_mask, 1],
                        color=retain_color,
                        marker='o',
                        s=retain_size,
                        alpha=0.8,
                        edgecolors='none',
                    )
                # Plot forget points (stars) with brighter color
                if np.any(forget_mask):
                    ax.scatter(
                        reduced_embeddings[forget_mask, 0],
                        reduced_embeddings[forget_mask, 1],
                        color=forget_color,  # Brighter version
                        marker='X',
                        s=forget_size,
                        alpha=0.9,
                    )
        # Add a legend for retain/forget
        if is_forget is not None:
            marker_legend = [
                plt.Line2D([0], [0], marker='o', color='gray', linestyle='None', 
                          markersize=10, label='Retain'),
                plt.Line2D([0], [0], marker='X', color='gray', 
                          linestyle='None', markersize=11, label='Forget')
            ]
            ax.legend(handles=marker_legend, fontsize=20)
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=25)
    
    # Hide axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def compute_landscape_flatness_stats(alpha_grid, beta_grid, losses):
    # Find center point (original model)
    center_idx = losses.shape[0] // 2
    center_loss = losses[center_idx, center_idx]
    # Measure deviation from center loss
    deviations = np.abs(losses - center_loss)
    # 1. Variability: Standard deviation of loss (higher = more uneven landscape)
    variability = np.std(losses)
    # 2. Average absolute gradient magnitude (higher = steeper landscape)
    # This measures how quickly loss changes as parameters change
    dx, dy = np.gradient(losses)
    gradient_magnitude = np.mean(np.sqrt(dx**2 + dy**2))
    # 3. Basin Ratio: Using an adaptive threshold based on statistics
    # Points within 1 standard deviation of minimum
    threshold = 0.5*variability
    basin_ratio = np.mean(deviations <= threshold) * 100

    relative_thresholds = [0.1, 0.25, 0.5, 0.75]
    stability_profile = []
    for t in relative_thresholds:
        # Absolute threshold based on center loss
        abs_threshold = center_loss * t
        # Direct pointwise measurement
        ratio = np.mean(deviations <= abs_threshold) * 100
        stability_profile.append(ratio)
    return {
        "variability": variability, 
        "grad_norm": gradient_magnitude,
        "basin_ratio": basin_ratio,
        "stability_profile": stability_profile
    }


def visualize_loss_landscape_minimal(model, data_loader, loss_fn, alpha_range=(-1, 1), beta_range=(-1, 1), 
                                     resolution=25, random_state=42, fig_size=(8, 8), normalize=False):
    """
    Compute loss landscape with Magma colormap and zoomed-in view.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    # Generate two random directions
    directions = []
    for direction_idx in range(2):
        direction = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if direction_idx == 0:
                    direction[name] = torch.randn_like(param)
                else:
                    if name in directions[0]:
                        v1 = directions[0][name].view(-1)
                        v2 = torch.randn_like(param).view(-1)
                        v2 = v2 - torch.dot(v2, v1) / torch.dot(v1, v1) * v1
                        direction[name] = v2.reshape(param.shape)
                    else:
                        direction[name] = torch.randn_like(param)
        # Normalize direction
        norm = torch.sqrt(sum([torch.sum(d**2) for d in direction.values()]))
        for name in direction:
            direction[name] /= norm
        directions.append(direction)
    for d in range(len(directions)): # filter normalization
        for name, param in model.named_parameters():
            if name in directions[d]:
                # Get the corresponding parameter's filter norm
                param_norm = torch.norm(param.data)
                if param_norm > 0 and len(param.size()) > 1:
                    # Normalize the direction and scale by the parameter's norm
                    directions[d][name] = directions[d][name] * param_norm / (torch.norm(directions[d][name])+1e-10)
    # Generate grid of perturbation values
    alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    # Compute loss for each perturbation
    losses = np.zeros((resolution, resolution))
    
    model.eval()
    # Store original parameters as tensor clones
    original_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()
    with torch.no_grad():
        for i, alpha in enumerate(tqdm(alphas, desc="Computing loss landscape")):
            for j, beta in enumerate(betas):
                # Apply perturbation
                for name, param in model.named_parameters():
                    if param.requires_grad and name in directions[0] and name in directions[1]:
                        param.data = original_params[name] + alpha * directions[0][name] + beta * directions[1][name]
                # Compute loss
                total_loss = 0
                num_batches = 0
                for batch in data_loader:
                    inputs, targets = batch
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                
                losses[j, i] = total_loss / max(1, num_batches)
                # Later, to restore:
                for name, param in model.named_parameters():
                    if param.requires_grad and name in original_params:
                        param.data.copy_(original_params[name])
    
    # Normalize if requested
    # if normalize:
    #     loss_min, loss_max = np.min(losses), np.max(losses)
    #     losses = (losses - loss_min) / (loss_max - loss_min)
    # Apply light smoothing
    # losses = ndimage.gaussian_filter(losses, sigma=0.3)
    # Compute and print flatness metrics
    flatness_stats = compute_landscape_flatness_stats(alpha_grid, beta_grid, losses)
    print(f"Loss landscape flatness metrics:")
    print(f"  Variability: {flatness_stats['variability']:.4f} (lower = flatter)")
    print(f"  Gradient Norm: {flatness_stats['grad_norm']:.4f} (lower = flatter)")
    print(f"  Basin Ratio: {flatness_stats['basin_ratio']:.4f} (higher = flatter)")
    print(f"  Basin/Stability Profile: {flatness_stats['stability_profile']} (higher = flatter)")
    # Return the data with flatness stats
    return alpha_grid, beta_grid, losses, flatness_stats

def save_loss_landscape_from_multiple_angles(model, data_loader, loss_fn, filepath, model_name, 
                                           resolution=25, angles=None, zoom_factor=0.8, testset=False):
    """
    Compute loss landscape once and save views from multiple angles with Magma colormap.
    """
    # Default angles if none provided
    if angles is None:
        angles = [
            (15, 30),  # Low angle view
            (15, 60),  # Low angle, rotated
            (30, 45),  # Medium angle
        ]
    # Compute the loss landscape once
    print("Computing loss landscape (once)...")
    ab_range = (-1, 1) 
    alpha_grid, beta_grid, losses, flatness_stats = visualize_loss_landscape_minimal(
        model=model, 
        data_loader=data_loader,
        loss_fn=loss_fn,
        alpha_range=ab_range,
        beta_range=ab_range,
        resolution=resolution
    )
    # grab the original magma
    orig_cmap = cm.get_cmap("magma")
    # sample its RGBA values
    N = orig_cmap.N  # usually 256
    colors = orig_cmap(np.linspace(0,1,N))
    # set all alpha entries to something lighter (e.g. 0.6)
    colors[:,-1] = 0.9
    # build a new ListedColormap
    magma_lighter = mcolors.ListedColormap(colors)
    blend = 0.2   # 0 = no change, 1 = pure white
    for i in range(N):
        rgb = colors[i,:3]
        colors[i,:3] = rgb + (1 - rgb) * blend
    # (you can still tweak colors[:,-1] as above)
    magma_brighter = mcolors.ListedColormap(colors)
    
    # Create and save plots from multiple viewing angles
    print("Rendering and saving from multiple angles...")
    for elev, azim in angles:
        # Create a new figure for this angle
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface with magma colormap (darker colors for higher values)
        surf = ax.plot_surface(alpha_grid, beta_grid, losses, cmap=magma_brighter.reversed(), 
                             antialiased=True, edgecolor='none')
        # Set this specific view angle
        ax.view_init(elev=elev, azim=azim)
        basin_ratio = flatness_stats['basin_ratio']
        # Create text for legend
        legend_text = f"Basin: {basin_ratio:.2f}"
        
        # Add text box overlay
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        ax.text2D(0, 0.95, legend_text, transform=ax.transAxes, fontsize=48,
               verticalalignment='top')
        # Zoom in by adjusting the axis limits and distance
        # This minimizes empty space
        x_range = [-1, 1]
        y_range = [-1, 1]
        z_min, z_max = np.min(losses), np.max(losses)
        z_range = [z_min - 0.05*(z_max-z_min), z_max + 0.05*(z_max-z_min)]
        # Set tight axis limits
        ax.set_xlim3d(x_range[0] * zoom_factor, x_range[1] * zoom_factor)
        ax.set_ylim3d(y_range[0] * zoom_factor, y_range[1] * zoom_factor)
        ax.set_zlim3d(z_range[0], z_range[1])
        # Adjust camera distance to fill the frame
        ax.dist = 8 * zoom_factor  # Lower values = more zoomed in
        # Hide everything
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Save with angle info
        plt.tight_layout()
        if testset:
            fig.savefig(f"{filepath}/loss_landscape_{model_name}_test_elev{elev}_azim{azim}.pdf", bbox_inches='tight', format='pdf')
        else:
            fig.savefig(f"{filepath}/loss_landscape_{model_name}_elev{elev}_azim{azim}.pdf", bbox_inches='tight', format='pdf')
        plt.close(fig)
        
    print(f"Loss landscapes from {len(angles)} angles saved to {filepath}")
