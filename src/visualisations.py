import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy import stats

import torch
import numpy as np

import Levenshtein


def prediction_visualization_figure(x_hat, x, charset_size, int_to_char):
    """
    Visualize model predictions compared to ground truth.
    Annotate the heatmap with characters using the int_to_char mapping.
    Returns the matplotlib figure for logging purposes.
    """
    # Convert one-hot encoded sequences to indices
    predictions = x_hat.argmax(-1)
    targets = x.argmax(-1)

    # Generate a custom colormap for the vocabulary size
    base_colors = sns.color_palette("tab20", n_colors=min(20, charset_size))
    if charset_size > len(base_colors):
        additional_colors = sns.color_palette("husl", n_colors=charset_size - len(base_colors))
        full_colors = base_colors + additional_colors
    else:
        full_colors = base_colors
    cmap = mcolors.ListedColormap(full_colors)

    # Select a few samples for visualization
    num_samples = min(4, x_hat.size(0))
    samples_to_show = range(num_samples)

    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))

    # If there's only one sample, `axes` is not iterable; wrap it in a list
    if num_samples == 1:
        axes = [axes]

    for i, ax in zip(samples_to_show, axes):
        # Create a combined array for the heatmap
        combined = torch.stack([predictions[i], targets[i]]).cpu().numpy()

        # Create a string annotation array using int_to_char mapping
        annotations = [[int_to_char[val] for val in row] for row in combined]

        sns.heatmap(
            combined,
            cmap=cmap,
            cbar=True,
            vmin=0,  # Set the minimum value for the colormap
            vmax=charset_size - 1,  # Set the maximum value for the colormap
            yticklabels=['Prediction', 'Ground Truth'],
            annot=annotations,  # Use the mapped characters as annotations
            fmt='',  # Disable formatting to show raw strings
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        ax.set_title(f'Sample {i} Prediction vs Ground Truth')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.close()

    return fig


def token_distribution_visualization_figure(x_hat, x, int_to_char):
    """
    Log token distribution visualization to TensorBoard

    Parameters:
    - x_hat: Predicted logits
    - x: Ground truth sequences
    - prefix: Logging prefix (train/val/test)
    - int_to_char: A dictionary mapping token indices to characters
    """
    # Convert logits to probabilities and aggregate over batch
    pred_probs = torch.softmax(x_hat, dim=-1)
    batch_pred_dist = pred_probs.mean(dim=0).detach().cpu().numpy()

    # Convert one-hot to indices for ground truth and aggregate
    ground_truth = x.argmax(dim=-1)
    charset_size = x.shape[-1]
    batch_gt_dist = torch.zeros((x.shape[1], charset_size), device=x.device)

    # Count occurrences of each token at each position
    for pos in range(x.shape[1]):
        counts = torch.bincount(ground_truth[:, pos], minlength=charset_size)
        batch_gt_dist[pos] = counts.float() / ground_truth.shape[0]

    batch_gt_dist = batch_gt_dist.cpu().numpy()

    # Prepare y-axis labels using int_to_char mapping
    y_labels_pred = [int_to_char.get(i, str(i)) for i in range(charset_size)]
    y_labels_gt = [int_to_char.get(i, str(i)) for i in range(charset_size)]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Batch-Level Token Distributions', fontsize=16)

    # Predictions subplot
    sns.heatmap(
        batch_pred_dist.T,
        cmap='YlGnBu',
        ax=ax1,
        cbar=True,
        xticklabels=range(batch_pred_dist.shape[0]),
        yticklabels=y_labels_pred,
        # annot=True, fmt='.2f',  # Add value annotations for clarity
        linewidths=0.5
    )
    ax1.set_title('Batch Averaged Predictions')
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Tokens')

    # Ground truth subplot
    sns.heatmap(
        batch_gt_dist.T,
        cmap='YlGnBu',
        ax=ax2,
        cbar=True,
        xticklabels=range(batch_gt_dist.shape[0]),
        yticklabels=y_labels_gt,
        # annot=True, fmt='.2f',  # Add value annotations for clarity
        linewidths=0.5
    )
    ax2.set_title('Batch Ground Truth Token Frequency')
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Tokens')

    plt.tight_layout()
    plt.close()

    return fig  # Return the figure for further use


def latent_space_visualization_figure(z_mean):
    """
    Generate and log visualization of the latent space
    Supports both PyTorch Lightning's built-in logging and wandb
    Returns the figure for external use
    """
    # PCA visualization of latent space
    z_mean_np = z_mean.detach().cpu().numpy()

    # Using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_mean_np)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7)
    ax.set_title(f'Latent Space Visualization')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')

    plt.close()

    # Return the figure for external use
    return fig


def distance_correlation_figure(z_mean, smiles_vectors, max_samples=16):
    """
    Generate multiple visualizations of the latent space and compute distance correlations.
    Limits the number of samples used for computation and visualization.

    Args:
    - z_mean (torch.Tensor): Latent space mean representations
    - smiles_vectors (np.ndarray): One-hot encoded SMILES vectors
    - max_samples (int): Maximum number of samples to use (default is 32)

    Returns:
    - fig (matplotlib.figure.Figure): The generated figure
    """

    # Limit the number of samples to max_samples
    num_samples = min(z_mean.shape[0], max_samples)

    # Convert z_mean to numpy for visualization (take only the selected samples)
    z_mean_np = z_mean.detach().cpu().numpy()[:num_samples]
    smiles_vectors = smiles_vectors.detach().cpu().numpy()[:num_samples]

    # Compute Levenshtein distances
    def compute_levenshtein_distances(smiles_vectors):
        def one_hot_to_string(one_hot_vector):
            token_indices = np.argmax(one_hot_vector, axis=-1)
            return ''.join(map(str, token_indices))

        n_samples = smiles_vectors.shape[0]
        levenshtein_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = Levenshtein.distance(
                    one_hot_to_string(smiles_vectors[i]),
                    one_hot_to_string(smiles_vectors[j])
                )
                levenshtein_matrix[i, j] = dist
                levenshtein_matrix[j, i] = dist

        return levenshtein_matrix

    # Compute normalized latent distances
    def compute_normalized_latent_distances(latent_representations):
        distances = squareform(pdist(latent_representations, metric='cosine'))
        return distances / 2

    # Compute distances
    levenshtein_distances = compute_levenshtein_distances(smiles_vectors)
    latent_distances = compute_normalized_latent_distances(z_mean_np)

    # Extract upper triangular distances
    levenshtein_upper = levenshtein_distances[np.triu_indices_from(levenshtein_distances, k=1)]
    latent_upper = latent_distances[np.triu_indices_from(latent_distances, k=1)]

    # Compute correlation
    correlation = np.corrcoef(levenshtein_upper, latent_upper)[0, 1]

    # Scatter plot of distances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(levenshtein_upper, latent_upper, alpha=0.6, label='Distance Pairs')

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(levenshtein_upper, latent_upper)
    line = slope * levenshtein_upper + intercept
    ax.plot(levenshtein_upper, line, color='r', label=f'Regression Line (R²={r_value ** 2:.4f})')

    ax.set_xlabel('Levenshtein Distances')
    ax.set_ylabel('Normalized Latent Space Distances')
    ax.set_title(f'Distance Correlation')
    ax.legend()

    plt.tight_layout()
    plt.close()

    # Return the figure
    return fig
