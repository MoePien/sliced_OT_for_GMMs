import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from GMM_utils import GaussianMixtureModel, nearest_psd
import matplotlib.pyplot as plt
import os

def load_mnist_image(index, device="cpu", train=True):
    """Load an MNIST image by index and return its pixel values."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root="./data", train=train, download=True, transform=transform)
    image, label = dataset[index]
    return image.squeeze(0).to(device), label  # Shape: (28, 28)


def load_mnist_by_class(index, target_label, device="cpu", train=True):
    """Load an MNIST image of the specified class label."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root="./data", train=train, download=True, transform=transform)
    
    # Find indices of images with the target label
    indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    
    if not indices:
        raise ValueError(f"No images found for label {target_label}.")

    image, label = dataset[indices[index]]
    return image.squeeze(0).to(device), label  # Shape: (28, 28)

def fit_gmm_to_image(image, num_components=10, device="cpu", requires_grad=False, 
                     nsample=2000, n_init=50, var=.2, type_cov="full"):
    """Fit a 2D GMM to the given MNIST image."""
    height, width = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([x.ravel(), y.ravel()], axis=1)
    intensities = image.cpu().numpy().ravel()
    
    # Normalize intensities to create weighted samples
    intensities = intensities / intensities.sum()
    sampled_pixels = pixels[np.random.choice(len(pixels), size=nsample, p=intensities, replace=True)]
    sampled_pixels = sampled_pixels.astype(float)
    sampled_pixels += var *  np.random.randn(*sampled_pixels.shape)
    # Fit Gaussian Mixture Model
    gmm = SklearnGaussianMixture(n_components=num_components, covariance_type=type_cov, random_state=42, n_init=n_init)
    gmm.fit(sampled_pixels)
    
    # Convert to PyTorch tensors
    weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=device)
    means = torch.tensor(gmm.means_, dtype=torch.float32, device=device)
    covariances = torch.tensor(gmm.covariances_, dtype=torch.float32, device=device)
    
    return GaussianMixtureModel(weights, means, covariances, optimize=requires_grad)

def get_mnist_gmm(index, num_components=10, device="cpu", train=True, 
                  requires_grad=False, target_label=None, nsample=1000, n_init=50, var=.2, type_cov="full"):
    """Load an MNIST image, fit a GMM, and return the trained GMM model."""
    if target_label is None:
        image, label = load_mnist_image(index, device=device, train=train)
    else:
        image, label = load_mnist_by_class(index, target_label, device=device, train=train)
    gmm_out = fit_gmm_to_image(image, num_components=num_components, device=device, 
                               requires_grad=requires_grad, nsample=nsample, n_init=n_init, var=var, type_cov="full")
    return gmm_out, label

def plot_gmm_contours(gmm, image_shape=(28, 28), label=None, savename=None):
    """Plot the contour of a fitted GMM over the corresponding MNIST image space."""
    x, y = np.meshgrid(np.linspace(0, image_shape[1] - 1, 2 * image_shape[1]),
                       np.linspace(0, image_shape[0] - 1, 2 * image_shape[0]))
    grid = np.stack([x.ravel(), y.ravel()], axis=1)
    
    # Compute probability density function
    density = np.zeros(grid.shape[0])
    for i in range(gmm.num_components):
        cov = nearest_psd(gmm.covariances[i:i+1].detach().cpu().numpy())[0]
        cov = np.nan_to_num(cov)
        cov += np.eye(cov.shape[-1]) * 1e-4 # Avoid numerical instability
        mvn = multivariate_normal(mean=gmm.means[i].detach().cpu().numpy(), cov=cov)
        density += gmm.weights[i].detach().cpu().numpy() * mvn.pdf(grid)
    
    # Reshape density to match image dimensions
    density = density.reshape(( 2* image_shape[0], 2 * image_shape[1]))
    
    # Plot contours
    plt.figure(figsize=(6, 6))
    plt.contourf(x, np.flip(y), density, levels=20, cmap="inferno")
    plt.xticks([])
    plt.yticks([])
    if savename:
        save_dir = "output_GMM/" + str(label)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "/" + str(savename), bbox_inches='tight', pad_inches=0)
        plt.close()