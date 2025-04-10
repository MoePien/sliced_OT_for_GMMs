import torch
from torch.distributions import Categorical, MultivariateNormal
import numpy as np
from sklearn.mixture import GaussianMixture as sklearnGaussianMixture
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

def batch_diagonal_matrices(tensor):
    """
    Convert a tensor of shape (N, D) into a batch of diagonal matrices of shape (N, D, D).
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, D).
        
    Returns:
        torch.Tensor: Batch of diagonal matrices of shape (N, D, D).
    """
    N, D = tensor.shape
    return torch.diag_embed(tensor)  # Shape: (N, D, D)


def nearest_psd(A):
    """
    Project each matrix in the input array to the nearest positive semi-definite matrix.
    
    Args:
        A (np.ndarray): An array of shape (N, D, D) containing N square matrices of dimension D x D.
        
    Returns:
        np.ndarray: An array of shape (N, D, D) with each matrix projected to be positive semi-definite.
    """
    # Ensure input is a 3D array where each slice A[i] is a D x D matrix
    if len(A.shape) != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("Input array must be of shape (N, D, D), where each matrix is square.")
    
    # Initialize an array to store the PSD matrices
    psd_matrices = np.zeros_like(A)
    
    for i in range(A.shape[0]):
        # Perform eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(A[i])
        
        # Set any negative eigenvalues to zero to enforce positive semi-definiteness
        eigvals[eigvals < 0] = 0
        
        # Reconstruct the matrix using the modified eigenvalues
        psd_matrices[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
    return psd_matrices

def get_cholesky(L_param):
    # Extract the lower-triangular part (strictly lower part remains unchanged)
    L_lower = torch.tril(L_param, diagonal=-1)
    # For the diagonal, enforce positivity via softplus
    diag = torch.diagonal(L_param, dim1=-2, dim2=-1)
    diag_positive = torch.nn.functional.softplus(diag)
    # Reconstruct L: add the modified diagonal back
    L = L_lower + torch.diag_embed(diag_positive)
    return L


def generate_random_covariances(batch_size, num_features, corr=False):
    """Generates batch_size covariance matrices using Cholesky decomposition."""
    if corr:
        random_matrices = torch.rand(batch_size, num_features, num_features)
        covariances = torch.tensor(nearest_psd(random_matrices))
        covariances += torch.eye(num_features).unsqueeze(0) * 1e-3
    else:
        L = torch.rand(batch_size, num_features, num_features)  # Sample a random matrix
        L = torch.tril(L)  # Keep only the lower triangular part
        covariances = torch.bmm(L, L.transpose(1, 2))  # Compute L L^T to ensure PSD
    
    return covariances
    
class GaussianMixtureModel(torch.nn.Module):
    """
    Gaussian Mixture Model (GMM) implemented in PyTorch.

    This class represents a Gaussian Mixture Model (GMM) with an optional optimization
    routine that allows learning of mixture weights, means, and covariances.

    Parameters:
        weights (Tensor): Mixture weights of shape (K,), where K is the number of components.
        means (Tensor): Component means of shape (K, D), where D is the number of features.
        covariances (Tensor): Covariance matrices of shape (K, D, D), one per component.
        optimize (bool, optional): If True, the weights, means, and Cholesky factors of 
            the covariances are made learnable parameters. Default: False.
        reg_cov (float, optional): Regularization constant added to the diagonal of each 
            covariance matrix for numerical stability. Default: 1e-4.
        device (str or torch.device, optional): Device to place the tensors on. If None,
            uses CUDA if available, otherwise CPU.
    """
    def __init__(self, weights, means, covariances, optimize=False, reg_cov=1e-4, device=None):
        super(GaussianMixtureModel, self).__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_components = weights.shape[0]
        self.num_features = means.shape[1]
        self.optimize = optimize
        self.reg_cov = reg_cov
        self.diag = False
        self.reg_cov = reg_cov
        if reg_cov:
            diag = torch.eye(covariances.shape[-1]).unsqueeze(0).to(device)
            covariances = covariances.to(device) + reg_cov * diag.repeat(covariances.shape[0], 1, 1)
            # Compute initial Cholesky factor (which is lower triangular)
            L_init = torch.linalg.cholesky(covariances).to(device)
        else:
            L_init = torch.tensor(0.)
        self.covariances_pre = covariances.to(device)
        # Now handle different optimization regimes:
        if optimize:
            # Here we optimize over means, weights, and the full covariance via its Cholesky factor.
            self.weights_pre = torch.nn.Parameter(weights, requires_grad=True).to(device)
            self.means = torch.nn.Parameter(means, requires_grad=True).to(device)
            # Instead of directly making L learnable, we use an unconstrained parameter
            # and later project it to a lower triangular matrix with positive diagonal.
            self.L_param = torch.nn.Parameter(L_init, requires_grad=True).to(device)
        else:
            self.weights = weights.to(device)
            self.means = means.to(device)
            self.L_param = L_init.to(device)
            
        self.covariances = self.get_covariances().to(device)
        self.set_conditions()
        
    def get_covariances(self):
        """Compute positive definite covariance matrices dynamically."""
        if self.optimize:
            # Use our helper to project self.L_param to a valid lower-triangular matrix.
            L = get_cholesky(self.L_param)
            return L @ L.transpose(-1, -2)
        else:
            return self.covariances_pre

    def set_covariances(self):
        """Set positive definite covariance matrices dynamically."""
        self.covariances = self.get_covariances().nan_to_num()
        return None

    def set_weights(self):
        """Set simplex weights dynamically (only if optimize)."""
        if hasattr(self, 'weights_pre'):
            self.weights = torch.nn.functional.softmax(self.weights_pre, dim=0)
        return None
        
    def set_conditions(self):
        """Update covariance and weights (if optimizing)."""
        self.set_covariances()
        self.set_weights()
        return None

    def forward(self, x):
        categorical = Categorical(logits=self.weights)
        covariances = self.get_covariances()
        mvns = [MultivariateNormal(self.means[i], covariances[i]) for i in range(self.num_components)]
        component_probs = torch.stack([torch.exp(mvn.log_prob(x)) for mvn in mvns], dim=1)
        weighted_probs = component_probs * categorical.probs
        return weighted_probs.sum(dim=1)

    def sample(self, num_samples):
        """
        Sample from the GMM.
        
        Args:
            num_samples (int): Number of samples to draw.
            return_components (bool): If True, also return the sampled component indices.
        
        Returns:
            samples (torch.Tensor): Samples from the mixture, shape (num_samples, num_features).
            (optional) components (torch.Tensor): Component indices for each sample.
        """
        # Ensure the model's parameters are up to date.
        self.set_conditions()
        covariances = self.get_covariances()
        # Construct the mixture distribution.
        cat = Categorical(self.weights)
        comp = MultivariateNormal(self.means, covariance_matrix=covariances)
        mixture = MixtureSameFamily(cat, comp)
        samples = mixture.sample((num_samples,))
        return samples


class RandomGaussianMixtureModel(GaussianMixtureModel):
    def __init__(self, num_components, num_features, device="cpu", corr=True, optimize=False, normal_means=False, reg_cov=1e-4):
        """Produces a random Gaussian mixture
        Args:
            num_components (int): Number of mixture components.
            num_features (int): Dimension.
            device (str): Device to run the model on ("cuda" or "cpu").
            corr (bool): True: full covariances, False: Diagional ones
            normal_means: True: Standard normal means, False: Uniform means in [0, 10]
            reg_cov: add reg_cov * Identity matric to all covariances for stability
        """
        weights = torch.rand(num_components).to(device)
        weights = weights / weights.sum()
        if normal_means:
            means = torch.randn(num_components, num_features).to(device)
        else:
            means = 10 * torch.rand(num_components, num_features).to(device)
        covariances = generate_random_covariances(num_components, num_features, corr=corr).to(device)
        super().__init__(weights, means, covariances, optimize=optimize, device=device, reg_cov=reg_cov)

class FittedGaussianMixtureModel(GaussianMixtureModel):
    def __init__(self, data, num_components, optimize=False, device=None, reg_cov=1e-4, n_init=1):
        """
        Fits a Gaussian Mixture Model (GMM) to the given data and initializes a PyTorch-compatible model.

        Args:
            data (torch.Tensor): Input data of shape (N, D), where N is the number of samples and D is the feature dimension.
            num_components (int): Number of mixture components.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gmm = sklearnGaussianMixture(n_components=num_components, covariance_type='full', n_init=n_init)
        gmm.fit(data.cpu().numpy())
        
        weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=device)
        means = torch.tensor(gmm.means_, dtype=torch.float32, device=device)
        covariances = torch.tensor(gmm.covariances_, dtype=torch.float32, device=device)
        
        super().__init__(weights, means, covariances, optimize=optimize, device=device, reg_cov=reg_cov)

class EmpiricalGMM(GaussianMixtureModel):
    def __init__(self, data, optimize=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Creates an empirical Gaussian Mixture Model where each data point is a component.

        Args:
            data (torch.Tensor): Input data of shape (N, D), where N is the number of samples and D is the feature dimension.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        num_components, num_features = data.shape
        weights = torch.ones(num_components, device=device) / num_components
        means = data.to(device)
        covariances = torch.zeros(num_components, num_features, num_features).to(device)
        super().__init__(weights, means, covariances, optimize=optimize, device=device, reg_cov=False)