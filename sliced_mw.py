import torch
import scipy
import ot
import numpy as np
import torch
#from functorch import vmap  # If you're using PyTorch 2.0+, vmap is available as torch.vmap
from torch import vmap
import ot 
from joblib import Parallel, delayed

def project_mu(mu, v):
    """
    Maps a mean vector to 1D mean

    Args:
        A (torch.Tensor): Mean Vector
        v (torch.Tensor): Projection direction

    Returns:
        torch.Tensor: Means of 1D GMM
    """
    dot_products = torch.bmm(v.unsqueeze(1), mu.unsqueeze(-1)).squeeze()
    return dot_products

def project_sigma(A, v):
    """
    Maps a covariance to 1D variance

    Args:
        A (torch.Tensor): Covariance Matrix
        v (torch.Tensor): Projection direction

    Returns:
        torch.Tensor: Variances of 1D GMM
    """
    Av = torch.matmul(A, v.unsqueeze(-1))   # Shape: (batch_size, 2, 1)
    vTAv = torch.bmm(v.unsqueeze(1), Av)    # Shape: (batch_size, 1, 1)
    
    # Remove singleton dimensions for final result
    result = vTAv.squeeze()
    return result

def project_gmm_1d(gmm, v):
    """
    Maps a GMM to to 1D means and 1D variances

    Args:
        gmm (GaussianMixtureModel): Input GMM
        v (torch.Tensor): Projection direction

    Returns:
        torch.Tensor: Weights of 1D GMM
        torch.Tensor: Means of 1D GMM
        torch.Tensor: Variances of 1D GMM
    """
    weights_1d = gmm.weights
    mus_1d = project_mu(gmm.means, v)
    sigmas_1d = project_sigma(gmm.covariances, v)

    return weights_1d, mus_1d, sigmas_1d

def calc_MSW(gmm1, gmm2, pnum = 1000, threshold=None):
    """
    Calculate the MSW metric between two GMMs

    Args:
        gmm1, gmm2 (GaussianMixtureModel): Input GMMs
        pnum (int): Number of random projections

    Returns:
        torch.Tensor: Distance between gmm1 and gmm2
    """
    means1 = gmm1.means
    device = means1.device
    cov1 = gmm1.get_covariances()
    means2 = gmm2.means
    cov2 = gmm2.get_covariances()
    num_features = gmm1.means.shape[-1]
    proj_ls = []
    msw = torch.tensor(0.).to(device)
    C_MSW_raw = torch.zeros(means1.shape[0], means2.shape[0]).to(device)
    C_MSW = torch.ones(means1.shape[0], means2.shape[0]).to(device)
    for p in range(pnum):
        v = torch.rand(num_features).to(device)
        v = v/v.norm()
        v1 = v.unsqueeze(0).repeat((means1.shape[0], 1))
        w1, m1, s1 = project_gmm_1d(gmm1, v1)
        v2 = v.unsqueeze(0).repeat((means2.shape[0], 1))
        w2, m2, s2 = project_gmm_1d(gmm2, v2)
        s1 = torch.sqrt(s1)
        s2 = torch.sqrt(s2)
        C_MSW_raw = C_MSW_raw + sq_gaussian_wasserstein_distance_matrix(m1, s1, m2, s2)
        if threshold and (p+1)%10 == 0:
            C_MSW_new = C_MSW_raw/(p+1)
            deviation = abs(C_MSW.flatten() - C_MSW_new.flatten()).mean()#/C_MSW_new.flatten()).mean()
            C_MSW = C_MSW_new 
            if deviation < threshold:
                break
        else:
            C_MSW = C_MSW_raw/(p+1)
    P = ot.emd(w1, w2, C_MSW).detach()
    msw = (P * C_MSW).sum()
    return msw
    
def calc_SMSW(gmm1, gmm2, pnum = 1000, threshold=None):
    """
    Calculate the SMSW (or DSMW) metric between two GMMs

    Args:
        gmm1, gmm2 (GaussianMixtureModel): Input GMMs
        pnum (int): Number of random projections

    Returns:
        torch.Tensor: Distance between gmm1 and gmm2
    """
    means1 = gmm1.means
    cov1 = gmm1.get_covariances()
    means2 = gmm2.means
    cov2 = gmm2.get_covariances()
    num_features = gmm1.means.shape[-1]
    device = means1.device
    proj_ls = []
    smsw_raw = torch.tensor(0.).to(device)
    smsw = torch.tensor(1.).to(device)
    for p in range(pnum):
        v = torch.rand(num_features).to(device)
        v = v/v.norm()
        v1 = v.unsqueeze(0).repeat((gmm1.means.shape[0], 1))
        w1, m1, s1 = project_gmm_1d(gmm1, v1)
        v2 = v.unsqueeze(0).repeat((gmm2.means.shape[0], 1))
        w2, m2, s2 = project_gmm_1d(gmm2, v2)
        w1, m1, s1 = project_gmm_1d(gmm1, v1)
        v2 = v.unsqueeze(0).repeat((gmm2.means.shape[0], 1))
        w2, m2, s2 = project_gmm_1d(gmm2, v2)
        s1 = torch.sqrt(s1)
        s2 = torch.sqrt(s2)
        n1_2d = torch.concatenate([m1.unsqueeze(-1), s1.unsqueeze(-1)], dim=-1)
        if len(n1_2d.shape) == 1:
            n1_2d = n1_2d.reshape(1, -1)
        n2_2d = torch.concatenate([m2.unsqueeze(-1), s2.unsqueeze(-1)], dim=-1)
        if len(n2_2d.shape) == 1:
            n2_2d = n2_2d.reshape(1, -1)
        v = torch.rand(2).to(device)
        v = v/v.norm()
        v1 = v.unsqueeze(0).repeat((gmm1.means.shape[0], 1))
        v2 = v.unsqueeze(0).repeat((gmm2.means.shape[0], 1))

        n1_2d_proj = project_mu(n1_2d, v1)
        n2_2d_proj = project_mu(n2_2d, v2)
        if len(n1_2d_proj.shape) == 0:
            n1_2d_proj = n1_2d_proj.unsqueeze(0) 
        if len(n2_2d_proj.shape) == 0:
            n2_2d_proj = n2_2d_proj.unsqueeze(0) 
        sq_wasserstein_1d = ot.lp.wasserstein_1d(n1_2d_proj, n2_2d_proj, u_weights=w1, v_weights=w2, p=2)
        smsw_raw = smsw_raw + sq_wasserstein_1d
        if threshold and (p+1)%10 == 0:
            smsw_new = smsw_raw/(p+1)
            deviation = abs(smsw_new -smsw)#/smsw
            smsw = smsw_new
            if deviation < threshold:
                break
        else:
            smsw = smsw_raw/(p+1)
    return smsw



def calc_parallel_SMSW(gmm1, gmm2, pnum=1000, max=False):
    """
    Calculate the SMSW/DSMW metric between two GMMs using vectorized projections.
    Use un-parallel SMSW for single-component GMMs, i.e., Gaussians.
    Args:
        gmm1, gmm2: Gaussian Mixture Models that have attributes .means and a method get_covariances(),
                    and which are compatible with the projection functions.
        pnum (int): Number of random projections.
        max (bool): If True, return the maximum over the projections; otherwise, return the mean.
    
    Returns:
        torch.Tensor: The SMSW distance between gmm1 and gmm2.
    """
    device = gmm1.means.device
    num_features = gmm1.means.shape[-1]
    
    # Generate pnum random unit vectors for the primary projection (in the gmm feature space)
    v_main = torch.rand(pnum, num_features, device=device)
    v_main = v_main / v_main.norm(dim=1, keepdim=True)
    
    # Generate pnum random unit vectors in 2D for projecting the (mean, std) pairs
    v_2d = torch.rand(pnum, 2, device=device)
    v_2d = v_2d / v_2d.norm(dim=1, keepdim=True)

    # Define a function that computes the squared 1D Wasserstein distance for a single projection.
    def compute_projection(v_main_single, v_2d_single):
        # For gmm1:
        v1 = v_main_single.unsqueeze(0).expand(gmm1.means.shape[0], -1)  # shape: (n_components, num_features)
        w1, m1, s1 = project_gmm_1d(gmm1, v1)
        
        # For gmm2:
        v2 = v_main_single.unsqueeze(0).expand(gmm2.means.shape[0], -1)
        w2, m2, s2 = project_gmm_1d(gmm2, v2)
        # Standard deviations: assume s1, s2 are variances and take square root.
        s1 = torch.sqrt(s1)
        s2 = torch.sqrt(s2)
        
        # Create (mean, std) pairs for each component.
        n1_2d = torch.cat([m1.unsqueeze(-1), s1.unsqueeze(-1)], dim=-1)
        n2_2d = torch.cat([m2.unsqueeze(-1), s2.unsqueeze(-1)], dim=-1)
        # Project the (mean, std) pairs onto the 2D random direction.
        # We repeat the 2D projection vector for each component.
        v1_2d = v_2d_single.unsqueeze(0).expand(gmm1.means.shape[0], -1)
        v2_2d = v_2d_single.unsqueeze(0).expand(gmm2.means.shape[0], -1)
        n1_2d_proj = project_mu(n1_2d, v1_2d)
        n2_2d_proj = project_mu(n2_2d, v2_2d)
        # Compute the squared 1D Wasserstein distance between the projected 1D distributions.
        sq_wass = ot.lp.wasserstein_1d(
            n1_2d_proj, n2_2d_proj, 
            u_weights=w1, v_weights=w2, p=2
        )
        return sq_wass

    # Vectorize over the pnum projections. vmap will map compute_projection over each row of v_main and v_2d.
    # Here, in_dims=0 means that both v_main and v_2d are batched along dimension 0.
    try:
        proj_results = vmap(compute_projection, in_dims=(0, 0))(v_main, v_2d)
        if max:
            return proj_results.max()
        else:
            return proj_results.mean()
    except:
        print("Error occurred - Try calc_SMSW for Gaussian inputs (num_components=1) and informative errors!")
