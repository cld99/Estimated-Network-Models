import math
import torch

# Set random seed
torch.manual_seed(42)

# Set datatype to float64 to reduce errors due to precision
torch.set_default_dtype(torch.float64)

# Print leading and trailing eigenvalues of given matrix
def print_evals(matrix):
    evals = torch.linalg.eigvalsh(matrix)
    print(f"{evals[0]}, {evals[-1]}")

# Compute appropriate degrees of freedom given desired entry-wise variance and dimension p
def calc_df(target_var, Psi, p):
    def calc_sse(df):
        # Create vector of computed variances for covariance matrix entries
        actual_vars = torch.zeros(p * (p + 1) // 2)
        # Compute variances for each entry
        count = 0
        for i in range(p):
            for j in range(i, p):
                actual_vars[count] = (df - p + 1) * Psi[i,j] + (df - p - 1) * Psi[i,i] * Psi[j,j]
                count += 1
        # Divide variances by common scaling factor
        actual_vars = actual_vars / (df - p) / (df - p - 1) ** 2 / (df - p - 3)
        return ((actual_vars - target_var) ** 2).sum()
    
    df_low = p + 4
    df_high = p * 100
    r = (math.sqrt(5) - 1) / 2
    a = int(df_high - r * (df_high - df_low))
    b = int(df_low + r * (df_high - df_low))
    sse_a = calc_sse(a)
    sse_b = calc_sse(b)

    while a != b:
        if sse_a < sse_b:
            df_high = b
            b = a
            sse_b = sse_a
            a = int(df_high - r * (df_high - df_low))
            sse_a = calc_sse(a)
        else:
            df_low = a
            a = b
            sse_a = sse_b
            b = int(df_low + r * (df_high - df_low))
            sse_b = calc_sse(b)
    
    return a

class CovarianceModel():
    def __init__(self, p, d, df=None, sigma_theta=None, sigma_Z=1.0, diag_shift=1e-4):
        """
        p: dimension of X
        d: dimension of latent space
        df: degrees of freedom in the Inverse-Wishart distribution (must specify one of df or sigma_theta)
        sigma_theta: desired variance of Inverse-Wishart distribution (must specify one of df or sigma_theta)
        sigma_Z: variance of latent positions z
        diag_shift: diagonal term added to ensure positive semidefiniteness
        """
        self.p = p
        self.d = d
        self.sigma_Z = sigma_Z
        self.diag_shift = diag_shift

        self.Z = torch.tensor(torch.randn((p, d)) * sigma_Z, requires_grad=True)

        if df != None:
            if df < d:
                raise ValueError("Invalid degrees of freedom provided (must be at least p)")
            self.df = df
        else:
            self.df = calc_df(sigma_theta^2, (self.Z @ self.Z.T).detach() + torch.eye(self.p) * self.diag_shift, self.p)
        
        self.theta = torch.tensor(self._gen_theta_entries(), requires_grad=True)
    
    # Generate initialization of theta based on latent positions and inverse-Wishart distribution
    def _gen_theta_entries(self):
        # Compute scale matrix for inverse-Wishart distribution as ZZ^T, adding small diagonal matrix to ensure positive definiteness
        Psi = self.Z @ self.Z.T + torch.eye(self.p) * self.diag_shift
        # Invert inverse-Wishart scale matrix to get scale matrix for Wishart distribution
        L1 = torch.linalg.cholesky(Psi)
        Psi_inv = torch.cholesky_inverse(L1)
        # Sample initial scatter matrix from Wishart distribution
        dist = torch.distributions.wishart.Wishart(df=self.df, covariance_matrix=Psi_inv)
        sample = dist.rsample()
        # Invert scatter matrix to get covariance matrix, and set theta as lower-triangular decomposition of it
        L2 = torch.linalg.cholesky(sample)
        theta = torch.linalg.cholesky(torch.cholesky_inverse(L2))
        return torch.round(theta, decimals=4).detach()
    
    # Generate covariance matrix based on current theta values
    def get_cov(self):
        masked_theta = torch.tril(self.theta)
        return masked_theta @ masked_theta.T + torch.eye(self.p) * self.diag_shift
    
    # Get list of model parameters to be estimated
    def get_model_params(self):
        return [self.Z, self.theta]
    
    # Generate n samples according to given covariance matrix
    def gen_samples(self, n):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return dist.sample((n,))
    
    # Log likelihood of latent positions given variance
    def _Z_llk(self):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.d), covariance_matrix=torch.eye(self.d) * self.sigma_Z^2)
        return torch.sum(dist.log_prob(self.Z), dim=None)
    
    # Log likelihood of covariance matrix entries given latent positions and degrees of freedom
    def _theta_llk(self):
        # Compute scale matrix for inverse-Wishart distribution as ZZ^T, adding small diagonal matrix to ensure positive definiteness
        Psi = self.Z @ self.Z.T + torch.eye(self.p) * self.diag_shift
        # Invert inverse-Wishart scale matrix to get scale matrix for Wishart distribution
        L1 = torch.linalg.cholesky(Psi)
        Psi_inv = torch.cholesky_inverse(L1)
        # Initialize Wishart distribution
        dist = torch.distributions.wishart.Wishart(df=self.df, covariance_matrix=Psi_inv)
        # Compute scatter matrix based on current theta values
        L2 = torch.linalg.cholesky(self.get_cov())
        cov_inv = torch.cholesky_inverse(L2)
        # Compute LLK (note the additional factor dependent on the determinant due to the inversion being applied)
        return dist.log_prob(cov_inv) - (self.d + 1) * torch.logdet(self.get_cov())
    
    # Log likelihood of observed data given covariance matrix
    def _X_llk(self, X):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return torch.sum(dist.log_prob(X), dim=None)

    # Compute loss as negative log likelihood of observed data given model parameters
    def loss(self, X):
        return -self._Z_llk() - self._theta_llk() - self._X_llk(X)
    
    # Optimize parameters based on provided optimizer
    def optimize(self, data, steps):
        optim = torch.optim.AdamW(self.get_model_params(), lr=0.001, eps=1e-6, betas=(0.9, 0.99))
        for _ in range(steps):
            # Zero gradients
            optim.zero_grad()
            # Compute loss
            loss = self.loss(data)
            # Backpropagate to perform gradient update
            loss.backward()
            optim.step()