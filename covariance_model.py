import math
import torch
import pandas as pd

# Set random seed
torch.manual_seed(42)

# Set datatype to float64 to reduce errors due to precision
torch.set_default_dtype(torch.float64)

# Print leading and trailing eigenvalues of given matrix
def print_evals(matrix):
    evals = torch.linalg.eigvalsh(matrix)
    print(f'{evals[0]}, {evals[-1]}')

class CovarianceModel():
    def __init__(self, p, d, df=None, sigma_theta=None, sigma_Z=1.0, shift=1e-4):
        '''
        p: dimension of X
        d: dimension of latent space
        df: degrees of freedom in the inverse-Wishart distribution
        sigma_theta: desired element-wise variance of inverse-Wishart distribution
        sigma_Z: variance of latent positions Z
        shift: coefficient on scaled identity matrix added to ensure positive definiteness

        If neither of df or sigma_theta are specified, degaults to arbitrarily strong prior (infinite degrees of freedom)
        '''
        self.p = p
        self.d = d
        self.sigma_Z = sigma_Z
        self.shift = shift

        self.Z = torch.tensor(torch.randn((p, d)) * sigma_Z, requires_grad=True)

        # If prior strength is specified in terms of df, set df as normal
        if df != None:
            if df < p:
                raise ValueError('Invalid degrees of freedom provided (must be at least p)')
            self.df = df
        # If prior strength is specified in terms of sigma_theta, compute corresponding df
        elif sigma_theta != None:
            self.df = self._calc_df(sigma_theta ** 2)
        # If prior strength is not specified, default to arbitrarily strong prior
        else:
            self.df = None
        
        if self.df != None:
            self.theta = torch.tensor(self._gen_theta_entries(), requires_grad=True)
        else:
            self.theta = None

    # Compute appropriate degrees of freedom given desired entry-wise variance
    def _calc_df(self, target_var):
        def calc_sse(df):
            Psi = (self.Z @ self.Z.T + torch.eye(self.p) * self.shift).detach() * (df - self.p - 1)
            # Create vector of computed variances for covariance matrix entries
            actual_vars = torch.zeros(self.p * (self.p + 1) // 2)
            # Compute variances for each entry
            count = 0
            for i in range(self.p):
                for j in range(i, self.p):
                    actual_vars[count] = (df - self.p + 1) * Psi[i,j] ** 2 + (df - self.p - 1) * Psi[i,i] * Psi[j,j]
                    count += 1
            # Divide variances by common scaling factor
            actual_vars = actual_vars / (df - self.p) / (df - self.p - 1) ** 2 / (df - self.p - 3)
            return ((actual_vars - target_var) ** 2).sum()
        
        # Minimize SSE vs target variance using modified golden section search
        df_low = self.p + 4     # Minimum df for which variance can be computed
        df_high = self.p * 1000
        r = (math.sqrt(5) - 1) / 2
        a = int(df_high - r * (df_high - df_low))
        b = int(df_low + r * (df_high - df_low))
        sse_a = calc_sse(a)
        sse_b = calc_sse(b)

        while a < b:
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
    
    # Generate initialization of theta based on latent positions and inverse-Wishart distribution
    def _gen_theta_entries(self):
        # Compute scale matrix for inverse-Wishart distribution as ZZ^T
        # Note: Rescaled to ensure that output of inverse-Wishart distribution has mean equal to ZZ^T
        Psi = (self.Z @ self.Z.T + torch.eye(self.p) * self.shift) * (self.df - self.p - 1)
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
        if self.df != None:
            masked_theta = torch.tril(self.theta)
            # Add scaled identity to ensure that resulting matrix is positive definite
            return masked_theta @ masked_theta.T + torch.eye(self.p) * self.shift
        else:
            return self.Z @ self.Z.T + torch.eye(self.p) * self.shift
    
    # Get list of model parameters to be estimated
    def get_model_params(self):
        if self.df != None:
            return [self.Z, self.theta]
        else:
            return [self.Z]
    
    # Generate n samples according to given covariance matrix
    def gen_samples(self, n):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return dist.sample((n,))
    
    # Log likelihood of latent positions given variance
    def Z_llk(self):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.d), covariance_matrix=torch.eye(self.d) * self.sigma_Z ** 2)
        return torch.sum(dist.log_prob(self.Z), dim=None)
    
    # Log likelihood of covariance matrix entries given latent positions and degrees of freedom
    def theta_llk(self):
        if self.df != None:
            # Compute scale matrix for inverse-Wishart distribution as ZZ^T
            # Note: Rescaled to ensure that output of inverse-Wishart distribution has mean equal to ZZ^T
            Psi = (self.Z @ self.Z.T + torch.eye(self.p) * self.shift) * (self.df - self.p - 1)
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
        else:
            return None
    
    # Log likelihood of observed data given covariance matrix
    def X_llk(self, X):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return torch.sum(dist.log_prob(X), dim=None)

    # Compute loss as negative log likelihood of observed data given model parameters
    def loss(self, X, print_loss=False):
        Z_llk = self.Z_llk()
        X_llk = self.X_llk(X)
        if self.df != None:
            theta_llk = self.theta_llk()
            if print_loss:
                print(f'{(-Z_llk - theta_llk - X_llk):.4f}: {-Z_llk:.4f} + {-theta_llk:.4f} + {-X_llk:.4f}')
            return -Z_llk - theta_llk - X_llk
        else:
            if print_loss:
                print(f'{(-Z_llk - X_llk):.4f}: {-Z_llk:.4f} + {-X_llk:.4f}')
            return -Z_llk - X_llk
    
    # Optimize model parameters with respect to observed data X
    def optimize(self, X, steps=5000, lr=0.001):
        optim = torch.optim.AdamW(self.get_model_params(), lr=lr)
        for _ in range(steps):
            optim.zero_grad()
            loss = self.loss(X)
            loss.backward()
            optim.step()

if __name__ == "__main__":
    target_model = CovarianceModel(100, 10, sigma_theta=1.0, sigma_Z=1.0)
    print('Target model initialized!')

    model = CovarianceModel(100, 10, sigma_theta=1.0, sigma_Z=1.0)
    print('Training model initialized!')

    data = target_model.gen_samples(50)
    print('Samples generated!')

    learning_data = pd.DataFrame(columns=['Epoch', 'Loss', 'Z_llk', 'theta_llk', 'X_llk', 'SSE', 'RMSE'])

    optim = torch.optim.AdamW(model.get_model_params(), lr=0.001)

    max_epochs = 5000

    for i in range(max_epochs):
        optim.zero_grad()

        loss = model.loss(data)

        loss.backward()
        optim.step()

        if i % 100 == 0:
            if model.theta_llk() != None:
                learning_data.loc[len(learning_data)] = [i, loss.item(), model.Z_llk().item(), model.theta_llk().item(), model.X_llk(data).item(),
                                                         (torch.linalg.matrix_norm(target_model.get_cov() - model.get_cov()) ** 2).item(),
                                                         torch.sqrt(torch.mean((target_model.get_cov() - model.get_cov()) ** 2)).item()]
            else:
                learning_data.loc[len(learning_data)] = [i, loss.item(), model.Z_llk().item(), model.theta_llk(), model.X_llk(data).item(),
                                                         (torch.linalg.matrix_norm(target_model.get_cov() - model.get_cov()) ** 2).item(),
                                                         torch.sqrt(torch.mean((target_model.get_cov() - model.get_cov()) ** 2)).item()]
    
    if model.theta_llk() != None:
        learning_data.loc[len(learning_data)] = [max_epochs, model.loss(data).item(), model.Z_llk().item(), model.theta_llk().item(),
                                                 model.X_llk(data).item(), (torch.linalg.matrix_norm(target_model.get_cov() - model.get_cov()) ** 2).item(),
                                                 torch.sqrt(torch.mean((target_model.get_cov() - model.get_cov()) ** 2)).item()]
    else:
        learning_data.loc[len(learning_data)] = [max_epochs, model.loss(data).item(), model.Z_llk().item(), model.theta_llk(), model.X_llk(data).item(),
                                                 (torch.linalg.matrix_norm(target_model.get_cov() - model.get_cov()) ** 2).item(),
                                                 torch.sqrt(torch.mean((target_model.get_cov() - model.get_cov()) ** 2)).item()]

    learning_data = learning_data.astype({'Epoch': int})
    learning_data.to_csv('covariance_learning_curve.csv', index=False)