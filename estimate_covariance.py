import torch

# Set random seed
torch.manual_seed(42)

# Set datatype to float64 to reduce errors due to precision
torch.set_default_dtype(torch.float64)

def print_evals(matrix):
    evals = torch.linalg.eigvalsh(matrix)
    print(f"{evals[0]}, {evals[-1]}")

def wishart_sample(df, covariance_matrix):
    p = covariance_matrix.size(dim=0)
    dist = torch.distributions.MultivariateNormal(torch.zeros(p), covariance_matrix=covariance_matrix)
    G = dist.sample((df,))
    # print(G)
    return G.T @ G + torch.eye(p) * 1e-4

class CovarianceModel():
    def __init__(self, p, d, df, sigma_Z):
        """
        p: dimension of X
        d: dimension of latent space
        df: degrees of freedom in the Inverse-Wishart distribution
        sigma_Z: standard deviation of latent positions z
        """
        self.p = p
        self.d = d
        if df < d:
            raise ValueError("Invalid degrees of freedom provided (must be at least p)")
        self.df = df
        self.sigma_Z = sigma_Z

        self.Z = torch.tensor(torch.randn((p, d)) * sigma_Z, requires_grad=True)
        self.theta = torch.tensor(self._gen_theta_entries(), requires_grad=True)
    
    def _gen_theta_entries(self):
        # print_evals(self.Z @ self.Z.T)
        Psi = self.Z @ self.Z.T + torch.eye(self.p) * 1e-4
        # print_evals(Psi)
        L1 = torch.linalg.cholesky(Psi)
        Psi_inv = torch.cholesky_inverse(L1)
        # print_evals(Psi_inv)
        dist = torch.distributions.wishart.Wishart(df=self.df, covariance_matrix=Psi_inv)
        sample = dist.rsample()
        # sample = wishart_sample(self.df, covariance_matrix=Psi_inv)
        # print_evals(sample)
        L2 = torch.linalg.cholesky(sample)
        # theta = torch.linalg.inv(L2).T
        theta = torch.linalg.cholesky(torch.cholesky_inverse(L2))
        return torch.round(theta, decimals=4).detach()
    
    def get_cov(self):
        return self.theta @ self.theta.T + torch.eye(self.p) * 1e-4
    
    def get_model_params(self):
        return [self.Z, self.theta]
    
    # Generate n samples according to given covariance matrix
    def gen_samples(self, n):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), scale_tril=self.theta + torch.eye(self.p) * 1e-4)
        # dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return dist.sample((n,))
    
    # Log likelihood of latent positions given variance
    def _Z_llk(self):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.d), covariance_matrix=torch.eye(self.d) * self.sigma_Z)
        return torch.sum(dist.log_prob(self.Z), dim=None)
    
    # Log likelihood of covariance matrix entries given latent positions and degrees of freedom
    def _theta_llk(self):
        Psi = self.Z @ self.Z.T + torch.eye(self.p) * 1e-4
        L1 = torch.linalg.cholesky(Psi)
        Psi_inv = torch.cholesky_inverse(L1)
        dist = torch.distributions.wishart.Wishart(df=self.df, covariance_matrix=Psi_inv)
        # L2 = torch.linalg.cholesky(self.get_cov())
        # cov_inv = torch.cholesky_inverse(L2)
        cov_inv = torch.cholesky_inverse(self.theta)
        llk = dist.log_prob(cov_inv) - (self.d + 1) * torch.logdet(self.get_cov())
        return llk
    
    # Log likelihood of observed data given covariance matrix
    def _X_llk(self, X):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), scale_tril=self.theta + torch.eye(self.p) * 1e-4)
        # dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return torch.sum(dist.log_prob(X), dim=None)

    # Compute loss as negative log likelihood of observed data given model parameters
    def loss(self, X):
        return -self._Z_llk() - self._theta_llk() - self._X_llk(X)

model = CovarianceModel(500, 10, 500, 1)
# print("Initial Covariance:")
# print(model.get_cov())

target_model = CovarianceModel(500, 10, 500, 1)
# print("Target Covariance:")
# print(target_model.get_cov())

data = target_model.gen_samples(100)

# print("Sample covariance:")
# print(torch.cov(data.T))

optim = torch.optim.Adam(model.get_model_params(), lr=0.01)

loss = model.loss(data)

for i in range(10000):
    optim.zero_grad()

    loss = model.loss(data)

    loss.backward()
    optim.step()

    if i % 1000 == 0:
        print("Fitted model:")
        print(torch.linalg.matrix_norm(target_model.get_cov() - model.get_cov()) ** 2)

# print("Fitted Covariance:")
# print(model.get_cov())

print("Sample covariance:")
print(torch.linalg.matrix_norm(target_model.get_cov() - torch.cov(data.T)) ** 2)

print("Fitted model:")
print(torch.linalg.matrix_norm(target_model.get_cov() - model.get_cov()) ** 2)

# sample = wishart_sample(df=5, covariance_matrix=torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
# print_evals(sample)