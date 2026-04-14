import torch

# Set random seed
torch.manual_seed(42)

class CovarianceModel():
    def __init__(self, p, d, alpha, sigma_Z, sigma_theta):
        """
        p: dimension of X
        d: dimension of latent space
        alpha: baseline affinity for covariances
        sigma_Z: standard deviation of latent positions z
        sigma_theta: standard deviation of correlation entries theta for observed data
        """
        self.p = p
        self.d = d
        self.sigma_Z = sigma_Z
        self.sigma_theta = sigma_theta

        self.alpha = torch.tensor(torch.ones(1) * alpha, requires_grad=True)
        self.Z = torch.tensor(torch.randn((p, d)) * sigma_Z, requires_grad=True)
        self.stdev = torch.ones(p, requires_grad=True)
        self.theta_entries = torch.tensor(self._gen_theta_entries() + torch.randn(p * (p - 1) // 2) * sigma_theta, requires_grad=True)
        # self.theta_entries = torch.tensor(self._gen_theta_entries() + torch.randn(p * (p + 1) // 2) * sigma_theta, requires_grad=True)
    
    def _gen_theta_entries(self):
        theta_entries = torch.zeros(self.p * (self.p - 1) // 2)
        # theta_entries = torch.zeros(self.p * (self.p + 1) // 2)
        theta_index = 0
        latent_dist = torch.cdist(self.Z, self.Z)
        # for j in range(self.p):
            # for k in range(j, self.p):
        for j in range(self.p-1):
            for k in range(j+1, self.p):
                affinity = self.alpha - latent_dist[j,k] ** 2
                theta_entries[theta_index] = 2 * torch.sigmoid(affinity) - 1
                # theta_entries[theta_index] = affinity
                theta_index += 1
        return theta_entries.detach()
    
    def get_cov(self):
        corr = torch.eye(self.p)
        # theta = torch.zeros((self.p, self.p))
        theta_index = 0
        # for j in range(self.p):
        #     for k in range(j, self.p):
        #         theta[j,k] = self.theta_entries[theta_index]
        #         theta[k,j] = theta[j,k]
        for j in range(self.p-1):
            for k in range(j+1, self.p):
                corr[j,k] = self.theta_entries[theta_index]
                corr[k,j] = corr[j,k]
                theta_index += 1
        cov = torch.diag(self.stdev) @ corr @ torch.diag(self.stdev)
        # return cov
        return self._validate_cov(cov)
        # return self._validate_cov(theta)
    
    def _validate_cov(self, cov):
        # for _ in range(2):
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals_clipped = torch.clamp(eigvals, min=1e-6)
        cov = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T + torch.eye(self.p) * 1e-6
        cov = (cov + cov.T) / 2
        return cov
    
    def get_model_params(self):
        return [self.alpha, self.Z, self.stdev, self.theta_entries]
        # return [self.alpha, self.Z, self.theta_entries]
    
    # Generate n samples according to given covariance matrix
    def gen_samples(self, n):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return dist.sample((n,))
    
    # Log likelihood of latent positions given variance
    def _Z_llk(self):
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.d), covariance_matrix=torch.eye(self.d) * self.sigma_Z)
        return torch.sum(dist.log_prob(self.Z), dim=None)
    
    # Log likelihood of covariance matrix entries given latent positions and variance
    def _theta_llk(self):
        distribution = torch.distributions.Normal(0, scale=self.sigma_theta)
        latent_dist = torch.cdist(self.Z, self.Z)
        cov = self.get_cov()
        llk = 0
        # for j in range(self.p):
            # for k in range(j, self.p):
        for j in range(self.p-1):
            for k in range(j+1, self.p):
                affinity = self.alpha - latent_dist[j,k] ** 2
                llk = llk + distribution.log_prob(cov[j,k] - 2 * torch.sigmoid(affinity) - 1)
                # llk = llk + distribution.log_prob(cov[j,k] - affinity)
        return llk
    
    # Log likelihood of observed data given covariance matrix
    def _X_llk(self, X):
        # cov = self.get_cov()
        # print(torch.allclose(cov, cov.T))
        # print(torch.min(torch.linalg.eigvalsh(cov)))
        # print(torch.isfinite(cov).all())
        dist = torch.distributions.MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.get_cov())
        return torch.sum(dist.log_prob(X), dim=None)

    # Compute loss as negative log likelihood of observed data given model parameters
    def loss(self, X):
        return -self._Z_llk() - self._theta_llk() - self._X_llk(X)

model = CovarianceModel(5, 5, 5, 1, 0.01)
print(model.get_cov())

model_2 = CovarianceModel(5, 5, 5, 1, 0.01)
print(model_2.get_cov())

data = model_2.gen_samples(250)

optim = torch.optim.Adam(model.get_model_params())

loss = model.loss(data)

for _ in range(100):
    optim.zero_grad()

    loss = model.loss(data)

    loss.backward()
    optim.step()

print(model.get_cov())

print(torch.cov(data.T))