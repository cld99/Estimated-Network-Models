import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class FeatureInteractionModel(nn.Module):
    def __init__(self, p, d,sigma_y=None, sigma_theta=None, sigma_z=None):
        """
        p: dimension of X
        d: dimension of latent space
        """
        super().__init__()

        self.p = p
        self.d = d
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta
        self.sigma_z = sigma_z
        # regression parameters
        self.beta0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(p))
        # interaction matrix parameters
        # self.Theta = nn.Parameter(torch.zeros(p, p))
        self.theta_upper = nn.Parameter(torch.zeros(p * (p - 1) // 2))
        # latent parameters
        self.Z = nn.Parameter(torch.randn(p, d))
        self.alpha = nn.Parameter(torch.zeros(1))
        
        # log variance parameters
        # sigma_y^2 = exp(log_sigma_y^2})  use log variance so variance is always positive after exp()
        # self.log_sigma_y2 = nn.Parameter(torch.zeros(1))
        # self.log_sigma_theta2 = nn.Parameter(torch.zeros(1))
        # self.log_sigma_z2 = nn.Parameter(torch.zeros(1))
        if sigma_y is None:
            self.log_sigma_y2 = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma_y2",torch.tensor([np.log(sigma_y**2)], dtype=torch.float32))
        if sigma_theta is None:
            self.log_sigma_theta2 = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma_theta2",torch.tensor([np.log(sigma_theta**2)], dtype=torch.float32))
        if sigma_z is None:
            self.log_sigma_z2 = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma_z2",torch.tensor([np.log(sigma_z**2)], dtype=torch.float32))
            
    def forward(self, X):
        """
        y_i = beta0 + sum_j beta_j x_ij + sum_{j<k} Theta_jk x_ij x_ik
        """
        n, p = X.shape
        y_hat = self.beta0 + X @ self.beta

        interaction_term = torch.zeros(n, device=X.device)
        idx = 0
        for j in range(p):
            for k in range(j + 1, p):
                # interaction_term += self.Theta[j, k] * X[:, j] * X[:, k]
                interaction_term += self.theta_upper[idx] * X[:, j] * X[:, k]
                idx += 1

        y_hat = y_hat + interaction_term
        return y_hat
    
    def _nll_theta_given_z(self):
        """
        -log p(Theta | Z)
        Assume for j<k: Theta_jk ~ N(alpha - ||z_j - z_k||^2, sigma_theta^2)
        """
        loss_theta_given_z = 0.0
        sigma_theta = torch.exp(0.5 * self.log_sigma_theta2)
        idx = 0
        for j in range(self.p):
            for k in range(j+1, self.p):
                mean_jk = self.alpha - torch.sum((self.Z[j] - self.Z[k]) ** 2)
                dist = Normal(loc=mean_jk, scale=sigma_theta)
                # loss_theta_given_z -= dist.log_prob(self.Theta[j, k])
                loss_theta_given_z -= dist.log_prob(self.theta_upper[idx])
                idx += 1

        return loss_theta_given_z
    
    def _nll_z(self):
        """
        -log p(Z)
        Assume each entry Z_jl ~ N(0, sigma_z^2)
        """
        loss_z = 0.0
        sigma_z = torch.exp(0.5 * self.log_sigma_z2)
        for j in range(self.p):
            for l in range(self.d):
                dist = Normal(loc=0, scale=sigma_z)
                loss_z -= dist.log_prob(self.Z[j,l])
        
        return loss_z
    
    def _nll_y_given_x_theta(self, X, y):
        """
        -log p(y | X, Theta)
        Assume y_i ~ N(mean_yi, sigma_y^2)
        """
        sigma_y = torch.exp(0.5 * self.log_sigma_y2)
        y_hat = self.forward(X).squeeze()
        dist = Normal(loc=y_hat, scale=sigma_y)
        loss_y_given_x_theta = -dist.log_prob(y).sum()
        return loss_y_given_x_theta
        
    
    def loss(self, X, y):
        """
        total loss = -log p(y | X, Theta) - log p(Theta | Z) - log p(Z) - log p(X)
        In this setting, p(X) is constant with respect to the model parameters, so it does not affect optimization.
        """
        loss_y_given_x_theta = self._nll_y_given_x_theta(X, y)
        loss_theta_given_z = self._nll_theta_given_z()
        loss_z = self._nll_z()
        total_loss = loss_y_given_x_theta + loss_theta_given_z + loss_z
        
        return total_loss
    
    def fit(self, X, y, lr=0.01, epochs=1000 ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = self.loss(X, y)
            total_loss.backward()
            optimizer.step()
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}: loss = {total_loss.item():.6f}")
        return

    
    def predict(self, X):
        return self.forward(X)