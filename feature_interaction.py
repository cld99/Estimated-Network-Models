import numpy as np
from simu_feature_interaction import simulate_data
import torch
import torch.nn as nn
import torch.optim as optim


def build_interaction_features(X):
    n, p = X.shape
    cols = []
    pair_idx = []

    for j in range(p):
        for k in range(j + 1, p):
            cols.append(X[:, j] * X[:, k])
            pair_idx.append((j, k))

    X_inter = np.column_stack(cols)
    return X_inter, pair_idx

class InteractionFeatureModel(nn.Module):
    def __init__(self, p, d,use_latent_penalty=False, regularization=False,
                 gamma=0.0, lambda_beta=0.0, lambda_theta=0.0):
        """
        p: dimension of X
        d: dimension of latent space
        regularization: whether to include L2 regularization
        lambda_beta: L2 penalty coefficient for beta
        lambda_theta: L2 penalty coefficient for Theta
        gamma: latent-space penalty coefficient
        """
        super().__init__()

        self.p = p
        self.d = d
        self.use_latent_penalty = use_latent_penalty

        self.lambda_beta = lambda_beta
        self.lambda_theta = lambda_theta

        self.beta0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(p))
        self.Theta = nn.Parameter(torch.zeros(p, p))

        # latent structure parameters for using latent space to get theta
        if self.use_latent_penalty:
            self.Z = nn.Parameter(torch.randn(p, d))
            self.alpha_ls = nn.Parameter(torch.zeros(1))
            
    def forward(self, X):
        n, p = X.shape
        y_hat = self.beta0 + X @ self.beta

        interaction_term = torch.zeros(n, device=X.device)
        for j in range(p):
            for k in range(j + 1, p):
                interaction_term += self.Theta[j, k] * X[:, j] * X[:, k]

        y_hat = y_hat + interaction_term
        return y_hat
    
    def _latent_penalty(self):
        # sum_{j<k} [Theta_jk - (alpha - ||z_j-z_k||^2)]^2
        if not self.use_latent_penalty:
            return torch.tensor(0.0, device=self.beta.device)

        penalty = 0.0
        idx = 0
        for j in range(self.p):
            for k in range(j + 1, self.p):
                latent_value = self.alpha_ls - torch.sum((self.Z[j] - self.Z[k]) ** 2)
                penalty = penalty + (self.theta_vec[idx] - latent_value) ** 2
                idx += 1

        return penalty
   
    def _regularization_loss(self):
        reg_loss = 0.0

        if self.regularization:
            reg_loss = reg_loss + self.lambda_beta * torch.sum(self.beta ** 2)

            if self.interaction:
                theta_penalty = 0.0
                for j in range(self.p):
                    for k in range(j + 1, self.p):
                        theta_penalty += self.Theta[j, k] ** 2

                reg_loss = reg_loss + self.lambda_theta * theta_penalty

        return reg_loss
    
    def _prediction_loss(self, X, y):
        y_hat = self.forward(X)
        pred_loss = torch.mean((y - y_hat) ** 2)
        return pred_loss
    
    def _loss(self, X, y):
        pred_loss = self._prediction_loss(X, y)
        reg_loss = self._regularization_loss()

        if self.use_latent_penalty:
            ls_loss = self.gamma * self._latent_penalty()
        else:
            ls_loss = 0.0

        total_loss = pred_loss + reg_loss + ls_loss
        return total_loss