# Estimated-Network-Models

In this project, we consider three primary application settings for estimated networks: correlation networks, feature interaction networks, and multivariate Hawkes processes.

If we use simulation data, each setting contains two parts:

1. **Simulation data generation**
2. **Parameter prediction**

Otherwise, we only need Parameter prediction.

## 1. Correlation Networks
## 2. Feature Interaction Networks 
### 2.1 Simulation data generation
$$
\begin{aligned}
&\qquad z_k \\
&\qquad \downarrow \\
z_j &\rightarrow \theta_{jk} \rightarrow y \leftarrow x \\
&\qquad\qquad\uparrow \\
&\qquad\qquad\;\beta
\end{aligned}
$$

$x_i \sim \mathcal{N}(0, \Sigma_x)$

$\Sigma_x = \sigma_x^2 \big(\rho^{|j-k|}\big)_{j,k=1}^p$

$\theta_{jk} \sim \mathcal{N}(\alpha - \|z_j - z_k\|_2^2,\; \sigma_\theta^2)$

$z_j \sim \mathcal{N}(0, \sigma_z^2 I_d)$

$y_i \sim \mathcal{N}\left(
\beta_0 + \beta^T x_i + \sum_{j<k}\theta_{jk}x_{ij}x_{ik},
\; \sigma_y^2
\right)$


### 3.2 Parameter prediction

The model factorization is

$$
p(y, x, \Theta, Z)
=
p(y \mid x, \Theta)\, p(x)\, p(\Theta \mid Z)\, p(Z)
$$

with

$$
p(y \mid x, \Theta)
=
\prod_{i=1}^n
\mathcal{N}\left(
y_i \;\middle|\;
\beta_0 + \beta^T x_i + \sum_{j<k}\theta_{jk}x_{ij}x_{ik},
\; \sigma_y^2
\right)
$$

$$
p(x)
=
\prod_{i=1}^n \mathcal{N}(x_i \mid 0, \Sigma_x)
$$

$$
p(\Theta \mid Z)
=
\prod_{j<k}
\mathcal{N}\left(
\theta_{jk}
\;\middle|\;
\alpha - \|z_j - z_k\|_2^2,
\; \sigma_\theta^2
\right)
$$

$$
p(Z)
=
\prod_{j=1}^p \prod_{\ell=1}^d
\mathcal{N}(z_{j\ell} \mid 0, \sigma_z^2)
$$


The full negative log-likelihood is

$ \mathrm{NLL}=-\log p(y, x, \Theta, Z) $

So,

$\mathrm{NLL}= -\log p(y \mid x, \Theta)-\log p(x)-\log p(\Theta \mid Z)-\log p(Z) $

Since \(p(x)\) does not depend on the trainable parameters, it is treated as a constant during optimization.

Therefore, the training loss is

$ \mathcal{L}=-\log p(y \mid x, \Theta)-\log p(\Theta \mid Z)-\log p(Z) $


### 3.3 Assessment

- **Simulation data:** compare the estimated interaction matrix with the true interaction matrix

$$
\|\Theta - \hat{\Theta}\|_F^2
$$

- **Real data:** compare the prediction with the observed response

$$
\|y - \hat y\|_2^2
$$
## 3. Multivariate Hawkes Processes