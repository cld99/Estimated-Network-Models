# Estimated-Network-Models

In this project, we consider three primary application settings for estimated networks: correlation networks, feature interaction networks, and multivariate Hawkes processes.

If simulation data are used, each setting contains two parts:

1. **Simulation data generation**
2. **Parameter prediction**

Otherwise, we only need **parameter prediction**.

## 1. Correlation Networks

### 1.1 Simulated Data Generation
```math
\begin{aligned}
&\qquad z_k \\
&\qquad \downarrow \\
z_j &\rightarrow \theta_{jk} \rightarrow x
\end{aligned}
```

$d$-dimensional latent positions are sampled according to a multivariate normal distribution:

```math
z_i \sim \mathcal{N}(0, I\sigma_z^2), \text{ } i \in \{0, 1, ..., p\}
```

The covariance matrix is sampled according to an inverse-Wishart distribution with scale matrix equal to $ZZ^T$ and $\nu$ degrees of freedom:

```math
\text{Cov} \sim \mathcal{W}^{-1}(ZZ^T, \nu)
```

The covariance matrix, which is guaranteed to be symmetric positive definite by the properties of the inverse-Wishart distribution, is parameterized by lower triangular matrix $\Theta$ such that $\text{Cov} = \Theta\Theta^T$.

### 1.2 Parameter Estimation

Parameter estimation is performed by maximizing the log-likelihood of the data and model parameters given the generative process described in section 1.1. Thus, the following loss function is minimized with respect to latent positions $Z$ and covariance parameters $\Theta$:

```math
\text{Loss} = -l(Z) - l(\Theta|Z) - l(X|\Theta)
```

```math
l(Z) = \sum_{i=1}^p\ln((2\pi)^{-k/2}\det(I\sigma_z^2)^{-1/2}\exp(\frac{-1}{2}z_i^T(I\sigma_z^2)^{-1}z_i))
```

```math
l(\Theta|Z) = \ln(\frac{|ZZ^T|^{\nu/2}}{2^{\nu p/2}\Gamma_p(\frac{\nu}{2})}|\Theta\Theta^T|^{-(\nu+p+1)/2}e^{-\frac{1}{2}\text{tr}(ZZ^T(\Theta\Theta^T)^{-1})})
```

```math
l(X|\Theta) = \sum_{i=1}^p\ln((2\pi)^{-p/2}\det(\Theta\Theta^T)^{-1/2}\exp(\frac{-1}{2}x_i^T(\Theta\Theta^T)^{-1}x_i))
```

### 1.3 Evaluation

- **Simulated Data:** For simulated data, we can directly compare the estimated covariance values to the true values using the Frobenius norm of the difference between the covariance matrices:

```math
\|\Theta - \hat{\Theta}\|_F^2
```

- **Real Data:** For real data, we do not known the true covariance values, so we instead evaluate the log-likelihood of a held out set of data on the fitted covariance model:

```math
l(X_{\text{test}}|\Theta) = \sum_{x_i \in X_{\text{test}}}\ln((2\pi)^{-p/2}\det(\Theta\Theta^T)^{-1/2}\exp(\frac{-1}{2}x_i^T(\Theta\Theta^T)^{-1}x_i))
```

## 2. Feature Interaction Networks

### 2.1 Simulation data generation

```math
\begin{aligned}
&\qquad z_k \\
&\qquad \downarrow \\
z_j &\rightarrow \theta_{jk} \rightarrow y \leftarrow x \\
&\qquad\qquad\uparrow \\
&\qquad\qquad\beta
\end{aligned}
```
$$
X \in \mathbb{R}^{n \times p}
$$

$$
y \in \mathbb{R}^{n}
$$

$$
Z \in \mathbb{R}^{p \times d}
$$

$$
\Theta \in \mathbb{R}^{p \times p}
$$

$$
\beta \in \mathbb{R}^{p}
$$

$$
\tilde{\beta} \in \mathbb{R}^{p+1}
$$

```math
x_i \sim \mathcal{N}(0, \Sigma_x)
```


```math
\Sigma_x = \sigma_x^2 \left(\rho^{|j-k|}\right)_{j,k=1,\dots,p}
```

```math
\theta_{jk} \sim \mathcal{N}\left(\alpha - \|z_j - z_k\|_2^2,\; \sigma_\theta^2\right)
```

```math
z_j \sim \mathcal{N}(0, \sigma_z^2 I_d)
```

```math
y_i \sim \mathcal{N}(
\beta_0 + \beta^\top x_i + \sum_{j \lt k}\theta_{jk}x_{ij}x_{ik},
\; \sigma_y^2
)
```

### 2.2 Parameter prediction

The model likelihood is
```math
p(y, \beta, \Theta, Z \mid x)
=
p(y \mid x, \beta, \Theta)\, p(\Theta \mid Z)\, p(\beta)\, p(Z)
```

with

```math
p(y \mid x, \Theta, \beta)
=
\prod_{i=1}^n
\mathcal{N}(
y_i \mid \beta_0 + \beta^\top x_i + \sum_{j \lt k}\theta_{jk}x_{ij}x_{ik},
\; \sigma_y^2
)
```


```math
p(\Theta \mid Z)
=
\prod_{j \lt k}
\mathcal{N}(
\theta_{jk} \mid \alpha - \|z_j - z_k\|_2^2,
\; \sigma_\theta^2
)
```

```math
p(\beta)
=
\prod_{j=1}^p
\mathcal{N}(\beta_j \mid 0,\sigma_\beta^2)
```

```math
p(Z)
=
\prod_{j=1}^p \prod_{l=1}^d
\mathcal{N}(z_{jl} \mid 0, \sigma_z^2)
```

The full negative log-likelihood is

```math
\mathrm{NLL} = -\log p(y, x, \Theta, Z)

=
-\log p(y, \beta, \Theta, Z \mid x)
-\log p(x)
-\log p(\Theta \mid Z)
-\log p(Z)
```


In our current implementation, we do not explicitly include the priors $p(\beta)$ and $p(z)$ in the optimization objective. This is because priors have large variances with ero mean priors 


Therefore, the training loss is

```math
\mathcal{L}
=
-\log p(y \mid x, \Theta)
-\log p(\Theta \mid Z)
```

### 2.3 Evaluation

- **Simulation data:** compare the estimated interaction matrix with the true interaction matrix

$$
\|\Theta - \hat{\Theta}\|_F^2
$$

- **Real data:** compare the prediction with the observed response

$$
\|y - \hat{y}\|_2^2
$$

## 3. Multivariate Hawkes Processes