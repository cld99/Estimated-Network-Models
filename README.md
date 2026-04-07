# Estimated-Network-Models

In this project, we consider three primary application settings for estimated networks: correlation networks, feature interaction networks, and multivariate Hawkes processes.

If simulation data are used, each setting contains two parts:

1. **Simulation data generation**
2. **Parameter prediction**

Otherwise, we only need **parameter prediction**.

## 1. Correlation Networks

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
p(y, x, \Theta, Z) = p(y \mid x, \Theta)\, p(x)\, p(\Theta \mid Z)\, p(Z)
```

with

```math
p(y \mid x, \Theta)
=
\prod_{i=1}^n
\mathcal{N}(
y_i \mid \beta_0 + \beta^\top x_i + \sum_{j \lt k}\theta_{jk}x_{ij}x_{ik},
\; \sigma_y^2
)
```

```math
p(x)
=
\prod_{i=1}^n
\mathcal{N}(x_i \mid 0, \Sigma_x)
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
p(Z)
=
\prod_{j=1}^p \prod_{l=1}^d
\mathcal{N}(z_{jl} \mid 0, \sigma_z^2)
```

The full negative log-likelihood is

```math
\mathrm{NLL} = -\log p(y, x, \Theta, Z)

=
-\log p(y \mid x, \Theta)
-\log p(x)
-\log p(\Theta \mid Z)
-\log p(Z)
```

Since \(p(x)\) does not depend on the trainable parameters, it is treated as a constant during optimization.

Therefore, the training loss is

```math
\mathcal{L}
=
-\log p(y \mid x, \Theta)
-\log p(\Theta \mid Z)
-\log p(Z)
```

### 2.3 Assessment

- **Simulation data:** compare the estimated interaction matrix with the true interaction matrix

$$
\|\Theta - \hat{\Theta}\|_F^2
$$

- **Real data:** compare the prediction with the observed response

$$
\|y - \hat{y}\|_2^2
$$

## 3. Multivariate Hawkes Processes