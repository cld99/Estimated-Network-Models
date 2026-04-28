# everything here is copied from the ipynb
# functions are unchanged
# demo is slightly changed

import numpy as np
from scipy import optimize

# if any row of the alpha matrix is >= 1 then model with explode
def is_valid_alpha_matrix(alpha):
    for row in alpha:
        if sum(row) >= 1:
            return False

        for i in row:
            if i < 0:
                return False
    return True

def generate_alpha_matrix(dim):
    alpha = np.random.uniform(0, 0.4/dim, size=(dim, dim))
    while(not is_valid_alpha_matrix(alpha)):
        alpha = np.random.uniform(0, 0.4/dim, size=(dim, dim))
        alpha = logistic(alpha)
    return alpha


# function for multivariate hawkes process
def simulation_by_cluster_representation(mu, alpha, beta, time):
    # code taken from https://arxiv.org/pdf/2502.18979 Algorithm 1
    # mu - d dimensional vector
    # alpha - dxd dimensional matrix
    # beta - scalar > 0
    # time - scalar > 0
    # decay is alpha * beta * exp(-beta * (t-t_i)) - this is so we can limit 0<alpha<1, which is easier than 0<alpha<beta

    d = len(mu) # dimension

    # family tree is not part of the paper's algorithm
    family_trees = [{} for _ in range(d)] # timestamp of event: (parent variable, timestamp of parent). parents of immigrants are -1

    # initialization
    T = [[] for _ in range(d)] # list of all immigrants and descendants
    A = [[] for _ in range(d)] # temporary list of ancestors

    # immigrant simulation
    for j in range(d):
        k = np.random.poisson(lam=mu[j]*time) # number of immigrants of type j

        # small_t = [[] for _ in range(d)] # t != T (from the paper's algorithm)
        small_t = []
        for _ in range(k):
            # small_t[j].append(np.random.uniform(low=0, high=time)) # small t is different from big T
            small_t.append(np.random.uniform(low=0, high=time))

        A[j] = small_t #small_t[j]
        T[j] = list(set(T[j] + A[j]))

        for immigrant in T[j]: # assigns the parent of each immigrant to -1
            family_trees[j][immigrant] = (j, -1)

    # helper function for while loop condition (self explanatory)
    def there_exists_at_least_one_j_st_Aj_neq_empty(A):
        for j in range(len(A)):
            if A[j] != []:
                return True
        return False

    # offspring simulation
    while there_exists_at_least_one_j_st_Aj_neq_empty(A):
        O = [[] for _ in range(d)] # offspring initialization
        for j in range(d):
            if A[j] != []:
                for l in range(len(A[j])):
                    for j_prime in range(d):
                        if alpha[j_prime][j] * beta > 0: # alpha * beta
                            k = np.random.poisson(lam=alpha[j_prime][j] * beta) # alpha * beta
                            
                            # finds elapsed time for descendants
                            # small_t = [[] for _ in range(d)]
                            small_t = []
                            for i in range(k):
                                # small_t[j_prime].append(np.random.exponential(beta))
                                small_t.append(np.random.exponential(beta))

                            # adds elapsed time of descendant to the timestamp of its parent (A[j][l] is the parent)
                            a_plus_t = []
                            for i in range(k):
                                a_plus_t.append(A[j][l] + small_t[i]) #small_t[j_prime][i])
                            O[j_prime] = list(set(O[j_prime] + a_plus_t))

                            # assigns the parent of each descendant
                            for descendant in a_plus_t:
                                family_trees[j_prime][descendant] = (j, A[j][l])

                        T[j_prime] = list(set(T[j_prime] + O[j_prime])) # offsprings are added to T
        
        A = O # offspring become ancestors

    # remove events beyond T and sort
    for j in range(d):
        i = 0
        while i < len(T[j]): # removes events that occur after time limit
            if T[j][i] > time:
                T[j].pop(i)
                i -= 1
            i += 1
        T[j] = sorted(T[j])

        # remove events from family_tree if they're beyond T
        for key in list(family_trees[j].keys()):
            if key > time:
                family_trees[j].pop(key)
    
    return T, family_trees

# list of tuples (timestamp, variable that generated the timestamp)
def generate_timestamp_tuple(timestamps):
    timestamp_tuple = []
    for i in range(len(timestamps)):
        for time in timestamps[i]:
            timestamp_tuple.append((time, i))
    return sorted(timestamp_tuple, key=lambda x: x[0])

# function for log likelihood of multivariate hawkes process
def calculate_intensity_matrix(alpha, timestamps, mu, beta, time):
    # code taken from Laub's textbook "The Elements of Hawkes Processes" section 5.3

    l = mu
    for t in timestamps:
        if t[0] < time:
            l = np.add(l, alpha[int(t[1].astype(np.int64))] * beta * np.exp(-beta * (time - t[0]))) # alpha[t[1]] is a 1D matrix
    return l

# function for log likelihood of multivariate hawkes process
def calculate_compensator_matrix(alpha, timestamps, mu, beta, time):
    # code taken from Laub's textbook "The Elements of Hawkes Processes" section 5.3

    l = mu * time
    for t in timestamps:
        if t[0] < time:
            l = np.add(l, (1/beta) * alpha[int(t[1].astype(np.int64))] * beta * (1 - np.exp(-beta * (time - t[0]))))
    return l

# function for log likelihood of multivariate hawkes process
def negative_log_likelihood(alpha, *args):
    # code taken from Laub's textbook "The Elements of Hawkes Processes" section 5.3

    timestamps, mu, beta, time, num_params = args
    timestamp_tuple = generate_timestamp_tuple(timestamps)
    alpha = alpha.reshape((num_params,num_params)) # alpha must be flattened and reshaped because optimize.fmin_l_bfgs_b requires a 1d array

    timestamp_tuple = np.array(timestamp_tuple)
    mu = np.array(mu)
    alpha = np.array(alpha)

    l = 0
    for t in timestamp_tuple:
        intensity = calculate_intensity_matrix(alpha, timestamp_tuple, mu, beta, t[0])
        l += np.log(intensity[int(t[1].astype(np.int64))])
        
    compensator = calculate_compensator_matrix(alpha, timestamp_tuple, mu, beta, time)
    for k in range(len(mu)):
        l -= compensator[k]
    return -l

# Generate latent variables according to Gaussian distribution with specified variance
def generate_latent_Z(p, d, sigma_z):
    # z_j ~ N(0,sigma_z^2 I_d)  Z: shape = (p, d)

    Z = np.random.normal(0, sigma_z, size=(p, d))
    return Z

# Generate Theta matrix with entries distributed according to Gaussian distribution with specified
def generate_theta(Z, alpha, sigma_theta):
    # variance centered at value determined by pairwise distance between latent positions
    # theta_jk = N(alpha - |z_j-z_k|^2,sigma_theta^2)   Theta: shape = (p, p)
    p = Z.shape[0]
    Theta = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            Theta[j, k] = np.random.normal(alpha - np.sum((Z[j] - Z[k])**2), sigma_theta)
    return Theta

# logistic function
def logistic(theta):
    theta_tilde = np.copy(theta)
    for i in range(len(theta)):
        for k in range(len(theta[i])):
            theta_tilde[i][k] = 1 / (1 + np.exp(-theta[i][k])) # np.log(1 + np.exp(theta[i][k]))

    return theta_tilde

# euclidan norm
def euclidean_norm(z1, z2):
    total = 0
    for i in range(len(z1)):
        total += (z1[i] - z2[i]) ** 2
    return total
    # return np.sum((z1 - z2)**2)

# log(P(X | theta_tilde, theta))
def log_p_x_given_thetas(alpha, *args):
    # return -np.log(1 + np.exp(negative_log_likelihood(alpha, *args))) # uses the hawkes log likelihood
    return -negative_log_likelihood(alpha, *args) # hawkes log likelihood

# log(P(theta | Z))
def log_p_theta_given_z(sigma_theta, theta, alph, z, p):
    total = 0
    for j in range(p):
        for k in range(p): # range(p), not range(j,p)
            total -= np.log(np.sqrt(2 * np.pi * sigma_theta))
            
            numerator = (theta[j][k] - (alph - euclidean_norm(z[j], z[k]))) ** 2
            denominator = 2 * sigma_theta
            total -= numerator / denominator
    # print(total)
    return total

# log likelihood = log(P(X | theta_tilde, theta)) + log(P(theta | Z))
def negative_complete_data_log_likelihood_of_theta(theta, *args):
    p, d, z, alph, sigma_theta, theta_tilde, timestamps, mu, beta, time = args
    theta = theta.reshape(p,p) # theta requres reshaping

    args = (timestamps, mu, beta, time, p) # for hawkes
    return -(log_p_x_given_thetas(theta_tilde, *args) + log_p_theta_given_z(sigma_theta, theta, alph, z, p))

# rmse
def rmse(actual, estimated):
    actual = np.array(actual.flatten())
    estimated = np.array(estimated.flatten())

    err = 0
    for i in range(len(actual)):
        err += ((actual[i] - estimated[i]) ** 2) / len(actual)
    return np.sqrt(err)


"""DEMO"""
if __name__ == "__main__":
    """generate hawkes data"""
    # model parameters
    np.random.seed(0)
    mu = [0.4, 0.4] # background intensity
    # alpha = [[0.3, 0.4],
    #         [0.35, 0.45]] # mutual excitation matrix; alpha_12 means that 1 is excited by 2
    alpha = generate_alpha_matrix(len(mu))
    print(alpha)
    beta = 1 # decay; assume beta is the same for all variables
    time = 400 # time
    num_params = len(alpha) # used for flattening/reshaping alpha matrix

    timestamps, _ = simulation_by_cluster_representation(mu, alpha, beta, time) # hawkes

    """generate latent variables and theta"""
    np.random.seed(0)
    p = len(mu) # number of latent nodes to generate
    d = 2 # dimension of latent space
    sigma_z = 1 # variance of latent space generation
    alph = -5 # constant for theta generation; we call this alpha but i don't want to confuse it with the hawkes alpha
    sigma_theta = 0.01 # variance for theta generation

    Z = generate_latent_Z(p, d, sigma_z)
    theta = generate_theta(Z, alph, sigma_theta)
    theta_tilde = logistic(theta)
    t, _ = simulation_by_cluster_representation(mu, theta_tilde, beta, time) # hawkes process using theta_tilde

    """optimizing theta"""
    theta = np.array(theta).flatten() # flatten theta_tilde for optimize.fmin_l_bfgs_b
    args = (p, d, Z, alph, sigma_theta, theta_tilde, t, mu, beta, time) # gathers variables

    # for comparison
    ll = negative_complete_data_log_likelihood_of_theta(theta, *args)
    print("\nNegative complete-data log-likelihood:")
    print(ll)

    initial_guess = [0.5 for _ in theta] # arbitrary initial guess

    # do the optimization, with auto gradient
    print('\nOptimize with scipy-calculated gradient:')
    opt_auto_grad = optimize.fmin_l_bfgs_b(func=negative_complete_data_log_likelihood_of_theta,
                                        args=args,
                                        x0=initial_guess,
                                        approx_grad=True) # theta doesn't have bounds; theta_tilde does
    for res in opt_auto_grad:
        print(res)

    print('\nActual theta values:')
    print(theta)

    theta = theta.reshape((p,p)) # reshape theta_tilde
    estimated_theta = opt_auto_grad[0].reshape((p,p))

    """print errors"""
    err_fro = np.linalg.norm(theta-estimated_theta, ord='fro') # frobenius norm
    err_rmse = rmse(alpha, opt_auto_grad[0])
    print("frobenius:", err_fro)
    print("rmse err :", err_rmse)