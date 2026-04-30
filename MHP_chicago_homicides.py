import pandas as pd
import datetime
import numpy as np
from scipy import optimize
from MultivariateHawkesProcess import *
from MHP_ADM4_trials import *

def to_month_int(row):
    if row['INJMONTH'] == 'January':
        return 1
    elif row['INJMONTH'] == 'February':
        return 2
    elif row['INJMONTH'] == 'March':
        return 3
    elif row['INJMONTH'] == 'April':
        return 4
    elif row['INJMONTH'] == 'May':
        return 5
    elif row['INJMONTH'] == 'June':
        return 6
    elif row['INJMONTH'] == 'July':
        return 7
    elif row['INJMONTH'] == 'August':
        return 8
    elif row['INJMONTH'] == 'September':
        return 9
    elif row['INJMONTH'] == 'October':
        return 10
    elif row['INJMONTH'] == 'November':
        return 11
    elif row['INJMONTH'] == 'December':
        return 12

"""configure data"""
df = pd.read_stata('homicides_in_chicago.dta')

# INJDTE doesn't include any info (which also means taht INJTIME is redundant)
# police district changes by year, police area changes less; so only use AREA and not DISTRICT
cols = ['INJYEAR', 'INJMONTH', 'AREA', 'GANG']
df = df[cols] # drops unecessary columns
df = df[df.GANG == 'Yes'] # get only the gang-related
df['INJMONTH'] = df.apply(to_month_int, axis=1); # convert month str to int
df['datetime'] = df.apply(lambda x: datetime.datetime(year=int(x['INJYEAR']), month=int(x['INJMONTH']), day=1), axis=1) # convert year/month to datetime
df['datetime'] = df.apply(lambda x: x['datetime'] - datetime.datetime(year=64, month=1, day=1), axis=1) # convert datetime to timedelta from 1/1/1964
df['datetime'] = df['datetime'].dt.days # convert timedelta to int
df['AREA'] = df.apply(lambda x: int(x['AREA'][len(x['AREA'])-1:]), axis=1) # convert police area to int
df = df[['datetime', 'AREA']] # drops unnecessary columns
max_datetime = int(df['datetime'].max())

"""separate into train and test set"""
percent_for_train_set = 0.8
df_train = df.loc[df['datetime'] <= max_datetime * percent_for_train_set]
df_train = df_train.groupby(by='AREA') # group by
df_test = df.loc[df['datetime'] > max_datetime * percent_for_train_set]
df_test = df_test.groupby(by='AREA')

num_params = int(df['AREA'].max())

timestamps_train = [[] for _ in range(num_params)] # instantiate array with length equal to the number of 
for key, item in df_train: # for each group
    timestamps_train[key-1] = np.array(df_train.get_group(key)['datetime']) # put each group's timestamps into arr
    timestamps_train[key-1] = sorted(list(set(timestamps_train[key-1]))) # drops duplicates

timestamps_test = [[] for _ in range(num_params)] # instantiate array with length equal to the number of 
for key, item in df_test: # for each group
    timestamps_test[key-1] = np.array(df_test.get_group(key)['datetime']) # put each group's timestamps into arr
    timestamps_test[key-1] = sorted(list(set(timestamps_test[key-1]))) # drops duplicates

num_events = 0
for k in timestamps_train:
    num_events += len(k)
for k in timestamps_test:
    num_events += len(k)
print("total number of events:", num_events)

"""optimize"""
np.random.seed(0)
mu = [0.2/num_params for _ in range(num_params)]
beta = 1
time = max_datetime * percent_for_train_set

p = num_params
d = 2
sigma_z = 1
alph = -5
sigma_theta = 0.5

Z = generate_latent_Z(p, d, sigma_z)
theta = generate_theta(Z, alph, sigma_theta)
theta_tilde = logistic(theta)

theta = np.array(theta).flatten()
args = (p, d, Z, alph, sigma_theta, theta_tilde, timestamps_train, mu, beta, time)
args_for_adm4 = (0.095, 0.047, timestamps_train, mu, beta, time, p)

initial_guess = [0.5 for _ in theta] # arbitrary initial guess
bounds = []
for _ in range(len(theta)):
    bounds.append((0.0001, 1-0.0001)) # bounds is (0,1), exclusive
bounds = np.array(bounds)

opt_auto_grad = optimize.fmin_l_bfgs_b(func=negative_complete_data_log_likelihood_of_theta,
                                       args=args,
                                       x0=initial_guess,
                                       approx_grad=True)
adm4_optimize = optimize.fmin_l_bfgs_b(func=adm4_nll,
                                       args=args_for_adm4,
                                       x0=initial_guess,
                                       bounds=bounds,
                                       approx_grad=True)

estimated_theta = opt_auto_grad[0].reshape((p,p))
estimated_theta_tilde = logistic(estimated_theta)
estimated_adm4 = adm4_optimize[0].reshape((p,p))

# for comparison
ll = negative_complete_data_log_likelihood_of_theta(theta, *args)
print("\nNegative complete-data log-likelihood:")
print(ll)

print('\nOptimize with scipy-calculated gradient:')
for res in opt_auto_grad:
    print(res)

print('\nAfter logistic:')
print(estimated_theta_tilde)

"""compare on held-out data"""
print("\nPercent for train set:", percent_for_train_set)
time = max_datetime * (1-percent_for_train_set)
args = (p, d, Z, alph, sigma_theta, estimated_theta_tilde, timestamps_test, mu, beta, time)
ll = negative_complete_data_log_likelihood_of_theta(estimated_theta, *args)
print("\nNegative log-likelihood of model:")
print(ll)

args_for_adm4 = (0.095, 0.047, timestamps_test, mu, beta, time, p)
ll_adm4 = adm4_nll(estimated_adm4, *args_for_adm4)
print("\nNegative log-likelihood of ADM-4:")
print(ll_adm4)