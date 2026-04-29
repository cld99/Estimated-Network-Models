import pandas as pd
import datetime
import numpy as np
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

df_train = df.loc[df['datetime'] <= max_datetime * 0.8]
df_train = df_train.groupby(by='AREA') # group by
df_test = df.loc[df['datetime'] > max_datetime * 0.8]
df_test = df_test.groupby(by='AREA')

timestams_train = [[] for _ in df_train['AREA'].max()] # instantiate array with length equal to the number of 
for key, item in df_train: # for each group
    timestams_train[key-1] = np.array(df_train.get_group(key)['datetime']) # put each group's timestamps into arr
    timestams_train[key-1] = sorted(list(set(timestams_train[key-1]))) # drops duplicates

timestamps_test = [[] for _ in df_test['AREA'].max()] # instantiate array with length equal to the number of 
for key, item in df_test: # for each group
    timestamps_test[key-1] = np.array(df_test.get_group(key)['datetime']) # put each group's timestamps into arr
    timestamps_test[key-1] = sorted(list(set(timestamps_test[key-1]))) # drops duplicates

# TODO:
# generate latent variables and theta
# do lbfgs and make prediction on theta
# then use the predicted theta to generate data for 20% of time?
# compare the log likelihood of 
# negative_complete_data_log_likelihood_of_theta, negative_complete_data_log_likelihood_for_adm4