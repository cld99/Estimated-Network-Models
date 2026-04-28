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
df = df.groupby(by='AREA') # group by

timestamps = [[] for _ in df['AREA'].max()] # instantiate array with length equal to the number of 
for key, item in df: # for each group
    timestamps[key-1] = np.array(df.get_group(key)['datetime']) # put each group's timestamps into arr
    timestamps[key-1] = sorted(list(set(timestamps[key-1]))) # drops duplicates

# TODO:
# split into 80/20 train/test set (based on max timestamp)
# generate latent variables and theta
# do lbfgs and make prediction on theta
# then use the predicted theta to generate data for 20% of time?
# compare the log likelihood of 
# negative_complete_data_log_likelihood_of_theta, negative_complete_data_log_likelihood_for_adm4
# TODO:
# adm4 shouldn't include the generate theta part; don't include theta given z