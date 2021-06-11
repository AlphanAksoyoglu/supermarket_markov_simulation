import pandas as pd
import os
import sys
from calculation_library import construct_freq_df, generate_markov_matrix

### Get the Raw Data

# Inserting the root folder to current path so that we can access data folder and the calculation_library
# sys.path.insert(0, '../')

days = ['monday','tuesday','wednesday','thursday','friday']

DATA_PATH = './data/market_data/'
OUT_PATH = './data/generated_matrices/'

#collect raw data to a dictionary of dataframes, e.g. df_monday <== monday.csv 
frames = {}
for day in days:
    frames[f'df_{day}'] = pd.read_csv(f'{DATA_PATH}{day}.csv',dtype={'timestamp':str}, sep=';')
    #locals()[f'df_{day}'] = pd.read_csv(f'{DATA_PATH}{day}.csv',dtype={'timestamp':str}, sep=';')

### Construct Frequency Frames

# This one takes a bit of time to run ~seconds, this is due to pd.concat() will take it out of the loop

#construct frequency frames and store in a dictionary with the similar naming convention ff_monday <== monday_freqs, ff_week <== whole week
freq_frames = {}
ff_week = pd.DataFrame()

for day in days:
    
    freq_frames[f'ff_{day}'] = construct_freq_df(frames[f'df_{day}'])
    
    #also create a complete table for the whole week
    #POTENTIAL BUG: concat should be mathematically sound for  matrix calculation, but might need a retest 
    ff_week = pd.concat([ff_week,freq_frames[f'ff_{day}']])


freq_frames;

### Construct Markov Matrices

# Transition Probabilities are from Column -> Row, column sum should add to 1

# Calculate the Markov Transition matrices for each day and store in a dictionary, similar naming convention mm_monday <== monday_markov_matrix, mm_week <== whole week
markov_frames = {}
for day in days:
    markov_frames[f'mm_{day}'] = generate_markov_matrix(freq_frames[f'ff_{day}']) 
# Generate and add the weekly markov matrix here 
mm_week = generate_markov_matrix(ff_week)   
markov_frames['mm_week'] = mm_week

markov_frames;

### Write out the Markov Matrices as csv for re-use

#Finally write the matrices out to data folder
for key, value in markov_frames.items():
    value.to_csv(f'{OUT_PATH}{key}.csv')