import sys
# sys.path.insert(0, '../')
from calculation_library import Customer, SuperMarket
import pandas as pd
import datetime
import numpy as np

### We get our desired transition matrix

MATRIX_PATH = './data/generated_matrices/'
MATRIX_FILENAME = 'mm_monday.csv'

matrix_monday = pd.read_csv(f'{MATRIX_PATH}{MATRIX_FILENAME}', index_col = 0).T
matrix_monday = np.array(matrix_monday)

### We initiate a market object from the class, it takes the transition matrix as a parameter

market = SuperMarket(matrix_monday)

### The Market has a simulate method which simulates customers

# We pass the initial (total) number of customers, the open time and the close time of the market

# market.simulate(3,'8:00','15:00')

# While the market is running it has a `current_state` and a ``total_state` attribute, `current_state` is the state of the market at the current time and will be useful for an animation if we do that. `total_state` keeps track of the complete market state over time

# The number of customers in the market is constant if `n` customers leave we insert `n` customers

# The `current_state` and `total_state` is kept as numpy arrays, this makes it a little tedius but faster. We can also keep these as data frames but there is a risk will be slower

### After a simulation finishes, we can access the `total_state` and see what happened over time, there is also a results() method which returns the `total_state` as a dataframe

a = market.total_state

market.results()