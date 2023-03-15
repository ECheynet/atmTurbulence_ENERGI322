# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:16:36 2023

@author: Etienne Cheynet
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import scipy.io
from scipy import signal
import scipy.stats as stats
from scipy.optimize import curve_fit

import os
if not os.path.exists('figures'):
   os.makedirs('figures')
   print(' The folder ''figures'' is created')
else:
    print(' The folder ''figures'' already exists')
   
# %% Load the data from the five masts
data= scipy.io.loadmat('Data_for_excercise_v2.mat')

u = np.squeeze(np.array(data['u']))
v = np.squeeze(np.array(data['v']))
w = np.squeeze(np.array(data['w']))
time = np.squeeze(np.array(data['t']))


# Remove NaNs if necessary
indNaN = np.argwhere(~np.isnan(np.mean(u,1)))
indNaN = indNaN[:,0]
time = time[indNaN]
v = v[indNaN,:]
w = w[indNaN,:]
u = u[indNaN,:]

# Get the number of sensors and time step
Nsensors = u[0,:].size
N = u[:,0].size

# %% Question 2

# %% Question 3

# %% Question 4

# %% Question 5

# %% Question 6

# %% Question 7