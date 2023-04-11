# -*- coding: utf-8 -*-
"""
Created on Tue May  3 07:31:39 2022
@author: Etienne Cheynet
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic


def datenum_to_datetime(datenums):
    """
    datenum_to_datetime(datenums) transform the datenum format into a datetime format

    Input: 
        datenums: array of datenums [1xN]        
    Output
        time: datetime array [1xN]

    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """

    time = pd.to_datetime(datenums-719529, unit='D')
    return time


def frictionVelocity(u, v, w):
    """
       frictionVelocity(u,v,w) computes the friction velocity

    Input Parameters:

    u: array-like, 1-D (along-wind component)
    v: array-like, 1-D (across-wind component)
    w: array-like, 1-D (vertical wind component)

    Output:

    u_star: float, friction velocity
    R: 3x3 array, Reynolds's stress tensor

    Function Description:

    This function takes in the along-wind component (u), the across-wind 
    component (v), and the vertical wind component (w) of the wind velocity 
    as input parameters. It then detrends the input data and computes 
    the variance and covariance of the three components. The friction 
    velocity (u_star) is then computed. The Reynolds's stress tensor (R) 
    is also computed using the variance and covariance values. 
    output of the function is the friction velocity (u_star) 
    and the Reynolds's stress tensor (R).

    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """

    u = signal.detrend(u)  # detrend the along-wind component u
    v = signal.detrend(v)  # detrend the across-wind component v
    w = signal.detrend(w)  # detrend the vertical wind component v

    uu = np.var(u.flatten())
    vv = np.var(v.flatten())
    ww = np.var(w.flatten())

    uv = np.mean(u.flatten()*v.flatten())
    uw = np.mean(u.flatten()*w.flatten())
    vw = np.mean(v.flatten()*w.flatten())

    # Compute the friction velocity
    u_star = (uw**2 + vw**2)**(0.25)

    # Compute Reynolds's stress tensoe
    R = np.array([[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]])

    return u_star, R


def stationaryTest(u, t, Nwin, thres1, thres2):
    """
    Input:

    u: array-like, 1-D (velocity component u, v, or w in m/s)
    t: array-like, 1-D (time vector in seconds)
    Nwin: integer (number of data points for the moving window function)
    thres1: float (threshold value for the absolute relative difference between
                   static mean and "instantaneous mean")
    thres2: float (threshold value for the absolute relative difference between
                   static standard deviation and "instantaneous standard deviation")

    Output:

    err1: float (maximum absolute relative error between static mean and 
                 "instantaneous mean")
    err2: float (maximum absolute relative error between static standard 
                 deviation and "instantaneous standard deviation")
    flag: integer (1 if the time series is found nonstationary, 0 otherwise)

    Function Description:
        The stationaryTest() function assesses the stationarity of a time series
        u using a moving window function for the mean and standard deviation. 
        It tests the first and second order stationarity of the time series.

    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """

    y = pd.Series(u, t)
    err1 = np.max(np.abs(y.rolling(Nwin).mean()/y.mean()-1))
    err2 = np.max(np.abs(y.rolling(Nwin).std()/y.std()-1))
    if err1 > thres1 or err2 > thres2:
        flag = 1 # time series are non-stationary
    else:
        flag = 0 # time series are stationary

    return err1, err2, flag


def Lx(y, meanU, maxLag, dt):
    """
    The function Lx(y, meanU, maxLag, dt) computes the integral length scale 
    using the autocovariance function and an exponential fit to estimate the 
    integral time scale. Taylor's hypothesis of frozen turbulence is applied 
    estimate the integral length scale.
    
    
    Inputs:
    y: [1 x N] or [Nx1] array: velocity component (u, v or w) (m/s)
    meanU: [1 x 1] scalar: mean wind speed (m/s)
    maxLag: [1 x 1] scalar: maximum lag for the crosscovariance function (s)
    dt: [1 x 1] scalar: time step

    Output

    Lu: [1 x 1] scalar: integral length scale (m) associated with the component u, v or w
    C: [1 x M] array : normalized cross-covariance function for positive lag only
    tLag: [1 x M] or [Mx1] array: array for the time lag
    
    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """

    # Exponential decay function for the integral time scale
    def func(t, T):
        C = np.exp(-t/T)
        return C

    N = y.size
    tLag = np.arange(0, N-1, 1)*dt
    indEnd = np.argmin(np.abs(tLag-maxLag))

    y = signal.detrend(y)  # Remove the linea trend!
    C = signal.correlate(y, y)
    C = C/np.max(C)  # Normalzie the cross-covariance function
    indStart = np.argmax(C)
    C = C[indStart:-1]  # Only keep positive lag

    # Fit an exponentiald ecay to the covariance function
    T, dummy = curve_fit(func, tLag[0:indEnd], C[0:indEnd], bounds=(0, 50))

    # Estimate the integral length scale

    Lu = T*meanU
    return Lu, C, tLag


def Ly(u, y):
    """
      Ly(u,y) computes the cross-wind (lateral or vertical) turbulence length 
      scale using the correlation coefficient and an exponential fit. 
      The cross-wind turbulence length scale is a multi-point integral statistics

    Input parameters:

    u: A 2D numpy array of shape (N, M), where N is the number of time steps 
    and M is the number of sensor locations. The array contains the input signal
    measured at each sensor location.
    y: A 1D numpy array of length M, containing the sensor locations.

    Output:

    L: A float value representing the estimated turbulence length scale
    d: A 1D numpy array of length M x M, containing the distances between all 
    pairs of sensor locations.
    Ru: A 1D numpy array of length M x M, containing the correlation coefficient
    between all pairs of detrended input signals.

    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """

    Nsensors = u[1, :].size
    N = u[:, 1].size

    # We want the number of time step in the first dimension
    if N < Nsensors:
        u = u.transpose()
        Nsensors = u[0, :].size
        N = u[:, 0].size

    # detrend your data!
    # Compute the matrix of correlation coefficient
    Ru = np.corrcoef(signal.detrend(u, axis=0).transpose())
    Ru = Ru.flatten()

    # Compute the matrix of distances
    d = np.zeros([Nsensors, Nsensors])
    for ii in range(Nsensors):
        for jj in range(Nsensors):
            d[ii, jj] = np.abs(y[ii] - y[jj])

    # flatten it
    d = d.flatten()

    def func(t, T):
        C = np.exp(-t/T)
        return C

    L, dummy = curve_fit(func, d, Ru, bounds=(0.1, 500))
    return L, d, Ru


def coherence(x, y, N, fs):
    """
        The coherence function computes the co-coherence and quad-coherence 
        between two input signals x and y using Welch's algorithm. 
        The function takes four arguments: x, y, N, and fs.

    Input:
    
        x: [Nx1] array: the first velocity component (u, v, or w) (m/s)
        y: [Nx1] array: the second velocity component (u, v, or w) (m/s)
        N: [1x1] scalar: the window length in number of points for Welch's method
        fs: [1x1] scalar: the sampling frequency of the signals x and y (Hz)
    
    Output:
    
        cocoh: [1xB] array: the co-coherence between x and y for each frequency bin
        quadcoh: [1xB] array: the quad-coherence between x and y for each frequency bin
        f: [1xB] array: the frequency vector associated with the coherence values

        Author: E. Cheynet - UiB - Last modified: 10-03-2023
        """
    x = signal.detrend(x)
    y = signal.detrend(y)

    f, Pxy = signal.csd(x, y, fs, nperseg=N)
    f, Pxx = signal.csd(x, x, fs, nperseg=N)
    f, Pyy = signal.csd(y, y, fs, nperseg=N)

    coh = Pxy/np.sqrt(Pxx*Pyy)
    cocoh = coh.real
    quadcoh = coh.imag

    if f[0] == 0.0:
        f = f[1:-1]
        cocoh = cocoh[1:-1]
        quadcoh = quadcoh[1:-1]

    return cocoh, quadcoh, f


def binSpectra(f, Su, Nb):
    """
        binSpectra(f,Su,Nb) smoothens the estimated power spectral density (PSD) 
    estimates by binning over logarithmic-spaced bins defined by the array newF

        Input:
                f: [Nx1] array:  original frequency vector

                Su: [Nx1] array:  PSD estimate

                Nb: [1 x M] or [Mx1] array:  target bins (as a frequency vector)


        Output
                newF: [1 x B] array: new frequency vector

                newS: [1 x B] array: new PSD estimate


        Author: E. Cheynet - UiB - Last modified: 10-03-2023
        """

    newF0 = np.logspace(np.log10(f[1]*0.8), np.log10(f[-1]*1.1), Nb)
    newSu, newF, ind = binned_statistic(f, Su,
                                        statistic='median',
                                        bins=newF0)
    newF = newF[0:-1]
    newF = newF[~np.isnan(newSu)]
    newSu = newSu[~np.isnan(newSu)]
    return newSu, newF
