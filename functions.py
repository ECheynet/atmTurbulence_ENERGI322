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
  
def frictionVelocity(u,v,w):
    """
       frictionVelocity(u,v,w) computes the friction velocity

    Input:
        u: [1 x N] or [Nx1] array: Along wind component (m/s)
        v: [1 x N] or [Nx1] array: Across wind component (m/s)
        w: [1 x N] or [Nx1] array: Vertical wind component (m/s)

    Output
     u_star: [1 x 1] scalar: friction velocity (m/s)
       
    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """    
     
    u = signal.detrend(u)
    v = signal.detrend(v)
    w = signal.detrend(w)
    
    uu = np.var(u.flatten())
    vv = np.var(v.flatten())
    ww = np.var(w.flatten())

    uv = np.mean(u.flatten()*v.flatten())
    uw = np.mean(u.flatten()*w.flatten())
    vw = np.mean(v.flatten()*w.flatten())
    
    u_star = (uw**2 + vw**2)**(0.25)
    
    R = np.array([[uu, uv, uw],[uv, vv, vw],[uw, vw, ww]])
    return u_star, R
    
    
def stationaryTest(u,t,Nwin, thres1,thres2):
    """
       stationaryTest(u,t,Nwin, thres1,thres2) assesses the stationarity of 
       the time series u using a moving window function for the mean and std of
       the time series. So the function tests the first and second order
       stationarity if the time series.

    Input:
        u: [1 x N] or [Nx1] array:  velocity component (u, v or w) (m/s)
        
        t: [1 x N] or [Nx1] array: time vector (s)
        
        thres1: [1 x 1]: threshold (thres1>0) for the absolute relative 
        difference between static mean and "instantaneous mean"
        
        thres2: [1 x 1]: threshold (thres2>0) for the absolute relative 
        difference between static std and "instantaneous std"

    Output
        err1: [1 x 1] scalar: Maximum absolure relative error between static mean
        and "instantaneous mean"
        
        err2: [1 x 1] scalar: Maximum absolure relative error between static std
        and "instantaneous std"      
        
        flag = 1 if the time series is found nonstationary and flag = 0 otherwise
        
    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """  
    
    y = pd.Series(u, t)
    err1 = np.max(np.abs(y.rolling(Nwin).mean()/y.mean()-1))
    err2 = np.max(np.abs(y.rolling(Nwin).std()/y.std()-1))
    if err1>thres1 or err2>thres2:
        flag=1
    else:
        flag=0
        
    return err1, err2, flag



def Lx(y,meanU,maxLag,dt):
   
    """
       Lx(u,meanU,maxLag,dt) computes the integral length scale using the 
       autocovariance function and an exponential fit to estimate the integral
       time scale. Taylor's hypothesis of frozen turbulence is applied to 
       estimate the integral length scale

    Input:
        y: [1 x N] or [Nx1] array:  velocity component (u, v or w) (m/s)
        
        meanU: [1 x 1] scalar: mean wind speed (m/s)
        
        maxLag: [1 x 1] scalar: max lag for the crosscovariance function (s)
        
        dt: [1 x 1] scalar: time step

    Output
        Lu: [1 x 1] scalar: Integral lengths cale (m) associated with the 
        component u, v or w
        
        C: [1 x M] array : Normalized cross-covariance function for
        positive lag only      
        
        tLag = [1 x M] or [Mx1] array: Array for the time lag
        
    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """  
    
    
    # Exponential decay function for the integral time scale
    def func(t,T):
        C = np.exp(-t/T) 
        return C
    
    N= y.size
    tLag = np.arange(0,N-1,1)*dt
    indEnd = np.argmin(np.abs(tLag-maxLag))
    
    y = signal.detrend(y) # Remove the linea trend!
    C = signal.correlate(y,y)
    C = C/np.max(C) # Normalzie the cross-covariance function
    indStart = np.argmax(C)
    C = C[indStart:-1] # Only keep positive lag
    
    # Fit an exponentiald ecay to the covariance function
    T,dummy = curve_fit(func, tLag[0:indEnd],C[0:indEnd], bounds=(0, 50))
    
    # Estimate the integral length scale
    
    Lu = T*meanU
    return Lu, C, tLag


def Ly(u,y):
    
    """
      Ly(u,y) computes the cross-wind (lateral or vertical) turbulence length 
      scale using the correlation coefficient and an exponential fit. 
      The cross-wind turbulence length scale is a multi-point integral statistics

    Input:
        u: [Nx1] array:  velocity component (u, v or w) (m/s)
        
        y: [1 x M] or [Mx1] array:  position of the sensors (lateral OR vertical)
        

    Output
        Lu: [1 x 1] scalar: Integral lengths cale (m) associated with the 
        component u, v or w
        
        Ru: [1 x B] array : Correlation coefficient cimputed for each sensor
        combination   
        
        d = [1 x B] or [Bx1] array: lateral or vertical distance associated 
        with each sensor combination
        
    Author: E. Cheynet - UiB - Last modified: 10-03-2023
    """  
    
    Nsensors = u[1,:].size
    N = u[:,1].size
    
    # We want the number of time step in the first dimension
    if N<Nsensors:
        u = u.transpose()
        Nsensors = u[0,:].size
        N = u[:,0].size
    
    
    # detrend your data!
    # Compute the matrix of correlation coefficient
    Ru = np.corrcoef(signal.detrend(u,axis=0).transpose())
    Ru = Ru.flatten()

    # Compute the matrix of distances
    d = np.zeros([Nsensors,Nsensors])
    for ii in range(Nsensors): 
        for jj in range(Nsensors): 
            d[ii,jj] = np.abs(y[ii] - y[jj])

    # flatten it
    d = d.flatten()

    def func(t,T):
        C = np.exp(-t/T) 
        return C

    L,dummy = curve_fit(func, d,Ru, bounds=(0.1, 500))
    return L, d, Ru



def coherence(x, y, N,fs):

    """
	coherence(x, y, N,fs) computes the co-coherence and quad-coherence betwen 
	x and y using Welch's algorithm, window length of N points, a sampling 
	frequency fs and 50% overlapping 
	
	Input:
		x: [Nx1] array:  velocity component (u, v or w) (m/s)
		
		y: [Nx1] array:  velocity component (u, v or w) (m/s)
		
		N: [1 x M] or [Mx1] array:  position of the sensors (lateral OR vertical)
		
	
	Output
		cocoh: [1 x B] array: co-coherence between x and y
		
		quadcoh: [1 x B] array: quad-coherence between x and y
		
		f = [1 x B] array : frequency vector
		
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
    
    if f[0]==0.0:
        f = f[1:-1]
        cocoh = cocoh[1:-1]
        quadcoh = quadcoh[1:-1]
   
    return cocoh, quadcoh, f




def binSpectra(f,Su,Nb):
    """
	binSpectra(f,Su,Nb) smoothens the estimated power spectral density (PSD) 
    estimates by binning over logarithmic-spaced bins defined by the array newF
	
	Input:
		f: [Nx1] array:  original frequency vector
		
		Su: [Nx1] array:  PSD estimate
		
		N: [1 x M] or [Mx1] array:  target bins (as a frequency vector)
		
	
	Output
		newF: [1 x B] array: new frequency vector
	
		newS: [1 x B] array: new PSD estimate
		
	
	Author: E. Cheynet - UiB - Last modified: 10-03-2023
	""" 
    
    newF0 = np.logspace(np.log10(f[1]*0.8),np.log10(f[-1]*1.1),Nb)
    newSu,newF,ind = binned_statistic(f, Su, 
                                 statistic='median', 
                                 bins=newF0)
    newF = newF[0:-1]
    newF= newF[~np.isnan(newSu)]
    newSu= newSu[~np.isnan(newSu)]
    return newSu, newF
    
    
    