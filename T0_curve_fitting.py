"""
17 October 2018
Sam Doyle
This scripts estimates the undisturbed ice temperature T_0 by (non-linear) 
fitting an exponential decay curve to the equilibriation phase of cooling, 
which is after the rapid cooling phase.
See Ryser et al. [2014, Thesis, p.20]
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utility as ut


" Define non-linear fitting function "
def fit_model(df, idx, i, breakthrough_time):  
    
    df = ut.add_doy_column(df)
    df = df.set_index('DOY')
    
    xfull = df.index - ut.dt2doy(breakthrough_time)
    
    
    idx_str = 'T' + str(i+1) 
    yfull = df[idx_str].values
    
    xdata = xfull[idx[i]:] # Time since drill reached the bed [Days] 
    ydata = df[idx_str].values[idx[i]:] # Temp [Celcius]
    
    ydata = ydata[np.where(~np.isnan(ydata))] # Remove NaNs
    xdata = xdata[np.where(~np.isnan(ydata))]
    xdata = np.asarray(xdata) # Convert xdata to np array
    
    def func(x, Q, s, T_0):
        k_i = 2.10; # [W m^-1 K^-1] Thermal conductivity of pure ice
        # From Ryser et al. [2014, Thesis, p. 20]
        return ((Q / (4 * np.pi * k_i)) * (1 / (x - s)) + T_0) 
    
    popt, pcov = curve_fit(func, xdata, ydata, p0=(0,0,0))
    
    Q = popt[0]
    s = popt[1]
    T_0 = popt[2]
    
    xmodel = np.arange(int(xdata[0]), int(xdata[0])+200, 0.05)
    ymodel = func(xmodel, Q, s, T_0)
    
    residuals = (ydata - func(xdata, Q, s, T_0))
    RMSE = np.sqrt(np.mean(residuals**2))
    #print('RMSE = ' + str(np.round(RMSE,5)) + ' Celcius')
    return Q, s, T_0, xdata, ydata, xmodel, ymodel, residuals, RMSE, \
        xfull, yfull
    
def plot_model(profile, xfull, yfull, xdata, ydata, xmodel, ymodel, \
        RMSE, T_0, T_no, xlims, figname): # Plot model
    #f = plt.figure(figsize=(15,8))
    f = plt.figure(figsize=(ut.cm2inch(14.4, 14.4)))
    ax = plt.gca()
    #plt.plot(xdata, ydata,'.k', label='Data')
    plt.plot(xfull, yfull,'.', label='Data', color=[0.2, 0.2, 0.2])
    plt.plot(xmodel, ymodel,'g-', label='Model', linewidth=2)
    
    plt.axhline(T_0, color='c', label='$T_0$')
    plt.axhline(profile.Tm_air[T_no], color='r', label='$T_{m} (air)$')
    plt.axhline(profile.Tm_pure[T_no], color='b', label='$T_{m} (pure)$')
    plt.axhline(profile.Tm_luthi[T_no], color='m', label='$T_{m} (luthi)$')
    #    plt.text(xdata[-1]-3, (ydata[0] + (ydata[-1] - ydata[0])/2)+1, 
    #                 '$T_{0}$ = $\minus$' + str(np.round(abs(T_0),3)) + '$^{\circ}$C')
    #    plt.text(xdata[-1]-3, ((ydata[0] + (ydata[-1] - ydata[0])/2)), 
    #                 'RMSE =' + str(np.round(RMSE*1000,3))[0:4] + ' mK')
    #    plt.text(xdata[-1]-3, profile.Tm_air[T_no] + 0.75, '$T_m$')
    
    plt.legend(loc='upper right', fontsize=12, numpoints=1)
    plt.xlabel('Time since drill reached the bed (Days)')
    plt.ylabel('Temperature ($^{\circ}$C)', labelpad=0)
    plt.xlim(xlims)        
    
    plt.title('T' + str(T_no))
    f.savefig('./fig/' + figname + str(T_no) + '.png')
    return f, ax