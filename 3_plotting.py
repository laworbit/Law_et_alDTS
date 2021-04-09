#script to plot results from 1_full_process.py
#science advances figure guidelines. Preferably 2.5, 5.0, or 7.3 inches wide
#and no more than 11.0 inches high. Miminum line width of 0.5 pt. 9 pt and
#bold for e.g. A, B, C, etc. 
#Robert Law, Scott Polar Research Institute, University of Cambridge, 2020. rl491@cam.ac.uk

import os
import sys
import glob
import scipy
import pylab
import seaborn
import datetime
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patch
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator
Polynomial = np.polynomial.Polynomial

os.chdir(os.path.dirname(sys.argv[0]))

from T0_curve_fitting import fit_model, plot_model

#define functions
def func(x, Q, s, T_0):
    k_i = 2.10; # [W m^-1 K^-1] Thermal conductivity of pure ice
    # From Ryser et al. [2014, Thesis, p. 20]
    return ((Q / (4 * np.pi * k_i)) * (1 / (x - s)) + T_0) 

#inputs
file_path_end = 'processed_data/ch1_end_processed.nc' 
file_path_full = 'processed_data/ch1_full_processed.nc'
plot_date = datetime.datetime(2019, 8, 14, 0, 0, 0) #date for plotting in datetime format
av_date = datetime.datetime(2019, 8, 10, 0, 0, 0) #date for averaging where in use. Takes average from av_date to plot_date
bh_depth = 1042.95          #(m) from Sam Doyle BH19c depth email thread
bh_depth_dts = 1062.        #(m) BH19c depth from DTS with refractive index error
z_start = 204.              #(m) z value where cable first enters ice (in non corrected distance)
fs = 8 
close_up_depth = 970        #(m) depth for basal close up to begin from
CTZ_lower = 982             #(m) interpreted depth of bottom of the CTZ
max_T = 1                   #(deg C) maximum temperature for image plot
min_T = -22                 #(deg C) minimum temperature for image plot
pmp_allow = 0.0           #(K) how far to allow pmp away from calculated value to include in temperate zone #WHERE TO DEFINE THIS AS? 0.075 FOR FIGURE, BUT LOWER VALUE WORKS BETTER FOR ANLYSIS. 
equib_cut = 35              #(ind) depth cut to remove top section where cooling is not clearly exponential
fail_depth = (1109.5 - z_start)*(bh_depth/bh_depth_dts)       #(m) at which point did the cable fail?

#input params. This is a bit of an art, check the animated plot to come up with good values for the particular input
equib_start = 1         #index for start of steepest gradient hunt
equib_end = 20          #index for end of gradient hunt
grad_max_pos = 4        #so e.g.1 = start data from 1 after max gradient, -1 = 1 before max gradient etc. 

#constants (i.e. things that definitely won't change unless some seriously strange shit happens)
T0 = 273.15         #(K) 0 degrees C in Kelvin
Ttr = 273.16        #(K) triple point temperature of water
ptr = 611.73        #(Pa) triple point pressure of water
g = 9.81            #(m/s^2) gravitational acceleration
Bs = 1.86           #(K kg mol^-1) constant for pmp calculations from Cuffey and Paterson following Lliboutry (1976)

#parameters (i.e. things that could change)
ccc = 9.14e-8       #(K/Pa) Clausius-Clapeyron constant 
ccc2 = 9.14e-8      #(K/Pa) for water and solute load analysis. This value keeps the pmp line away from from the obvserved at all points
slope = 0.96        #(degrees) slope under borehole
rho_ice = 910.       #(kg/m^3) ice density

#load datasets
ds_end = xr.open_dataset(file_path_end)
ds_full = xr.open_dataset(file_path_full)
#ds_end.tmpw.isel(t = -1).plot(linewidth = 0.7)
#plt.show()
print(ds_end)
print(ds_full)
sys.exit()

#correct depth
ds_end.z.values = (ds_end.z.values - z_start)*(bh_depth/bh_depth_dts)
ds_full.z.values = (ds_full.z.values - z_start)*(bh_depth/bh_depth_dts)

#extract useful part
#ds_end = ds_end.isel(t = -1)

#load data from Sam Doyle
Doyle_df = pd.read_csv('Doyle_data/analog_blue.csv')
Doyle_dt_val = Doyle_df.loc[:,'datetime'].values #datetime values
Doyle_dt_list = list(Doyle_dt_val) #datetime list
Doyle_dts = [datetime.datetime.strptime(x, r'%d/%m/%Y %H:%M') for x in Doyle_dt_list]
Doyle_dt_np = np.array(Doyle_dts) #into np array

#get plotting index
ind_date_Doyle = np.argmax(Doyle_dt_np > plot_date)
av_ind_date_Doyle = np.argmax(Doyle_dt_np > av_date)

#get T values from Doyle_df
T1 = Doyle_df.loc[ind_date_Doyle,"T1"]
T2 = Doyle_df.loc[ind_date_Doyle,"T2"]
T3 = Doyle_df.loc[ind_date_Doyle,"T3"]
T4 = Doyle_df.loc[ind_date_Doyle,"T4"]
T5 = Doyle_df.loc[ind_date_Doyle,"T5"]
T_doyle = np.array([T1, T2, T3, T4, T5])

#means
av_T1 = np.mean(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T1"])
av_T2 = np.mean(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T2"])
av_T3 = np.mean(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T3"])
av_T4 = np.mean(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T4"])
av_T5 = np.mean(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T5"])
av_T_doyle = np.array([av_T1, av_T2, av_T3, av_T4, av_T5])

#stds
std_T1 = np.std(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T1"])
std_T2 = np.std(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T2"])
std_T3 = np.std(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T3"])
std_T4 = np.std(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T4"])
std_T5 = np.std(Doyle_df.loc[av_ind_date_Doyle:ind_date_Doyle,"T5"])
std_T_doyle = np.array([std_T1, std_T2, std_T3, std_T4, std_T5])

#manualy input thermistor depths (T1:T5)
T_depths = np.array([0.28, 1, 3, 5.04, 10.05])
T_depths = bh_depth - T_depths

#set scatter coords
x_scat = T_doyle
y_scat = T_depths

#Clausius-Clapeyron calculation
p_ice = rho_ice*g*ds_full.z.sel(z = slice(0+z_start, bh_depth+z_start))*np.cos(np.deg2rad(slope))
T_pmp_cc = Ttr - ccc*(p_ice - ptr)
T_pmp_cc_w_sol = Ttr - ccc2*(p_ice - ptr) #for water and solute load analysis 

#obtain indicies
depth_ind = np.argmax(ds_end.z.values > bh_depth)
#start_ind = np.argmax(ds_end.z.values > z_start) - 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#figure 1. Time series image plot with close ups for solute load and water content 

#image plot
y = ds_full.z.values
T = ds_full.tmpw.sel(z = slice(0+z_start, bh_depth+z_start)).values
close_up_ind = np.argmax(y > close_up_depth)
temp_min = -0.85
temp_max = -0.75

#create image of temperate zone
pmp_cut = T_pmp_cc
pmp_cut_w_sol = T_pmp_cc_w_sol
pmp_im = np.zeros(np.shape(T)) #pmp image
pmp_im_w_sol = np.zeros(np.shape(T)) #pmp image for water and solute analysis
pmp_ind = np.zeros(pmp_im.shape[1])

for i in range(pmp_im.shape[1]):
    pmp_im[:,i] = pmp_cut
    pmp_im_w_sol[:,i] = pmp_cut_w_sol

pmp_im_w_sol = pmp_im_w_sol - T0 #w_sol means for water and solute analysis
pmp_im = pmp_im - pmp_allow - T0

#find where temperate zone is exceeded
pmp_dif = T - pmp_im
pmp_dif_w_sol = pmp_im_w_sol - T
pmp_ind = np.greater(pmp_dif, np.zeros(np.shape(T)))

matplotlib.rcParams.update({'font.size': fs})
x_lims = mdates.date2num(ds_full.t.values)

fig1 = plt.figure(figsize = (7.3,130/25.4), constrained_layout=True)
gs = fig1.add_gridspec(10,20)
ax1a = fig1.add_subplot(gs[:6,:-1]) #main image
ax1b = fig1.add_subplot(gs[:6,-1]) #T colorbar
ax1c = fig1.add_subplot(gs[6:8,:-1]) #close up temperate zone T
ax1d = fig1.add_subplot(gs[6:8,-1]) #T colorbar for close up
ax1e = fig1.add_subplot(gs[8:10,:-1]) #water content
ax1f = fig1.add_subplot(gs[8:10,-1]) #water content colourbar

#main image
ax1a.imshow(T, vmin=min_T, vmax=max_T, aspect='auto', cmap='viridis',
            extent = [x_lims[0], x_lims[-1], bh_depth, 0])
ax1a.hlines(close_up_depth, x_lims[0], x_lims[-1], colors = 'r', lw=0.75, linestyles='dashed')
#print(T)
#ax1a.contour(T, levels = [-25, -20, -15, -10, -5, 0])
#ax1a.hlines(CTZ_lower, x_lims[0], x_lims[-1], colors = 'white', lw=0.8, linestyles='dashed')
#ax1a.contour(   pmp_ind, levels=[0], colors='white', linewidths=0.75, aspect='auto',
#                extent = [x_lims[0], x_lims[-1], 0, bh_depth])      
ax1a_contours = ax1a.contour(   T, levels=[-25, -20, -15, -10, -5, 0], colors='white', linewidths=0.75, aspect='auto',
                extent = [x_lims[0], x_lims[-1], 0, bh_depth])  
#ax1a.clabel(ax1a_contours, fontsize = fs)   
#ax1a.set_ylim([bh_depth, 0])
#ax1a.xaxis.set_tick_params(rotation=30)
ax1a.xaxis_date()
ax1a.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1a.set_ylabel(' ', fontsize= fs)
days = mdates.DayLocator()
ax1a.xaxis.set_minor_locator(days)
ax1a.set_xticklabels([])

#create T colorbar as seperate plot 
cbar_plot1 = np.zeros((1000, 2))
cbar_plot1[:,0] = np.linspace(max_T, min_T, 1000)
cbar_plot1[:,1] = np.linspace(max_T, min_T, 1000)

im2 = ax1b.imshow(  cbar_plot1, aspect='auto', cmap='viridis',
                    extent = [0, 1, min_T, max_T])

ax1b.set_xticks([])
ax1b.set_yticks(np.arange(min_T, max_T, 1), minor=True)
ax1b.yaxis.set_label_position("right")
ax1b.yaxis.tick_right()
ax1b.tick_params(axis='y', which='minor')
#ax1b.set_ylabel('Temperature ($^\circ$ C)')

#temp close up
ax1c.imshow(T, vmin=temp_min, vmax=temp_max, aspect='auto', cmap='viridis',
            extent = [x_lims[0], x_lims[-1], bh_depth, 0])
#ax1c.contour(pmp_ind, levels=[0], colors='white', linewidths=1, aspect='auto',
#            extent = [x_lims[0], x_lims[-1], 0, bh_depth]) 
ax1c.hlines(CTZ_lower, x_lims[0], x_lims[-1], colors = 'black', lw=0.75, linestyles='dashed')
ax1c.set_ylim([bh_depth, close_up_depth])
#ax1a.xaxis.set_tick_params(rotation=30)
ax1c.xaxis_date()
ax1c.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
days = mdates.DayLocator()
ax1c.xaxis.set_minor_locator(days)
ax1c.set_xticklabels([])

#create T colorbar as seperate plot 
cbar_plot2 = np.zeros((1000, 2))
cbar_plot2[:,0] = np.linspace(temp_max, temp_min, 1000)
cbar_plot2[:,1] = np.linspace(temp_max, temp_min, 1000)

im3 = ax1d.imshow(  cbar_plot2, aspect='auto', cmap='viridis',
                    extent = [0, 1, temp_min, temp_max])

ax1d.set_xticks([])
ax1d.set_yticks(np.arange(temp_min, temp_max, 0.025), minor=True)
ax1d.yaxis.set_label_position("right")
ax1d.yaxis.tick_right()
ax1d.tick_params(axis='y', which='minor')
#ax1d.set_ylabel(' ', fontsize= fs)

#^^^^^^^^^^^^^^^^^^^^^^^^^^
#temperature deviation (n)
n_min = -0.03     #minimum salt concentration for plotting
n_max = 0.03      #maximum salt concentration for plotting
n = pmp_dif_w_sol

ax1e.imshow(n, vmin=n_min, vmax=n_max, aspect='auto', cmap='viridis',
            extent = [x_lims[0], x_lims[-1], bh_depth, 0])
#ax1e.contour(pmp_ind, levels=[0], colors='white', linewidths=1, aspect='auto',
#            extent = [x_lims[0], x_lims[-1], 0, bh_depth])
ax1e.hlines(CTZ_lower, x_lims[0], x_lims[-1], colors = 'black', lw=0.75, linestyles='dashed')
ax1e.set_ylim([bh_depth, close_up_depth])
#ax11a.xaxis.set_tick_params(rotation=30)
ax1e.xaxis_date()
ax1e.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1e.set_xlabel('Date (2019)', fontsize= fs)
days = mdates.DayLocator()
ax1e.xaxis.set_minor_locator(days)

#create salt concentration colorbar as seperate plot 
cbar_plot4 = np.zeros((1000, 2))
cbar_plot4[:,0] = np.linspace(n_max, n_min, 1000)
cbar_plot4[:,1] = np.linspace(n_max, n_min, 1000)

im3 = ax1f.imshow(cbar_plot4, aspect='auto', cmap='viridis',
                    extent = [0, 1, n_min, n_max])

ax1f.set_xticks([])
#ax1f.set_yticks(np.arange(n_min, n_max, 1), minor=True)
ax1f.yaxis.set_label_position("right")
ax1f.yaxis.tick_right()
ax1f.tick_params(axis='y', which='minor')
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
#format = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
#ax1f.yaxis.set_major_formatter(mticker.FuncFormatter(format))
#ax1f.set_ylabel('Salt concentration (mol/kg)', fontsize= fs)
ax1f.set_ylabel('\n ')

#^^^^^^^^^^^^^^^^^^^^
#text labels
fig1.text(0.01, 0.5, 'Depth (m)', va='center', rotation='vertical', fontsize = fs)
fig1.text(0.96, 0.40, 'Temperature ($^\circ$C)', va='center', rotation='vertical', fontsize = fs)
text1 = fig1.text(0.96, 0.135, 'Temperature\ndeviation ($^\circ$C)', va='center', rotation='vertical', fontsize = fs)
text1.set_multialignment('center')
#text2 = fig1.text(0.96, 0.15, 'Solute\nconcentration', va='center', rotation='vertical', fontsize = fs)
#text2.set_multialignment('center')

#fig1.savefig('figures/T_series.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)

plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#figure 5. 3 part profile plot (full, gradient, temperate zone close up) with rectangle cut outs

#EQUILIBRIUM ANALYSIS: FULL LOOP
#get time in seconds
t = ds_full.t.values
xdata = [float((t[i] - t[0])*(1e-9)) for i in range(len(t))]
xdata = np.array(xdata)
xdata = xdata - xdata[0] + 1 #add a second on to prevent 0 value

T_equib = T

#create empty arrays
ice_T_0 = np.squeeze(np.zeros([1, T.shape[0]]))
ice_T_0[:] = np.nan
RMSE_T_0 = np.squeeze(np.zeros([1, T.shape[0]]))
RMSE_T_0[:] = np.nan

#input params. This is a bit of an art, check the animated plot to come up with good values for the particular input
equib_start = 1         #index for start of steepest gradient hunt
equib_end = 20          #index for end of gradient hunt
grad_max_pos = 4        #so e.g.1 = start data from 1 after max gradient, -1 = 1 before max gradient etc. 

#for loop for each depth
#for i in range(T_equib.shape[0]):
print('Running equilibrium loop..')
#y_equib = ds_full.z.isel(z = slice(equib_cut, 3766))
for i in range(equib_cut, 8300):

    #analyse
    ydata = T_equib[i,:]

    #obtain gradient
    ydata_grad = np.gradient(ydata)
    grad_max = np.argmin(ydata_grad[equib_start:equib_end])
    
    #calculate index from where to begin x and y data
    exp_ind = grad_max + grad_max_pos - equib_start

    #set x and y data for the loop
    xdata_loop = xdata[exp_ind:]
    ydata_loop = ydata[exp_ind:]

    #run fitting model
    popt, pcov = scipy.optimize.curve_fit(func, xdata_loop, ydata_loop, p0=(0,0,0))

    #record temperature
    ice_T_0[i] = popt[2]

    #obtain residuals
    Q = popt[0]
    s = popt[1]
    residuals = (ydata_loop - func(xdata_loop, Q, s, ice_T_0[i]))
    RMSE_T_0[i] = np.sqrt(np.mean(residuals**2))

#plot values
y = ds_end.z.values
y_equib = ds_full.z.sel(z = slice(0+z_start, bh_depth+z_start)).values
co_T1 = -17.9
co_T2 = -17.0
co_d1 = 200
co_d2 = 240
a = 0.5
#Clausius-Clapeyron calculation (seperate to figure 1 as easier to keep coords seperate)
p_ice = rho_ice*g*y*np.cos(np.deg2rad(slope))
T_pmp_cc = Ttr - ccc*(p_ice - ptr)

fig5, (ax5a, ax5b, ax5c) = plt.subplots(1,3)
fig5.set_size_inches(7.3,140/25.4)
fig5.subplots_adjust(wspace = 0.23)

T_mean_grad = np.gradient(ds_end.tmpw, ds_end.z)
ax5b.scatter(-0.0815, 105, color='orange')
ax5b.plot(T_mean_grad, y, lw = 0.25, label = 'Temperature gradient', color='k')
ax5b.invert_yaxis()
ax5b.set_xlim([-0.3, 0.3])
ax5b.set_ylim([bh_depth,0]) #orig = [total_depth,0], temp zone = [1300,1100]
ax5b.set_xlabel("Temperature gradient ($^\circ$C m$^-1$)")
ax5b.locator_params(axis='x', nbins=6)
ax5b.grid(True)
ax5b.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
#ax5b.set_yticklabels([])

ax5a.fill_betweenx(y, ds_end.tmpw_25, ds_end.tmpw_975, facecolor='k', alpha=a, edgecolor='k', linewidth=0.0, label=r'95% confidence interval', zorder=4)
ax5a.fill_betweenx(y_equib, ice_T_0 + 0.5*RMSE_T_0, ice_T_0 - 0.5*RMSE_T_0, facecolor='k', alpha=0.8, edgecolor='r', linewidth=0.0, label=r'95% confidence interval', zorder=4)
ax5a.plot(ice_T_0, y_equib, lw=0.5, color='r')
ax5a.plot(ds_end.tmpw, y, lw = 0.5, label = 'Mean Temperature', color='k')
ax5a.scatter(x_scat, y_scat, s=20, facecolors='none', edgecolors='black', zorder=6, label='Thermistor data') 
ax5a.invert_yaxis()
ax5a.set_xlim([-25, 2])
ax5a.set_ylim([bh_depth,0]) #orig = [total_depth,0], temp zone = [1300,1100]
ax5a.set_xlabel("Temperature ($^\circ$C)")
ax5a.set_ylabel("Depth (m)")
ax5a.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax5a.grid(True)

rect1 = patch.Rectangle((co_T1, co_d2), co_T2 - co_T1, co_d1 - co_d2, linewidth=1, facecolor='none', edgecolor = 'k')
ax5a.add_patch(rect1)

rect2 = patch.Rectangle((-6, 880), 6.5, bh_depth - 880, linewidth=1, facecolor='none', edgecolor = 'k')
ax5a.add_patch(rect2)

ax5c.fill_betweenx(y, ds_end.tmpw_25, ds_end.tmpw_975, facecolor='k', alpha=a, edgecolor='k', linewidth=0.0, label=r'95% confidence interval', zorder=4)
ax5c.plot(ds_end.tmpw, y, lw=1, label='Temperature', zorder=3, color='k')
ax5c.scatter(av_T_doyle, y_scat, s=20, facecolors='none', edgecolors='black') 
ax5c.errorbar(av_T_doyle, y_scat, xerr=std_T_doyle, linestyle='None', linewidth=1)
ax5c.invert_yaxis()
ax5c.set_xlim([-6, -0.5]) #orig = [-25, 2], temp zone = [-1.5, -0.5]quit()
ax5c.set_ylim([bh_depth, 880]) #orig = [total_depth,0], temp zone = [1300,1100]
ax5c.plot(T_pmp_cc - T0, y, zorder=1, lw=1, label='T_pmp')
ax5c.set_xlabel("Temperature ($^\circ$C)")
ax5c.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax5c.grid(True)

#rect3 = patch.Rectangle((-0.95, bh_depth - 85), 0.3, bh_depth - 85, linewidth=1, facecolor='none', edgecolor = 'k')
#ax5c.add_patch(rect3)

#cut out
xspacing = 0.1
yspacing = 5
minorlocatorx = MultipleLocator(xspacing)
majorlocatory = MultipleLocator(yspacing)

ax5d = fig5.add_axes([0.225, 0.46, 0.1, 0.21])
ax5d.fill_betweenx(y, ds_end.tmpw_25, ds_end.tmpw_975, facecolor='k', alpha=a, edgecolor='k', linewidth=0.0, label=r'95% confidence interval', zorder=4)
ax5d.fill_betweenx(y_equib, ice_T_0 + 0.5*RMSE_T_0, ice_T_0 - 0.5*RMSE_T_0, facecolor='k', alpha=a, edgecolor='r', linewidth=0.0, label=r'95% confidence interval', zorder=4)
ax5d.plot(ice_T_0, y_equib, lw=0.5, color='r')
ax5d.plot(ds_end.tmpw, y, lw = 0.5, label = 'Mean Temperature', color='k')
ax5d.invert_yaxis()
ax5d.set_xlim([co_T1, co_T2])
ax5d.set_ylim([co_d2,co_d1]) #orig = [total_depth,0], temp zone = [1300,1100]
ax5d.xaxis.set_minor_locator(minorlocatorx)
#ax5d.yaxis.set_minor_locator(majorlocatory)
ax5d.grid(which='major')
ax5d.grid(which='minor')

ax5e = fig5.add_axes([0.73, 0.15, 0.1, 0.43])
ax5e.fill_betweenx(y, ds_end.tmpw_25, ds_end.tmpw_975, facecolor='k', alpha=a, edgecolor='k', linewidth=0.0, label=r'95% confidence interval', zorder=4)
ax5e.plot(ice_T_0, y_equib, lw=0.5, color='r')
ax5e.plot(ds_end.tmpw, y, lw = 0.5, label = 'Mean Temperature', color='k')
ax5e.scatter(av_T_doyle, y_scat, s=20, facecolors='none', edgecolors='black') 
ax5e.errorbar(av_T_doyle, y_scat, xerr=std_T_doyle, linestyle='None', linewidth=1)
ax5e.invert_yaxis()
ax5e.plot(T_pmp_cc - T0, y, zorder=1, lw=1, label='T_pmp')
ax5e.set_xlim([-0.95, -0.65])
ax5e.set_ylim([bh_depth, bh_depth - 85]) #orig = [total_depth,0], temp zone = [1300,1100]
ax5e.xaxis.set_minor_locator(minorlocatorx)
ax5e.yaxis.set_minor_locator(majorlocatory)
ax5e.grid(which='major')
ax5e.grid(which='minor')

#fig5.savefig('figures/T_profile_mean4.png', dpi=600, bbox_inches = 'tight', format = 'png')

plt.show()
#plt.close('all')

#save outdatacd 
data_out = np.column_stack((y, ds_end.tmpw))
#np.savetxt('results/T_profile.txt', data_out)

#clausius clapeyron calculation for each time step
sys.exit()
y_2 = ds_full.z.sel(z = slice(0+z_start, bh_depth+z_start)).values #introducing second y cut to region of interest

#for loop to run over area within temperate zone and calculate clausius clapeyron slope and goodness of fit

#create output array 
rms_out = np.zeros(len(t)) #store root mean square error
r2_out = np.zeros(len(t)) #r squared value 
cc_out = np.zeros(len(t)) #store Clausius Clapeyron

#get index where passes inferred CTZ
t_zone_top = [ n for n,i in enumerate(y_2) if i>982 ][0]

for i in range(len(t)):
    #prepare regression inputs
    #t_zone_top = min([j for j, x in enumerate(pmp_ind[:,i]) if x])
    #print(t_zone_top)
    T_full = np.squeeze(T[:,i])
    t_zone_ind = np.squeeze(pmp_ind[:,i])
    T_t_zone = T_full[t_zone_top:]
    y_t_zone = y_2[t_zone_top:]

    #perform regression
    #m = slope, A0 = intercept
    ymin, ymax = min(y_t_zone), max(y_t_zone)
    pfit, stats = Polynomial.fit(y_t_zone, T_t_zone, 1, full=True, window=(ymin, ymax),
                                                        domain=(ymin, ymax))
    #print('Raw fit results:', pfit, stats, sep='\n')
    A0, m = pfit
    resid, rank, sing_val, rcond = stats
    rms = np.sqrt(resid[0]/len(y_t_zone))

    #perform R2 regressoin
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_t_zone, T_t_zone)

    #print('Fit: T = {:.6f}m + {:.3f}'.format(m, A0),
    #      '(rms residual = {:.4f})'.format(rms))

    #pylab.plot(T_t_zone, y_t_zone, 'o', color='k')
    #pylab.plot(pfit(y_t_zone), y_t_zone, color='k')
    #pylab.xlabel('Temperature $^{o}C$')
    #pylab.ylabel('Depth (m)')
    #plt.gca().invert_yaxis()
    #pylab.show()

    #save outputs
    rms_out[i] = rms
    r2_out[i] = r_value**2
    cc_out[i] = m #convert from K m-1 to K MPa-1

plt.plot(t, -0.8 - 1043*cc_out)
plt.show()

plt.plot(t, (-cc_out/(rho_ice*g))*1e6)

#plt.plot(t, (rms_out/(rho_ice*g))*1e6)
plt.show()

plt.plot(t, r2_out)
plt.show()

#seperate plots
#t_zone_top = min([j for j, x in enumerate(pmp_ind[:,i]) if x])
#print(t_zone_top)
T_full = np.squeeze(T[:,120])
t_zone_ind = np.squeeze(pmp_ind[:,120])
T_t_zone = T_full[t_zone_top:]
y_t_zone = y_2[t_zone_top:]

#perform regression
#m = slope, A0 = intercept
ymin, ymax = min(y_t_zone), max(y_t_zone)
pfit, stats = Polynomial.fit(y_t_zone, T_t_zone, 1, full=True, window=(ymin, ymax),
                                                    domain=(ymin, ymax))
#print('Raw fit results:', pfit, stats, sep='\n')
A0, m = pfit
resid, rank, sing_val, rcond = stats
rms = np.sqrt(resid[0]/len(y_t_zone))

print('Fit: T = {:.6f}m + {:.3f}'.format(m, A0),
      '(rms residual = {:.4f})'.format(rms))

pylab.plot(T_t_zone, y_t_zone, 'o', color='k')
pylab.plot(pfit(y_t_zone), y_t_zone, color='k')
pylab.xlabel('Temperature $^{o}C$')
pylab.ylabel('Depth (m)')
plt.gca().invert_yaxis()
pylab.show()
 














