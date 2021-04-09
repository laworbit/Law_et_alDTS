#script to process cumulative attenuation DataSet from FO_diff_attenuation_v1-0.py
#Robert Law, Scott Polar Research Institute, University of Cambridge, 2020. rl491@cam.ac.uk

import os
import sys
import glob
import pickle
import datetime
import matplotlib
import numpy as np 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt 
import matplotlib.patches as patch
from mpl_toolkits.axes_grid.inset_locator import inset_axes

#change working directory to the location of the python file
os.chdir(os.path.dirname(sys.argv[0]))

#inputs

#channel 1
bh_start = 220.#211.944     #(m) apex of T dip at start of borehole
bh_end = 2319.844#2327.9    #(m) apex of T dip at end of borehole (up)
ta_start = 1266.          #(m) start of the turnaround (using apex)
ta_end = 1271.58            #(m) end of the turnaround (using apex) (up)

#channel 3
bh_start3 = bh_start         #(m) apex of T dip at start of borehole
bh_end3 = bh_end             #(m) apex of T dip at end of borehole (up)
ta_start3 = 1267.39          #(m) start of the turnaround (using apex)
ta_end3 = 1271.46            #(m) end of the turnaround (using apex) (up)

z_start = 204.                      #(m) borehole start depth
fail_depth = 1109.5 - z_start       #(m) at which point did the cable fail?

file_loc = 'processed_data/DataSet_ST_data_ch1.nc'
file_loc3 = 'processed_data/DataSet_ST_data_ch3.nc'
t_start = datetime.datetime(2019, 7, 5)
t_end = datetime.datetime(2019, 8, 14, 1)

#load
ds = xr.open_dataset(file_loc)

ds3 = xr.open_dataset(file_loc3)

#take z and t slice
ds = ds.sel(z=slice(bh_start, bh_end))
ds = ds.sel(t = slice(t_start, t_end))

ds3 = ds3.sel(z=slice(bh_start, bh_end))
ds3 = ds3.sel(t = slice(t_start, t_end))

#calculate differential attenuation
a = 0.5*( np.log(ds['ST'].values[1::,:]/ds['AST'].values[1::,:]) - np.log(ds['ST'].values[0:-1,:]/ds['AST'].values[0:-1,:])
    + np.log(ds['RST'].values[0:-1,:]/ds['RAST'].values[0:-1,:]) - np.log(ds['RST'].values[1::,:]/ds['RAST'].values[1::,:]) )

a3 = 0.5*( np.log(ds3['ST'].values[1::,:]/ds3['AST'].values[1::,:]) - np.log(ds3['ST'].values[0:-1,:]/ds3['AST'].values[0:-1,:])
    + np.log(ds3['RST'].values[0:-1,:]/ds3['RAST'].values[0:-1,:]) - np.log(ds3['RST'].values[1::,:]/ds3['RAST'].values[1::,:]) )

#cumulative sum and concatenate
a_cum = np.cumsum(a,axis=0)
a_cum = np.concatenate((a_cum, np.ones((1,len(ds.t.values)))*a_cum[-1,:] ), axis = 0)

a_cum3 = np.cumsum(a3,axis=0)
a_cum3 = np.concatenate((a_cum3, np.ones((1,len(ds3.t.values)))*a_cum3[-1,:] ), axis = 0)


da = xr.DataArray( a_cum, 
                        dims = ('z', 't'),
                        coords= {   'z': ds.z.values,
                                    't': ds.t.values})

da3 = xr.DataArray( a_cum3, 
                        dims = ('z', 't'),
                        coords= {   'z': ds3.z.values,
                                    't': ds3.t.values})

#resample
da = da.resample(t='96H').mean()
da_down = da.sel(z = slice(bh_start, ta_start))

da3 = da3.resample(t='96H').mean()
da_down3 = da3.sel(z = slice(bh_start3, ta_start3))

#vertical reflection of lower half
da_up = da.sel(z = slice(ta_end, bh_end))
da_up.values = 0 - da_up.values

da_up3 = da3.sel(z = slice(ta_end3, bh_end3))
da_up3.values = 0 - da_up3.values

#horizontal reflection
da_up.z.values = max(da_up.z.values) - da_up.z.values

da_up3.z.values = max(da_up3.z.values) - da_up3.z.values

#horizontal shift
da_up.values = da_up.values[:] + (da_down.values[0,:] - da_up.values[-1,:])

da_up3.values = da_up3.values[:] + (da_down3.values[0,:] - da_up3.values[-1,:])

#vertical shift
da_up.z.values = da_up.z.values + (da_down.z.values[0] - da_up.z.values[-1])

da_up3.z.values = da_up3.z.values + (da_down3.z.values[0] - da_up3.z.values[-1])

#plots
#plot inputs
bh_start = 204.
bh_depth = 1042.95 #(m) from Sam Doyle BH19c depth email thread
bh_depth_dts = 1062. #(m) BH19c depth from DTS with refractive index error
y_axis = (da_down.z - bh_start)*(bh_depth/bh_depth_dts)
fail_depth = fail_depth*(bh_depth/bh_depth_dts)
lw = 0.8
fs = 8
inset_h = 1.15 #(inch)
inset_w = 1.15

up_an_top = 200
up_an_bot = 250
up_an_l = 0.0110
up_an_r = 0.0135

bot_an_top = 810
bot_an_bot = 1045
bot_an_l = 0.045
bot_an_r = 0.06

matplotlib.rcParams.update({'font.size': fs})

fig1, ax1 = plt.subplots(1, 2)
fig1.set_size_inches(7.3, 3.5)

rect1 = patch.Rectangle((up_an_l, up_an_bot), up_an_r - up_an_l, up_an_top - up_an_bot, linewidth=lw, facecolor='none', edgecolor = 'k')
rect2 = patch.Rectangle((bot_an_l, bot_an_bot), bot_an_r - bot_an_l, bot_an_top - bot_an_bot, linewidth=lw, facecolor='none', edgecolor = 'k')
rect3 = patch.Rectangle((up_an_l, up_an_bot), up_an_r - up_an_l, up_an_top - up_an_bot, linewidth=lw, facecolor='none', edgecolor = 'k')
rect4 = patch.Rectangle((bot_an_l, bot_an_bot), bot_an_r - bot_an_l, bot_an_top - bot_an_bot, linewidth=lw, facecolor='none', edgecolor = 'k')

ax1[0].plot(da_down.isel(t = 4), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[0].invert_yaxis()
ax1[0].set_ylim([1045, 0])
ax1[0].set_xlim([-0.003, 0.0625])
ax1[0].grid(True)
ax1[0].set_ylabel('Depth (m)', fontsize=fs)
ax1[0].add_patch(rect1)
ax1[0].add_patch(rect2)
ax1[0].axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[4]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 4)
ax1[0].text(0.0045, 55, date_string, fontsize=fs)

ax1[1].plot(da_down.isel(t = -1), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[1].invert_yaxis()
ax1[1].set_ylim([1045, 0])
ax1[1].set_xlim([-0.003, 0.0625])
ax1[1].grid(True)
ax1[1].set_yticklabels([])
ax1[1].add_patch(rect3)
ax1[1].add_patch(rect4)
ax1[1].scatter(0.0046, 105, color='orange')
ax1[1].axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[-1]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 4)
ax1[1].text(0.0045, 55, date_string, fontsize=fs)

#add insets
ax1c = inset_axes(ax1[0], width=inset_w, height=inset_h, loc=3)
ax1c.plot(da_down.isel(t = 4), y_axis, label='channel 1 down', lw = lw, color='k')
ax1c.invert_yaxis()
ax1c.set_ylim([bot_an_bot, bot_an_top])
ax1c.set_xlim([bot_an_l, bot_an_r])
ax1c.grid(True)
ax1c.set_yticklabels([])
ax1c.set_xticklabels([])
ax1c.tick_params(axis='both', which='both', length=0)
ax1c.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')

ax1d = inset_axes(ax1[0], width=inset_w, height=inset_h)
ax1d.plot(da_down.isel(t = 4), y_axis, label='channel 1 down', lw = lw, color='k')
ax1d.invert_yaxis()
ax1d.set_ylim([up_an_bot, up_an_top])
ax1d.set_xlim([up_an_l, up_an_r])
ax1d.grid(True)
ax1d.set_yticklabels([])
ax1d.set_xticklabels([])
ax1d.tick_params(axis='both', which='both', length=0)

ax1g = inset_axes(ax1[1], width=inset_w, height=inset_h, loc=3)
ax1g.plot(da_down.isel(t = -1), y_axis, label='channel 1 down', lw = lw, color='k')
ax1g.invert_yaxis()
ax1g.set_ylim([bot_an_bot, bot_an_top])
ax1g.set_xlim([bot_an_l, bot_an_r])
ax1g.grid(True)
ax1g.set_yticklabels([])
ax1g.set_xticklabels([])
ax1g.tick_params(axis='both', which='both', length=0)
ax1g.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')

ax1h = inset_axes(ax1[1], width=inset_w, height=inset_h)
ax1h.plot(da_down.isel(t = -1), y_axis, label='channel 1 down', lw = lw, color='k')
ax1h.invert_yaxis()
ax1h.set_ylim([up_an_bot, up_an_top])
ax1h.set_xlim([up_an_l, up_an_r])
ax1h.grid(True)
ax1h.set_yticklabels([])
ax1h.set_xticklabels([])
ax1h.tick_params(axis='both', which='both', length=0)

fig1.text(0.5, 0.02, 'Integrated differential attenuation', fontsize = fs, ha='center')

plt.show()
#fig1.savefig('figures/integrated_attenuation.png', dpi=600, bbox_inches = 'tight', format='png')





