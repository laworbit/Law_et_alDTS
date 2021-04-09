#script to process cumulative attenuation DataSet from FO_diff_attenuation.py
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
t1 = datetime.datetime(2019, 7, 9)
t2 = datetime.datetime(2019, 7, 19)
t3 = datetime.datetime(2019, 7, 29)
t4 = datetime.datetime(2019, 7, 8)

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
da = da.resample(t='48H').mean()
da_down = da.sel(z = slice(bh_start, ta_start))

da3 = da3.resample(t='48H').mean()
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

# plt.plot(da_down.isel(t = -1), da_down.z, label='channel 1 down')
# plt.plot(da_up.isel(t = -1), da_up.z, label='channel 1 up')
# plt.plot(da_down3.isel(t = -1), da_down3.z, label='channel 3 down')
# plt.plot(da_up3.isel(t = -1), da_up3.z, label='channel 3 up')
# plt.legend()
# plt.gca().invert_yaxis()
# plt.show()

#plot inputs
#this entire plot has been altered for the revision suplementary figure. See v1-2 for working version to produce the bottom half of figure 2
bh_start = 204.
bh_depth = 1042.95 #(m) from Sam Doyle BH19c depth email thread
bh_depth_dts = 1062. #(m) BH19c depth from DTS with refractive index error
y_axis = (da_down.z - bh_start)*(bh_depth/bh_depth_dts)
fail_depth = fail_depth*(bh_depth/bh_depth_dts)
xlim1 = 0.045
xlim2 = 0.058
lw = 0.8
fs = 8
inset_h = 1.15 #(inch)
inset_w = 1.15
CTZ_upper = 959
CTZ_lower = 982 + 40            #(m) interpreted depth of bottom of the CTZ + 40 for good measure
LGIT = 889 -40      #(m) at which point did the cable fail, with - 40 for good measure
cu_top = 910
cu_bot = 925 

up_an_top = 200
up_an_bot = 250
up_an_l = 0.0110
up_an_r = 0.0135

bot_an_top = 810
bot_an_bot = 1045
bot_an_l = 0.045
bot_an_r = 0.06

matplotlib.rcParams.update({'font.size': fs})

fig1, ax1 = plt.subplots(1, 5)
fig1.set_size_inches(7.3, 70/25.4)
fig1.subplots_adjust(wspace = 0.27)
fig1.subplots_adjust(hspace = 0.27)

ax1[0].plot(da_down.isel(t = 2), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[0].invert_yaxis()
ax1[0].set_ylim([CTZ_lower, LGIT])
ax1[0].set_xlim([xlim1, xlim2])
ax1[0].grid(True)
ax1[0].set_ylabel('Depth (m)', fontsize=fs)
ax1[0].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[0].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[0].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
#date = pd.to_datetime(str(da.t.values[2]))
#date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
#ax1[0].text(0.051, 862, date_string, fontsize=fs)

ax1[1].plot(da_down.isel(t = 6), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[1].invert_yaxis()
ax1[1].set_ylim([CTZ_lower, LGIT])
ax1[1].set_xlim([xlim1, xlim2])
ax1[1].grid(True)
ax1[1].set_yticklabels([])
ax1[1].scatter(0.0046, 105, color='orange')
ax1[1].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[1].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[1].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[7]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
print('date_7')
print(date_string)
#ax1[1].text(0.051, 862, date_string, fontsize=fs)

ax1[2].plot(da_down.isel(t = 10), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[2].invert_yaxis()
ax1[2].set_ylim([CTZ_lower, LGIT])
ax1[2].set_xlim([xlim1, xlim2])
ax1[2].grid(True)
ax1[2].set_yticklabels([])
ax1[2].scatter(0.0046, 105, color='orange')
ax1[2].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[2].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[2].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
#date = pd.to_datetime(str(da.t.values[12]))
#date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
#ax1[2].text(0.051, 862, date_string, fontsize=fs)

ax1[3].plot(da_down.isel(t = 14), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[3].invert_yaxis()
ax1[3].set_ylim([CTZ_lower, LGIT])
ax1[3].set_xlim([xlim1, xlim2])
ax1[3].grid(True)
ax1[3].set_yticklabels([])
ax1[3].scatter(0.0046, 105, color='orange')
ax1[3].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[3].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[3].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[17]))
print('date_17')
print(da_down.t.values)
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
#ax1[3].text(0.051, 862, date_string, fontsize=fs)

ax1[4].plot(da_down.isel(t = 18), y_axis, label='channel 1 down', lw = lw, color='k')
ax1[4].invert_yaxis()
ax1[4].set_ylim([CTZ_lower, LGIT])
ax1[4].set_xlim([xlim1, xlim2])
ax1[4].grid(True)
ax1[4].set_yticklabels([])
ax1[4].scatter(0.0046, 105, color='orange')
ax1[4].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[4].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax1[4].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')

ax1[0].axhspan(cu_top, cu_bot, color = 'gray', alpha = 0.2)
ax1[1].axhspan(cu_top, cu_bot, color = 'gray', alpha = 0.2)
ax1[2].axhspan(cu_top, cu_bot, color = 'gray', alpha = 0.2)
ax1[3].axhspan(cu_top, cu_bot, color = 'gray', alpha = 0.2)
ax1[4].axhspan(cu_top, cu_bot, color = 'gray', alpha = 0.2)

fig1.text(0.5, 0.01, 'Integrated differential attenuation', fontsize = fs, ha='center')

plt.show()
#fig1.savefig('figures/integrated_attenuation_5.png', dpi=600, bbox_inches = 'tight', format='png')

#figure 2

fig2, ax2 = plt.subplots(1, 5)
fig2.set_size_inches(7.3, 70/25.4)
fig2.subplots_adjust(wspace = 0.27)
fig2.subplots_adjust(hspace = 0.27)

ax2[0].plot(da_down.isel(t = 2), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[0].invert_yaxis()
ax2[0].set_ylim([cu_bot, cu_top])
ax2[0].set_xlim([0.051, 0.055])
ax2[0].grid(True)
ax2[0].set_ylabel('Depth (m)', fontsize=fs)
ax2[0].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[0].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[0].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[2]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
print('date_2')
print(date_string)
#ax2[0].text(0.051, 862, date_string, fontsize=fs)

ax2[1].plot(da_down.isel(t = 6), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[1].invert_yaxis()
ax2[1].set_ylim([cu_bot, cu_top])
ax2[1].set_xlim([0.051, 0.055])
ax2[1].grid(True)
ax2[1].set_yticklabels([])
ax2[1].scatter(0.0046, 105, color='orange')
ax2[1].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[1].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[1].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[6]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
print('date_6')
print(date_string)
#ax2[1].text(0.051, 862, date_string, fontsize=fs)

ax2[2].plot(da_down.isel(t = 10), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[2].invert_yaxis()
ax2[2].set_ylim([cu_bot, cu_top])
ax2[2].set_xlim([0.051, 0.055])
ax2[2].grid(True)
ax2[2].set_yticklabels([])
ax2[2].scatter(0.0046, 105, color='orange')
ax2[2].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[2].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[2].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[10]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
print('date_10')
print(date_string)
#ax2[2].text(0.051, 862, date_string, fontsize=fs)

ax2[3].plot(da_down.isel(t = 14), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[3].invert_yaxis()
ax2[3].set_ylim([cu_bot, cu_top])
ax2[3].set_xlim([0.051, 0.055])
ax2[3].grid(True)
ax2[3].set_yticklabels([])
ax2[3].scatter(0.0046, 105, color='orange')
ax2[3].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[3].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[3].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[14]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
print('date_14')
print(date_string)
#ax2[3].text(0.051, 862, date_string, fontsize=fs)

ax2[4].plot(da_down.isel(t = 18), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[4].invert_yaxis()
ax2[4].set_ylim([cu_bot, cu_top])
ax2[4].set_xlim([0.051, 0.055])
ax2[4].grid(True)
ax2[4].set_yticklabels([])
ax2[4].scatter(0.0046, 105, color='orange')
ax2[4].axhline(y = LGIT + 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[4].axhline(y = CTZ_upper, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
ax2[4].axhline(y = CTZ_lower - 40, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[18]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 2)
print('date_18')
print(date_string)

fig2.text(0.5, 0.01, 'Integrated differential attenuation', fontsize = fs, ha='center')
#fig2.savefig('figures/integrated_attenuation.png', dpi=600, bbox_inches = 'tight', format='png')

plt.show()

sys.exit()

fig2, ax2 = plt.subplots(1, 2)
fig2.set_size_inches(7.3, 3.5)

rect5 = patch.Rectangle((up_an_l, up_an_bot), up_an_r - up_an_l, up_an_top - up_an_bot, linewidth=lw, facecolor='none', edgecolor = 'orange')
rect6 = patch.Rectangle((bot_an_l, bot_an_bot), bot_an_r - bot_an_l, bot_an_top - bot_an_bot, linewidth=lw, facecolor='none', edgecolor = 'orange')
rect7 = patch.Rectangle((up_an_l, up_an_bot), up_an_r - up_an_l, up_an_top - up_an_bot, linewidth=lw, facecolor='none', edgecolor = 'orange')
rect8 = patch.Rectangle((bot_an_l, bot_an_bot), bot_an_r - bot_an_l, bot_an_top - bot_an_bot, linewidth=lw, facecolor='none', edgecolor = 'orange')

#easier to put y_axis calculations in line as there are so many
ax2[0].plot(da_up.isel(t = 4), (da_up.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 1 up', lw = lw)
ax2[0].plot(da_down3.isel(t = 4), (da_down3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 down', lw = lw)
ax2[0].plot(da_up3.isel(t = 4), (da_up3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 up', lw = lw)
ax2[0].plot(da_down.isel(t = 4), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[0].invert_yaxis()
ax2[0].set_ylim([1045, 0])
ax2[0].set_xlim([-0.003, 0.0625])
ax2[0].grid(True)
ax2[0].set_ylabel('Depth (m)', fontsize=fs)
ax2[0].add_patch(rect5)
ax2[0].add_patch(rect6)
ax2[0].axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[4]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 4)
ax2[0].text(0.0045, 55, date_string, fontsize=fs)

ax2[1].plot(da_up.isel(t = -1), (da_up.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 1 up', lw = lw)
ax2[1].plot(da_down3.isel(t = -1), (da_down3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 down', lw = lw)
ax2[1].plot(da_up3.isel(t = -1), (da_up3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 up', lw = lw)
ax2[1].plot(da_down.isel(t = -1), y_axis, label='channel 1 down', lw = lw, color='k')
ax2[1].invert_yaxis()
ax2[1].set_ylim([1045, 0])
ax2[1].set_xlim([-0.003, 0.0625])
ax2[1].grid(True)
ax2[1].set_yticklabels([])
ax2[1].add_patch(rect7)
ax2[1].add_patch(rect8)
ax2[1].scatter(0.0046, 105, color='orange')
ax2[1].axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')
date = pd.to_datetime(str(da.t.values[-1]))
date_string = date.strftime('%b %d')+'-'+str(date.day + 4)
ax2[1].text(0.0045, 55, date_string, fontsize=fs)

#add insets
ax2c = inset_axes(ax2[0], width=inset_w, height=inset_h, loc=3)
ax2c.plot(da_down.isel(t = 4), y_axis, label='channel 1 down', lw = lw, color='k')
ax2c.plot(da_up.isel(t = 4), (da_up.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 1 up', lw = lw)
ax2c.plot(da_down3.isel(t = 4), (da_down3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 down', lw = lw)
ax2c.plot(da_up3.isel(t = 4), (da_up3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 up', lw = lw)
ax2c.invert_yaxis()
ax2c.set_ylim([bot_an_bot, bot_an_top])
ax2c.set_xlim([bot_an_l, bot_an_r])
ax2c.grid(True)
ax2c.set_yticklabels([])
ax2c.set_xticklabels([])
ax2c.tick_params(axis='both', which='both', length=0)
ax2c.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')

ax2d = inset_axes(ax2[0], width=inset_w, height=inset_h)
ax2d.plot(da_down.isel(t = 4), y_axis, label='channel 1 down', lw = lw, color='k')
ax2d.plot(da_up.isel(t = 4), (da_up.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 1 up', lw = lw)
ax2d.plot(da_down3.isel(t = 4), (da_down3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 down', lw = lw)
ax2d.plot(da_up3.isel(t = 4), (da_up3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 up', lw = lw)
ax2d.invert_yaxis()
ax2d.set_ylim([up_an_bot, up_an_top])
ax2d.set_xlim([up_an_l, up_an_r])
ax2d.grid(True)
ax2d.set_yticklabels([])
ax2d.set_xticklabels([])
ax2d.tick_params(axis='both', which='both', length=0)

ax2g = inset_axes(ax2[1], width=inset_w, height=inset_h, loc=3)
ax2g.plot(da_down.isel(t = -1), y_axis, label='channel 1 down', lw = lw, color='k')
ax2g.plot(da_up.isel(t = -1), (da_up.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 1 up', lw = lw)
ax2g.plot(da_down3.isel(t = -1), (da_down3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 down', lw = lw)
ax2g.plot(da_up3.isel(t = -1), (da_up3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 up', lw = lw)
ax2g.invert_yaxis()
ax2g.set_ylim([bot_an_bot, bot_an_top])
ax2g.set_xlim([bot_an_l, bot_an_r])
ax2g.grid(True)
ax2g.set_yticklabels([])
ax2g.set_xticklabels([])
ax2g.tick_params(axis='both', which='both', length=0)
ax2g.axhline(y = fail_depth, xmin=0, xmax=1, lw = 0.7, linestyle='dashed', color='gray')

ax2h = inset_axes(ax2[1], width=inset_w, height=inset_h)
ax2h.plot(da_down.isel(t = -1), y_axis, label='channel 1 down', lw = lw, color='k')
ax2h.plot(da_up.isel(t = -1), (da_up.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 1 up', lw = lw)
ax2h.plot(da_down3.isel(t = -1), (da_down3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 down', lw = lw)
ax2h.plot(da_up3.isel(t = -1), (da_up3.z - bh_start)*(bh_depth/bh_depth_dts), label='channel 3 up', lw = lw)
ax2h.invert_yaxis()
ax2h.set_ylim([up_an_bot, up_an_top])
ax2h.set_xlim([up_an_l, up_an_r])
ax2h.grid(True)
ax2h.set_yticklabels([])
ax2h.set_xticklabels([])
ax2h.tick_params(axis='both', which='both', length=0)

fig2.text(0.5, 0.02, 'Integrated differential attenuation', fontsize = fs, ha='center')

#fig2.savefig('figures/integrated_attenuation_sup.png', dpi=600, bbox_inches = 'tight', format='png')
plt.show()




