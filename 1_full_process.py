#script to complete full processing of DTS files using matching sections. 
#This uses xarray interpolation method to produce a consitently sized output.
#Robert Law, Scott Polar Research Institute, University of Cambridge, 2020. rl491@cam.ac.uk

import os
import sys
import glob
import pickle
import datetime
import numpy as np 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from dtscalibration import read_silixa_files, DataStore

#change working directory to the location of the python file
os.chdir(os.path.dirname(sys.argv[0]))

#inputs
file_dir1 = "channel_1" #input file path 1 (channel 1)
file_dir2 = "channel_3" #input file path 2 (channel 3)
export_dir = "processed_data" #export processed data
export_file = 'ch1_full_processed.nc' #processed stokes and cummulative attenuation data 
avg_time = '8H'
avg_time_float = 8
z_num = 11078 #number of z valus to interpolate to. It's convinient down the line if this is an even number
usr_gamma = (476.53, 0)         

#CHANNEL 1 values below
bh_down_ref = 211.94          #(m) downwards start for borehole, determined from apex of temperature drop just after entrance
bh_up_ref = 2327.90           #(m) upwards end of borehole, determined from apex of temperature drop just before exit
bh_down_splice = 1267.88      #(m) downwards end of borehole, determined as point where temperature begins to rise (even slightly) before splice
bh_splice_mid = 1269.41       #(m) determined as the location of peak temperature disturbance from the splice
bh_up_splice = 1270.92        #(m) upwards start of borehole, determined as point where temperature begins to rise (even slightly) after splice
bh_depth = 1042.95            #(m) from Sam Doyle BH19c depth email thread
bh_depth_dts = 1062.          #(m) BH19c depth from DTS with refractive index error
start_cut = 10.                         #(m) cut outside the DTS so connector losses do not need to be accounted for
end_cut = bh_down_splice - 0.5              #(m) cut at the surface
start_bh = 204.5                        #(m) depth at which borehole begins
temp_zone_loc = (1000.*(bh_depth_dts/bh_depth) + start_bh, 1010.*(bh_depth_dts/bh_depth) + start_bh)        #location of temperate zone
matching_section = (bh_up_splice + bh_down_splice - temp_zone_loc[0], bh_up_splice + bh_down_splice - temp_zone_loc[1]) #set matching section to = reference section 

#constants (i.e. if these things change we will have some deep existential issues on our hands)
T0 = 273.15         #(K) 0 degrees C in Kelvin
Ttr = 273.16        #(K) triple point temperature of water
ptr = 611.73        #(Pa) triple point pressure of water
g = 9.81            #(m/s^2) gravitational acceleration

#parameters (i.e. things that could change)
ccc = 9.14e-8       #(K/MPa) Clausius-Clapeyron constant 
slope = 0           #(degrees) slope under borehole
rho_ice = 910       #(kg/m^3) ice density 

#obtain paths for directories within channel directorys and create export path
directs = [x[0] for x in os.walk(file_dir1)]
directs = sorted(directs[1:]) #remove the first value as this is just the parent directory and sort 
export_path = os.path.join(export_dir, export_file)

#convert to slice
temp_zone_loc = slice(temp_zone_loc[0], temp_zone_loc[1])   
matching_section = slice(matching_section[1], matching_section[0]+0.15) 

sections = {
    'pmpTemperature':       [temp_zone_loc]}    #pressure melting point 

#labels
st_label = 'st'
ast_label = 'ast'
rst_label = 'rst'
rast_label = 'rast'

directs = directs[1:-1]

for i, directory in enumerate(directs):

    #print out
    files = sorted(glob.glob(os.path.join(directory, '*.xml')))
    now = datetime.datetime.now()
    print(now.strftime("%H:%M:%S-"), 'Processing', str(len(files)), 'files in directory', directory, '...')

    #create datastore
    ds_in = read_silixa_files(
        directory = directory,
        timezone_netcdf='UTC',
        file_ext='*.xml')

    #obtain channel information
    #xml_name = ds_in['filename'].values[0]
    #channel = int(xml_name[8])

    #cut to section of interest
    ds_in = ds_in.sel(x = slice(start_cut, end_cut))

    #extract z values
    #array dimensions: (z,y,x) = (data_fields, z, t)
    z_values = ds_in.x.values

    #Clausius-Clapeyron calculation
    p_ice = rho_ice*g*((z_values - start_bh)*(bh_depth/bh_depth_dts))*np.cos(np.deg2rad(slope))
    T_pmp_cc = Ttr - ccc*(p_ice - ptr)
    T_pmp_cc_rz = T_pmp_cc[np.argmax(z_values > ((temp_zone_loc.start+temp_zone_loc.stop)/2))] - T0 #reference zone temperature

    #create time series for ref_pmp_T
    nd_cols = np.size(ds_in['st'].values,1)    #columns in array
    ds_in['pmpTemperature'] = (('time',), np.ones(nd_cols)*T_pmp_cc_rz)

    #set sections
    ds_in.sections = sections

    #resample before calibration
    ds_in = ds_in.resample_datastore(how='mean', time=avg_time, keep_attrs=True)

    ds_in.acquisitionTime = avg_time_float*60*60

    #extract t values
    t_values = ds_in.time.values

    #interpolate if needed 
    if len(z_values) != z_num:

        #create z values for interpolation
        zi = np.linspace(z_values[0], z_values[-1], z_num)

        #interp datastore
        ds_in = ds_in.interp(x = zi, method='slinear')

        #set z_values for outgoing dataset
        z_values = zi

    #obtain variance
    st_var, resid = ds_in.variance_stokes(st_label=st_label)
    ast_var, _ = ds_in.variance_stokes(st_label=ast_label)
    rst_var, _ = ds_in.variance_stokes(st_label=rst_label)
    rast_var, _ = ds_in.variance_stokes(st_label=rast_label)

    now = datetime.datetime.now()
    print(now.strftime("%H:%M:%S-"), 'Performing calibration...')

    #perform calibration
    ds_in.calibration_double_ended(
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        method='wls',
        solver='sparse',
        fix_gamma=usr_gamma)
        #transient_asym_att_x=[bh_splice_mid],
        #matching_sections=[(temp_zone_loc, matching_section, True)])

    #confidence intervals
    ds_in.conf_int_double_ended(
        p_val='p_val',
        p_cov='p_cov',
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        store_tempvar='_var',
        conf_ints=[2.5, 50., 97.5],
        mc_sample_size=2000)
        #store_ta='talpha')  # <- choose a much larger sample size

    print('creating outgoing array...')
    print(ds_in)

    #creat DataSet. Setting z coord here results in lots of errors, so it is input after the loop
    print('tmpw')
    tmpw = ds_in.tmpw.values
    print('tmpw_25')
    tmpw_25 = ds_in.tmpw_mc.isel(CI=0).values
    print('tmpw_central')
    tmpw_central = ds_in.tmpw_mc.isel(CI=1).values
    print('tmpw_975')
    tmpw_975 = ds_in.tmpw_mc.isel(CI=2).values

    #remove nans and outlying values
    print('remove nans')
    tmpw[np.isnan(tmpw)] = 30
    tmpw_25[np.isnan(tmpw_25)] = 30
    tmpw_central[np.isnan(tmpw_975)] = 30
    tmpw_975[np.isnan(tmpw_975)] = 30

    print('remove outliers')
    tmpw[(tmpw > 20) | (tmpw < -30)] = 30
    tmpw_25[(tmpw_25 > 20) | (tmpw_25 < -30)] = 30
    tmpw_central[(tmpw_central > 20) | (tmpw_central < -30)] = 30
    tmpw_975[(tmpw_975 > 20) | (tmpw_975 < -30)] = 30

    ds = xr.Dataset(
                data_vars = {   'tmpw'  : (('z', 't'), tmpw),
                                'tmpw_25':(('z', 't'), tmpw_25),
                                'tmpw_central':(('z', 't'), tmpw_central),
                                'tmpw_975':(('z', 't'), tmpw_975)},
                coords={'t' : t_values})

    print(ds)
    #ds['tmpw_central'].isel(t=-1).plot(linewidth=0.7, figsize=(12, 8))
    #plt.show()

    #concatenate DataArray if not first loop iteration
    if i == 0:

        ds_out = ds

    else:        

        print(ds_out)
        print(ds)
        #ds_out = xr.merge([ds_out, ds])
        ds_out = xr.concat([ds_out, ds], dim =  't')

#put in z coords
ds_out.coords['z'] = z_values

print(ds_out)

#save output
ds_out.to_netcdf(export_path)





    







































