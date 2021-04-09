"""
#!/usr/bin/python
utility.py
-----------------------------------------------------
This script contains general utilities
Sam Doyle 
3 June 2016
-----------------------------------------------------
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

def deg2rad(deg):
    '''Function to convert to degrees to radians '''
    return deg * np.pi / 180
    
def rad2deg(rad):
    '''Function to convert to radians to degrees '''
    return rad *  180 / np.pi

def unix_time_millis(dt): 
    '''Function to convert to milliseconds since 1970 epoch '''
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0

def m2dt(matlab_datenum): 
    ''' Converts matlab datenumber to python datetime '''
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac
    
def mH20_to_MPA(mH20):
    g = 9.81 # [m s^-2]
    rho_w = 1000 #[kg m^-3]
    MPA = mH20 * g * rho_w * 1e-6
    return MPA

def MPA_to_mH20(MPA):
    g = 9.81 # [m s^-2]
    rho_w = 1000 #[kg m^-3]
    mH20 = MPA * (g * rho_w * 1e-6)**-1 
    return mH20   
    
def daterange(start_date, end_date): 
    ''' Creates a range of dates for 'for' loops '''
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)
        
def shade(periods, color): 
    ''' Shades periods on timeseries plots 
    
    Parameters
    ----------
    periods : array of n rows of periods in datetime string. Column 0 is start of period. Column 1 is end of period
    
    color: colour for shading
    
    Returns
    -------
    None
    
    Example
    -------
    shade(periods, 0.8)
    
    '''
    for i in range(len(periods)):
        ax.axvspan(pd.to_datetime(periods[i,0]), pd.to_datetime(periods[i,1]), 
                                   color=color, alpha=0.5, lw=0)
def shade2(periods, color, *ax): 
    ''' Shades periods on timeseries plots 
    
    Parameters
    ----------
    periods : array of n rows of periods in datetime string. Column 0 is start of period. Column 1 is end of period
    
    color: colour for shading
    
    Returns
    -------
    None
    
    Example
    -------
    shade(periods, 0.8)
    
    '''
    for i in range(len(periods)):
        ax.axvspan(pd.to_datetime(periods[0,0]), pd.to_datetime(periods[0,1]), 
                                   color=color, alpha=0.5, lw=0)
def draw_inset_box(ax1,ax2,ls,lw):
    ''' Draws the extent of an inset plot (ax2) on the main plot (ax1) 
    
    Parameters
    ----------
    ax1 : axes handle of main subplot (i.e. long time series)
    
    ax1 : axes handle of inset subplot (i.e. zoomed in plot)
    
    ls : line style (and optionally colour)
    
    lw : line width
    
    Returns
    -------
    None
    
    Example
    -------
    draw_inset_box(ax1, ax2, 'r--', 1.5)
    
    '''
    ax1.plot(ax2.get_xlim(),(ax2.get_ylim()[0], ax2.get_ylim()[0]),
              ls, linewidth=lw)
    ax1.plot(ax2.get_xlim(),(ax2.get_ylim()[1], ax2.get_ylim()[1]),
              ls, linewidth=lw)
    ax1.plot((ax2.get_xlim()[0],ax2.get_xlim()[0]),ax2.get_ylim(),
              ls, linewidth=lw)
    ax1.plot((ax2.get_xlim()[1],ax2.get_xlim()[1]),ax2.get_ylim(),
              ls, linewidth=lw)
              
def daily_ticks(ax,*hr_int):
    ''' Configures daily ticks on x-axis
        ax = axes
        hr_int = minor tick interval [h] (optional; defaults to 4 hours)'''
    import matplotlib.dates as dates
    days = dates.DayLocator()
    if not hr_int:
        hr_int = 4
    hours = dates.HourLocator(interval=hr_int)
    dfmt = dates.DateFormatter('%d\n%b')
    
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(dfmt)
    ax.xaxis.set_minor_locator(hours)
    
def conical_frustum(R, r, h): 
    ''' Calculates volume of a conical frustrum (truncated cone) 
        R = radius of the base
        r = radius of the upper surface
        h = height of the conical frustum
    See http://mathworld.wolfram.com/ConicalFrustum.html '''
    V = h * np.pi/3 * (R**2 + R*r + r**2)
    return V
def dt2doy(dt): 
    ''' Converts pandas datetime to fractional day of year '''
    td = dt - pd.to_datetime(str(dt.year - 1) + ' -12-31')
    doy = td.days + td.seconds/60/60/24
    return doy
def doy2dt(doy, year): 
    ''' Converts fractional day of year to pandas datetime 
        
    Parameters
    ----------
    doy : fractional day of year
    
    year: year
    
    Returns
    -------
    Pandas Timestamp
    
    Example
    -------
    doy2dt(180.75, 2010) '''

    return pd.Timedelta(doy, 'D') + pd.to_datetime(str(year - 1) + ' -12-31')
    
def add_doy_column(df): 
    ''' Adds DOY column to Pandas dataframes that have datetime index '''
    thedoys = np.empty((df.shape[0],1), dtype=float) * np.NaN
    for i in range(len(thedoys)):
        thedoys[i] = dt2doy(df.index[i])
    df['DOY'] = thedoys
    return df
def add_date_column(df, year): 
    ''' Adds date column to Pandas dataframes that have DOY index '''
    thedates = np.empty((df.shape[0],1), dtype=float) * np.NaN
    for i in range(len(thedates)):
        thedates[i] = unix_time_millis(doy2dt(df.index[i], year))
    thedates = np.round(thedates / 1000) * 1000 # Round to nearest second
            
    t = thedates.astype('int').astype("datetime64[ms]")
    df['index'] = pd.to_datetime(t[:,0])
    df = df.set_index('index')
    return df
#def stair(x,y,c,lw): 
#    ''' Correctly plots time period for step plot 
#            x is a DatetimeIndex 
#            y is the independent variable 
#            c is the line color string 
#            lw is the line width '''
#    Interval = int(np.round(float(np.mean((np.diff(x))))*1E-9)) / 60 / 60 / 24 # [Days]
#    x = pd.DatetimeIndex.append(pd.date_range(str(x[0] 
#        - pd.Timedelta(Interval,'D'))[0:10], periods=1, freq='D'), x)
#    y = np.concatenate(([y[0]],y), axis=0)
#    x = x + pd.Timedelta(Interval,'D')
#    h = plt.step(x,y,c, linewidth=lw)
#    return h
def stair(x,y,c,lw,ax,*ls): 
    ''' Correctly plots time period for step plot 
            x is a DatetimeIndex 
            y is the independent variable 
            c is the line color string 
            lw is the line width
            ax is the axes instance'''
    Interval = int(np.round(float(np.mean((np.diff(x))))*1E-9)) / 60 / 60 / 24 # [Days]
    x = pd.DatetimeIndex.append(pd.date_range(str(x[0] 
        - pd.Timedelta(Interval,'D'))[0:10], periods=1, freq='D'), x)
    y = np.concatenate(([y[0]],y), axis=0)
    x = x + pd.Timedelta(Interval,'D')
    if ls:
        h = ax.step(x,y,c, linewidth=lw, linestyle=ls)
    else:
        h = ax.step(x,y,c, linewidth=lw, linestyle='-')
    return h
def azimuth(dE, dN): 
    ''' Calculate azimuth [deg] from Cartesian easting and northing  '''
    az = np.arctan2(np.array([dE]), np.array([dN])) * 180 / np.pi
    if az < 0: # Convert to 0-360 rather than negative degrees
        az = az + 360
    return az
def RunningMean(x, N): 
    ''' Calculate running mean of x over N periods '''
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
    
def nan_helper(x):
    ''' Helper to handle indices and logical indices of NaNs.
    From https://stackoverflow.com/questions/6518811/
    interpolate-nan-values-in-a-numpy-array
    Parameters
    ----------
        x: 1d numpy array with possible NaNs
    Returns
    -------
        nans: logical indices of NaNs (i.e. a mask)
        idx: a function, with signature indices = index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices    
    Example
    -------
        # linear interpolation of NaNs
        nans, idx = nan_helper(x)
        x[nans]= np.interp(idx(nans), idx(~nans), x[~nans])
    '''
    return np.isnan(x), lambda z: z.nonzero()[0]

def cma(x, w):
    ''' Calculates the centred moving-average (see p. 55 of my thesis).
    x must not contain NaNs.
        
    Parameters
    ----------
    x : 1d numpy array (must not containe NaNs)
    w : moving average half-length
    
    Returns
    -------
    xf : 1d numpy array with moving average applied 
    
    Example
    -------
    xf = cma(x, w) '''
    
    if np.isnan(x).any(): # check for NaN
        print('--> x contains NaN: Remove these before continuing, or use gapped cma')
        return
        
    Wn = (w * 2) + 1 # Cut-off period [samples]
    xf = np.convolve(x, np.ones((Wn,))/Wn, mode='same')
    xf[:w] = np.NaN # remove boundary effect at start
    xf[-w:] = np.NaN # remove boundary effect at end
    return xf
    
def gapped_cma(x, w): 
    ''' Calculates the centred moving-average (see p. 55 of my thesis)
    on gapped data. It first interpolates the gaps and applies the moving 
    average to the interpolated data. It then adds the gaps back in and 
    removes the boundary effects around the gaps.
        
    Parameters
    ----------
    x : 1d numpy array (can contain NaNs)
    w : moving average half-length
    
    Returns
    -------
    xf : 1d numpy array with moving average applie 
    
    Example
    -------
    xf = cma(x, w) '''
    
    xi = np.copy(x) 
    ''' Create a copy of x ready for interpolation. Not 
    creating a copy  results in this function changing variable x outside the 
    function '''
    
    # Find the NaNs
    nans, idx = nan_helper(xi) 
    
    # Above idx is a function than gets the indices of the NaNs.  

    # Interpolate the nans
    xi[nans]= np.interp(idx(nans), idx(~nans), xi[~nans])
    
    # Apply the filter to the interpolated dataset
    xf = cma(xi, w)
    
    # Add the NaNs back in and remove boundary effects
    for i in range(len(idx(nans))):            
        ix0 = idx(nans)[i] - w # left index to set to NaN
        ix1 = idx(nans)[i] + w # right index to set to NaN
        
        if ix0 < 0: # ensure index is within bounds
            ix0 = 0
        
        if ix1 > len(xi): # ensure index is within bounds
            ix1 = len(xi) - 1
        
        xf[ix0:ix1+1] = np.NaN # replace with NaNs
    return xf
    
def vmg(rhumb, cog, v):
    ''' Calculates velocity (or displacement) made good 
        
    Parameters
    ----------
    rhumb : rhumb line bearing [deg]
    cog : course over ground [deg]
    v : velocity or displacement
    
    Returns
    -------
    vmg : velocity (or displacement) made good [same units as input].
    co  : course offset [deg]
    
    Example
    -------    
    vmg, co = vmg_calc(280, 270, 5) '''
    co = rhumb - cog # course offset
    vmg = v * np.cos(deg2rad(co))
    return vmg, co
def inline(E,N):
    ''' Calculates detrended inline position, residual displacement, 
    inline position, displacement made good, and some other variables. 
        
    Parameters
    ----------
    E : Eastings [m]  (East is positive)
    
    N : Northings [m] (North is positive)
    
    Returns
    -------
    s       : epoch displacement [m]
    az      : epoch azimuth [deg]
    rhumb   : rhumb line bearing [deg]
    vmg     : velocity (or displacement) made good [same units as input]
    s_mean  : mean displacement [m]
    co      : epoch course offsets [deg]
    pos     : inline position
    rpos    : detrended inline position
    
    
    Example
    -------    
    s, az, rhumb, s_mean, dmg, r, pos, rpos = inline(dE, dN) '''
    
    dE = np.diff(E) 
    dN = np.diff(N) 
    s = np.sqrt(dE**2 + dN**2) # Calculate epoch displacement
    
    # Calculate epoch azimuth
    az = np.arctan2(np.array([dE]), np.array([dN])) * 180 / np.pi
    az[az < 0] = az[az < 0] + 360 # Convert to 0-360 rather than negative degrees
    
    # Calculate mean azimuth (i.e the rhumb line) and epoch displacement
    rhumb = np.mean(az)
    s_mean = (np.sqrt(np.mean(dE)**2 + np.mean(dN)**2))
    
    # Calculate and reshape displacement made good
    dmg, co = vmg(rhumb, az, s) 
    dmg  = dmg.reshape(dmg.shape[1],dmg.shape[0])
    
    r = dmg - s_mean # Residual displacement
    pos = np.cumsum(dmg) # Inline position
    rpos = np.cumsum(r) # De-trended inline position
    return s, az, rhumb, s_mean, dmg, r, pos, rpos

def linear(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    r, p = pearsonr(x, y)
    m, c = np.polyfit(x,y,1)  
    Y = (m * x) + c
    return x, y, Y, r, p, m, c
    
def p_i_calc(rho_i, g, H, B):
    ''' Calculates ice overburden pressure p_i for an inclined bed. 
        
    Parameters
    ----------
    rho_i   : Ice density [kg m^-3] 
    g       : Gravitational acceleration [m s^-2]
    H       : Ice thickness [m]
    B       : Bedslope [deg]    (Positive is downwards sloping) 
    
    Returns
    -------
    p_i     : Ice overburden pressure [Pa]
    
    Example
    -------    
    p_i = p_i_calc(rho_i, g, H, B) '''
    p_i = rho_i * g * H * np.cos(deg2rad(B)) # Ice overburden pressure [MPa]
    return p_i
    
  
def local_gravity(L, H):   
    ''' http://www.npl.co.uk/reference/faqs/how-can-i-determine-my-local-values-of-gravitational-acceleration-and-altitude-(faq-mass-and-density)
    The uncertainty in the value of g so obtained is generally less than ±5 parts in 10^5.    
    
    Parameters
    ----------    
    L = latitude
    H = height in metres above sea level    
    
    Returns
    -------
    g       : Local gravitational acceleration [m s^-2]
    
    Example
    ------- 
    g = local_gravity(70,1000)
    '''
    A = 0.0053024
    B = 0.0000058
    g = 9.780327 * (1 + A * (np.sin(L))**2 - B * (np.sin(2*L))**2) - 3.086 * 1e-6 * H # [m·s-2]
    return g

def rms(x):
    ''' Calculates root mean square (RMS) of x ignoring NaN '''
    n_nan = np.sum(np.isnan(x))
    if n_nan !=0:
         print(["--> Warning: " + str(n_nan) + " NaN values ignore."])
    return np.sqrt(np.nanmean(x**2))

def tilt_resolve(pitch, roll):
    tilt = np.arccos(np.cos((pitch)*(np.pi/180)) *  \
              np.cos((roll)*(np.pi/180))) * (180/np.pi)  # [Degrees] resolve pitch and roll; rotation method better for large tilt angles
    return tilt

def constants(): 
    Ttp = 273.16 # [K] triple point temp
    Ptp = 611.73 # [Pa] triple point pressure
    lamda_pure = 0.0742e-6 # [K MPa^-1] Clausius-Clapeyron constant for pure ice and pure water 
    lamda_air = 0.098e-6 # [K MPa^-1] Clausius-Clapeyron constant for pure ice and air-saturated water 
    lamda_luthi2002 = 0.079e-6 # [K MPa^-1] Clausius-Clapeyron constant from Luthi et al. [2002]
    rho_i = 900 # [kg m^-3] +- 18
    rho_w = 999.841 # [kg m^-3] (at 0 Celcius)
    g = 9.81 # [m s^-2] +-0.07
    k_i = 2.10 # [W m^-1 K^-1] Thermal conductivity of pure ice
    return Ttp, Ptp, lamda_pure, lamda_air, lamda_luthi2002, rho_i, rho_w, g, k_i

def Tm(H, icetype, rho_i, g, B): # PMP Calculation
    Ttp = 273.16 # [K] triple point temp
    Ptp = 611.73 # [Pa] triple point pressure
    lamda_pure = 0.0742e-6 # [K MPa^-1] Clausius-Clapeyron constant for pure ice and pure water 
    lamda_air = 0.098e-6 # [K MPa^-1] Clausius-Clapeyron constant for pure ice and air-saturated water 
    lamda_luthi2002 = 0.079e-6 # [K MPa^-1] Clausius-Clapeyron constant from Luthi et al. [2002]
        
    if icetype=='air':
        p_i = rho_i * g * H * np.cos(deg2rad(B))
        return Ttp - lamda_air * (p_i - Ptp) - 273.15
    else:
        if icetype=='pure':   
            p_i = rho_i * g * H * np.cos(deg2rad(B))
            return Ttp - lamda_pure * (p_i - Ptp) - 273.15
        else:
            if icetype=='luthi':# From Luthi et al. [2002]
                p_i = rho_i * g * H * np.cos(deg2rad(B))
                return Ttp - lamda_luthi2002 * (p_i - Ptp) - 273.15

def dot_plot(data):
    ''' Produces a dot plot, which is similar to a histogram.
    Parameters
    ----------    
    data: 1-d numpy array   
    
    Returns
    -------
    f       : figure handle    
    
    Example
    ------- 
    np.random.seed(13)
    data = np.random.randint(0,12,size=72)
    f = dot_plot(data)
    '''
    
    bins = np.arange(max(data)+2)
    
    hist, edges = np.histogram(data, bins=bins)
    
    y = np.arange(1,hist.max()+1)
    x = np.arange(max(data)+1)
    X,Y = np.meshgrid(x,y)
    
    Y = Y.astype(np.float)
    Y[Y>hist] = np.nan
    
    f = plt.figure()
    plt.scatter(X,Y)
    plt.show()
    plt.ylabel('Freq.')
    return f

def mad(x):
    ''' Finds the median absolute deviation from the median (MAD). It 
    assumes normal distribution using a consistency constant of 1.4826
    
    https://eurekastatistics.com/using-the-median-absolute-deviation-to-
    find-outliers/
    Parameters
    ----------    
        x: 1-d numpy array   
        c: consistency constnt
    
    Returns
    -------
        MAD       : Median absolute deviaton from the median
    
    Example
    ------- 
        x_mad =  mad(x)
    '''
    c = 1.4826 # assumes normal distribiution
    return c * np.median(abs(x - np.median(x)))

def find_outlier(x, n, method):
    ''' Finds outliers of x using one of two methods. 
    
    SIGMA Method: Finds outliers that are n standard deviations from the mean.
    MAD Method: Finds outliers that are n * MAD from the median. It uses the 
    mad() function, which assumes normal distribution.
    
    The MAD method is thought to be more robust. See 
    https://eurekastatistics.com/using-the-median-absolute-deviation-to-
    find-outliers/
    
    Note, that this function handles NaNs by interpolating the input array x
    before finding the outliers. NaNs are not counted as outliers.
    
    Parameters
    ----------    
        x: 1-d numpy array with possible NaNs 
        n: cutoff (number of sigmas or MADs from the mean)
        method: see above, either 'sigma', or 'mad'
    
    Returns
    -------
        outliers: logical indices of outliers (i.e. a mask)
    
    Example
    ------- 
        # Find outliers in x using 2 * MAD as the cutoff
        outliers =  find_mad_outliers(x, 2, 'mad')
        
        # Find 1-sigma outliers in x 
        outliers =  find_mad_outliers(x, 1, 'sigma')
    '''
    xi = np.copy(x) # create a copy so as not to overwrite
    if np.isnan(x).any(): # check for NaN
        print('--> Interpolating NaNs')
        nans, idx = nan_helper(xi) 
        xi[nans]= np.interp(idx(nans), idx(~nans), xi[~nans])
    
    if method == 'mad' or method == 'MAD':
        outliers = abs(xi - np.median(xi)) / mad(xi) > n
    elif method == 'sigma' or method == 'SIGMA':
        outliers = abs(xi) >= (n * np.std(xi))
    else: 
        print("--> Method to find outliers not specified")
        return
    
    outliers = outliers & ~nans # do not count NaNs as outliers
    return outliers
    
def derivative(x, y, method='central', fill=True):
    '''Compute the first derivitive of x with respect to y using either 
    forwards, backwards or central differentiation. Where a method cannot 
    compute a value (e.g. at the start or end of a time series) setting the 
    option 'fill' to True will use alternative methods to calculate the 
    derivative at the start and/or end.
    
    Note that the fill option does not fill NaNs, it only fills where there 
    is valid data but the method of diffferentiation does not work due to the
    start/end. This function handles NaNs by ensuring they are not used in 
    the calculation. It does not fill them.
    
    Parameters
    ----------
    x : 1-d numpy array
    y : 1-d numpy array (can contain NaNs)
    method : Difference formula: 'forward', 'backward' or 'central'
    fill : [Boolean]. If True alternative methods will be used to 
        calculate the derivative at the start and/or end
    Returns
    -------
    dy: derivitave of y with respect to x      
    '''
    
    if not type(fill) == bool:
        raise ValueError("fill must be boolean True or False")
        
    dy= np.zeros([len(x),1]) # Initialise array
     
    if method == 'central':     
        if fill == True:   
            dy[0] = (y[0] - y[1]) / (x[0] - x[1]) # forwards difference
            dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) # backwards difference
        elif fill == False:
            dy[0] = np.NaN
            dy[-1] = np. NaN

        for i in range(1,len(y)-1):
            dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])  
    
    elif method == 'forward':
        if fill == True:
            dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) # backwards difference        
        elif fill == False:
            dy[-1] = np. NaN
           
        for i in range(len(y)-1):
            dy[i] = (y[i + 1] - y[i]) / (x[i + 1]-x[i])

    elif method == 'backward':
        if fill == True:
            dy[0] = (y[0] - y[1]) / (x[0] - x[1]) # forwards difference
        elif fill == False:
            dy[0] = np. NaN
            
        for i in range(1,len(y)):
            dy[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
    
    return dy
    
def cm2inch(*tupl):
    ''' Converts a tuple of x and y size in cm to inches, to be used to set 
    the size of figures.
    
    Parameters
    ----------
    tupl : Figure size as Tuple in cm (x, y)
    Returns
    -------  
    tupl : Figure size as Tuple in inches (x, y)
    
    Example
    -------
    # For a figure of size 14.4 x 14.4 cm:
    f, ax = plt.subplots(2, 1, figsize=ut.cm2inch(14.4, 14.4)) 
    '''
    
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)