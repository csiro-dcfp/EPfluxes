def EPflux_quasigeostrophic(U, V, T, magf=None, scale_sqrt_p=None, time_name=None, lat_name=None, lon_name=None, level_name=None, version_date=None):
    '''
    Based on https://www.ncl.ucar.edu/Document/Functions/Contributed/epflux.shtml
    doppyo: https://github.com/csiro-dcfp/doppyo
  
    U/V units of m/s
    T units of K
  
    U must have level_units coordinate attribute.
    Options to scale flux vectors in the stratosphere/mesosphere.
    Option to scale entire atmosphere's flux vectors by the sqrt(p0/p).
  
    magf = None or float
    scale_sqrt_p = True/False
    level,time,lat,lon name auto-detect.
  
    '''
  
    import inspect
    version_date_local = '03 December 2020'
    if(version_date is not None):
      if(version_date!=version_date_local): #should match one in comments above.
        print('Warning: version_date, '+version_date+', does not match one, '+\
              version_date_local+', embedded in function '+__file__+'.')

    import xarray as xr
    import numpy as np
    from doppyo import utils
  
    if lat_name is None:
        lat_name = utils.get_lat_name(U)
    if lon_name is None:
        lon_name = utils.get_lon_name(U)
    if level_name is None:
        level_name = utils.get_level_name(U)
    if time_name is None:
        time_name = utils.get_time_name(U)
      
    if(U.coords['level_units']=='hPa'):
      PLVL_pa = U[level_name][:]*100
    elif(U.coords['level_units']=='Pa'):
      PLVL_pa = U[level_name][:]
    else:
      raise SystemExit('level_units only can be Pa/hPa:'+__file__+' line number: '+str(inspect.stack()[0][2]))

    if(magf is None): magf=False
    if(scale_sqrt_p is None): scale_sqrt_p=False

    P0_pa = 100000

    LAT = U[lat_name][:]

    degtorad = utils.constants().pi / 180
    a     = utils.constants().R_earth # 6.37122e06       # radius of the earth (m)
    PI = utils.constants().pi # np.pi
    omega = utils.constants().Omega # 7.2921e-5
    phi   = LAT * degtorad # LAT*PI/180.0     # latitude in radians
    acphi = a*np.cos(phi)
    asphi = a*np.sin(phi)       # a*sin latitude for use in calculating the divergence.
    f     = 2*omega*np.sin(phi) # coriolis parameter

    latfac= acphi*np.cos(phi)       # scale factor includes extra cos(phi)
                             # for graphical display of arrows
                             # see Edmon et al, 1980

    THETA = np.power(T*(PLVL_pa/P0_pa),-.286)
    THETA['long_name'] = 'potential teperature'
    THETA['units'] = 'K'

    THETAzm = THETA.mean(lon_name)

    #dT/dp = (1/p)*dT/d(log(p)):
    THETAp = xr.DataArray(np.gradient(THETAzm, np.log(PLVL_pa), axis=1), \
      coords=[(time_name, U[time_name].values), \
              (level_name, U[level_name][:]), \
              (lat_name, U[lat_name][:])]).rename('THETAp') / PLVL_pa

    THETAp['long_name'] = 'dT/dp'
    THETAp['units'] = 'K/Pa'

    THETAptm = THETAp.mean(time_name)

    Uza = U - U.mean(lon_name)

    Vza = V - V.mean(lon_name)

    THETAza = THETA - THETA.mean(lon_name)

    UV = Uza*Vza
    UV['long_name'] = 'U\'V\''
    UV['units'] = 'm^2 / s^2'

    VTHETA = Vza * THETAza

    UVzm = UV.mean(lon_name)

    UVzmtm = UVzm.mean(time_name)

    VTHETAzm = VTHETA.mean(lon_name)

    VTHETAzmtm = VTHETAzm.mean(time_name)

    Fphi = -UVzmtm*latfac
    Fphi['long_name'] = 'meridional component of EP flux'

    Fp = (VTHETAzmtm/THETAptm)*f*acphi
    Fp['long_name'] = 'vertical component of EP flux'

    EPdiv1 = np.gradient(Fphi, asphi, axis=1) #gradient wrt lat

    EPdiv2 = np.gradient(Fp, PLVL_pa, axis=0) #gradient wrt lev

    EPdiv = xr.DataArray(EPdiv1+EPdiv2, \
      coords=[(level_name, U[level_name][:]), \
              (lat_name, U[lat_name][:])]).rename('EPdiv')

    dudt = (86400*EPdiv / acphi).rename('dudt')
    dudt.attrs['units'] = 'm / s^2'
    dudt.attrs['long_name'] = 'acceleration of zonal wind'

    Fp = Fp * np.cos(phi)
    Fp = Fp/1e5

    Fphi = Fphi/a
    Fphi = Fphi/np.pi
  
    if(type(magf)==type(float(1.0))):
      hPa100=10000
      #scale the stratos/mesosphere:
      Fphi = Fphi.where(Fphi[level_name]<=hPa100).fillna(0)*magf + \
        Fphi_plot.where(Fphi[level_name]>hPa100).fillna(0)
      Fp = Fp_plot.where(Fp[level_name]<=hPa100).fillna(0)*magf + \
        Fp.where(Fp[level_name]>hPa100).fillna(0)
    
    if(scale_sqrt_p):
      #apply sqrt(p0/p) scaling:
      Fphi = Fphi * xr.ufuncs.sqrt(P0_pa/PLVL_pa)
      Fp = Fp * xr.ufuncs.sqrt(P0_pa/PLVL_pa)
    
    for Q in (Fphi,Fp):  
      Q.attrs['magf'] = magf
      Q.attrs['scale_sqrt_p'] = scale_sqrt_p
  
    return((Fphi, Fp, EPdiv, dudt))
