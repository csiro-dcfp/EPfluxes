def EPflux_primitive(U, V, T, W, time_name=None, lat_name=None, lon_name=None, level_name=None, version_date=None):
    '''
    Based on https://www.ncl.ucar.edu/Document/Functions/Contributed/epflux.shtml
    doppyo: https://github.com/csiro-dcfp/doppyo

    U/V units of m/s
    W units of Pa/s (OMEGA)
    T units of K

    U must have level_units coordinate attribute.

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

    P0_pa = 100000

    LAT = U[lat_name][:]

    degtorad = utils.constants().pi / 180
    a     = utils.constants().R_earth
    PI = utils.constants().pi
    omega = utils.constants().Omega
    phi   = LAT * degtorad
    acphi = a*np.cos(phi)
    asphi = a*np.sin(phi)
    f     = 2*omega*np.sin(phi)

    latfac = acphi*np.cos(phi)       # scale factor includes extra cos(phi)
                           # for graphical display of arrows
                           # see Edmon et al, 1980

    THETA = np.power(T*(PLVL_pa/P0_pa),-.286)
    
    THETAzm = THETA.mean(lon_name)
    
    Uzm = U.mean(lon_name)

    Uzmtm = Uzm.mean(time_name)
    
    #dT/dp = (1/p)*dT/d(log(p)):
    THETAp = xr.DataArray(np.gradient(THETAzm, np.log(PLVL_pa), axis=1), \
    coords=[(time_name, U[time_name].values), \
            (level_name, U[level_name][:]), \
            (lat_name, U[lat_name][:])]).rename('THETAp') / PLVL_pa

    THETAptm = THETAp.mean(time_name)

    Uza = U - U.mean(lon_name)

    Vza = V - V.mean(lon_name)

    Wza = W - W.mean(lon_name)

    THETAza = THETA - THETA.mean(lon_name)

    UpVp = Uza * Vza
    
    WpUp = Wza * Uza
    
    UpVpzm = UpVp.mean(lon_name)
    
    WpUpzm = WpUp.mean(lon_name)
    
    VpTHETAp = Vza * THETAza
    
    VpTHETApzm = VpTHETAp.mean(lon_name)
    
    UpVpzmtm = UpVpzm.mean(time_name)
    
    WpUpzmtm = WpUpzm.mean(time_name)

    VpTHETApzmtm = VpTHETApzm.mean(time_name)

    Up = xr.DataArray(np.gradient(Uzm, np.log(PLVL_pa), axis=1), \
    coords=[(time_name, U[time_name].values), \
            (level_name, U[level_name][:]), \
            (lat_name, U[lat_name][:])]).rename('Up') / PLVL_pa

    Uptm = Up.mean(time_name)
    
    Fphi = acphi * ( (Uptm * VpTHETApzmtm/THETAptm ) - UpVpzmtm)
    
    Fphi = Fphi.transpose(level_name,lat_name)

    Fphi['long_name'] = 'meridional component of EP flux'

    dUzmtmdphi = xr.DataArray( \
        np.gradient(Uzmtm*np.cos(phi), asphi, axis=1), \
        dims=[level_name, lat_name], \
        coords={level_name:U[level_name][:] , lat_name:U[lat_name][:]})
    Fp = acphi * ( (f - dUzmtmdphi/acphi)*VpTHETApzmtm/THETAptm - WpUpzmtm)
    
    Fp = Fp.transpose('level','lat')
    
    Fphi['long_name'] = 'meridional component of EP flux'
    
    EPdiv1 = np.gradient(Fphi, asphi, axis=1) #gradient wrt lat

    EPdiv2 = np.gradient(Fp, PLVL_pa, axis=0) #gradient wrt lev

    EPdiv = xr.DataArray(EPdiv1+EPdiv2, \
    coords=[(level_name, U[level_name][:]), \
            (lat_name, U[lat_name][:])]).rename('EPdiv')

    Fp = Fp/1e5
    Fp = Fp * np.cos(phi)

    Fphi = Fphi/a

    return((Fphi, Fp, EPdiv))
