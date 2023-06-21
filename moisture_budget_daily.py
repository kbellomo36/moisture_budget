import os
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from geocat.f2py import dpres_plevel_wrapper 
import warnings
warnings.filterwarnings('ignore')

dirin = '/home/bellomo/work/ec_hosing'
dirout= '/home/bellomo/work/ec_hosing/analysis/moisture_budget/tests'

years='2000-2009'

#input files directories
f0 = xr.open_dataset(dirin+'/cntrl/day/ua_day_DJFM_cntrl_1850-1999.nc')
f1 = xr.open_dataset(dirin+'/cntrl/day/va_day_DJFM_cntrl_1850-1999.nc')
f2 = xr.open_dataset(dirin+'/ho03/day/r1/ua_day_DJFM_ho03_'+years+'.nc')
f3 = xr.open_dataset(dirin+'/ho03/day/r1/va_day_DJFM_ho03_'+years+'.nc')
f4 = xr.open_dataset(dirin+'/cntrl/day/hus_day_DJFM_cntrl_1850-1999.nc')
f5 = xr.open_dataset(dirin+'/ho03/day/r1/hus_day_DJFM_ho03_'+years+'.nc')
f6 = xr.open_dataset(dirin+'/cntrl/day/ps_day_DJFM_cntrl_1850-1999.nc')
fa = xr.open_dataset(dirin+'/cntrl/day/pr_day_DJFM_cntrl_1850-1999.nc')
fb = xr.open_dataset(dirin+'/cntrl/day/evspsbl_day_DJFM_cntrl_1850-1999.nc')
fc = xr.open_dataset(dirin+'/ho03/day/r1/pr_day_DJFM_ho03_'+years+'.nc')
fd = xr.open_dataset(dirin+'/ho03/day/r1/evspsbl_day_DJFM_ho03_'+years+'.nc')

#compute mean wind
ua_day_cntrl = f0.ua
ua_mon_cntrl = ua_day_cntrl.mean(dim='time')

va_day_cntrl = f1.va
va_mon_cntrl = va_day_cntrl.mean(dim='time')

ua_day_hos = f2.ua
ua_mon_hos = ua_day_hos.mean(dim='time')

va_day_hos = f3.va
va_mon_hos = va_day_hos.mean(dim='time')

#mean wind diff
ua_diff = ua_mon_hos - ua_mon_cntrl
va_diff = va_mon_hos - va_mon_cntrl

#monthly mean horizontal wind speed
uv_cntrl = np.sqrt((ua_mon_cntrl**2 + va_mon_cntrl**2))
uv_hos   = np.sqrt((ua_mon_hos**2 + va_mon_hos**2))
uv_diff  = uv_hos - uv_cntrl

#compute mean moisture
hus_day_cntrl = f4.hus
hus_mon_cntrl = hus_day_cntrl.mean(dim='time')

hus_day_hos = f5.hus
hus_mon_hos = hus_day_hos.mean(dim='time')

#mean moisture diff
hus_diff = hus_mon_hos - hus_mon_cntrl


#deviation of daily wind data from monthly mean multiplied by
#deviation of daily humidity data from monthly mean

qdev_cntrl = (hus_day_cntrl - hus_mon_cntrl)
qu_cntrl0  = (ua_day_cntrl - ua_mon_cntrl)*qdev_cntrl
qv_cntrl0  = (va_day_cntrl - va_mon_cntrl)*qdev_cntrl
qu_cntrl   = qu_cntrl0.mean(dim='time')
qv_cntrl   = qv_cntrl0.mean(dim='time')

qdev_hos = (hus_day_hos - hus_mon_hos)
qu_hos0 = (ua_day_hos - ua_mon_hos)*qdev_hos
qv_hos0 = (va_day_hos - va_mon_hos)*qdev_hos
qu_hos = qu_hos0.mean(dim='time')
qv_hos = qv_hos0.mean(dim='time')

qu_diff = qu_hos - qu_cntrl
qv_diff = qv_hos - qv_cntrl


#mean sfc pressure
ps_cntrl = f6.ps.mean(dim='time')

#precip minus evap
pr_cntrl = fa.pr.mean(dim='time')*86400.
ev_cntrl = fb.evspsbl.mean(dim='time')*86400.

pr_hos = fc.pr.mean(dim='time')*86400.
ev_hos = fd.evspsbl.mean(dim='time')*86400.

pe_cntrl = pr_cntrl - ev_cntrl
pe_hos   = pr_hos - ev_hos
pe_diff  = pe_hos - pe_cntrl

pr_diff = pr_hos - pr_cntrl
ev_diff = ev_hos - ev_cntrl

#rename some variables
ua_cntrl = ua_mon_cntrl.copy()
va_cntrl = va_mon_cntrl.copy()
ua_hos = ua_mon_hos.copy()
va_hos = va_mon_hos.copy()
hus_cntrl = hus_mon_cntrl.copy()
hus_hos = hus_mon_hos.copy()
dqu = qu_diff.copy()
dqv = qv_diff.copy()
dPE = pe_diff.copy()

dudq = hus_diff*ua_diff
dvdq = hus_diff*va_diff

#divergence
div_cntrl = ua_cntrl.copy()
div_diff  = ua_cntrl.copy()

div_cntrl = mpcalc.divergence(ua_cntrl, va_cntrl)
div_diff  = mpcalc.divergence(ua_diff, va_diff)

dnl   = mpcalc.divergence(dudq, dvdq)              
dte   = mpcalc.divergence(dqu, dqv)   
dthd  = hus_diff*div_cntrl
ddyd  = hus_cntrl*div_diff

#finite difference/gradient
grad_cntrl = mpcalc.gradient(hus_cntrl,axes=['lon','lat'])
qx_cntrl   = grad_cntrl[0]
qy_cntrl   = grad_cntrl[1]

grad_diff = mpcalc.gradient(hus_diff,axes=['lon','lat'])
qx_diff   = grad_diff[0]
qy_diff   = grad_diff[1]

dtha  = ua_cntrl*qx_diff + va_cntrl*qy_diff
ddya  = ua_diff*qx_cntrl + va_diff*qy_cntrl

#vertical integration
dp0 = dpres_plevel_wrapper.dpres_plevel(np.array(ua_cntrl.plev),np.array(ps_cntrl),np.float(ua_cntrl.plev.min()))
dp  = xr.DataArray(dp0,coords=[ua_cntrl.plev,ps_cntrl.lat,ps_cntrl.lon],dims=['plev','lat','lon'])

#define constants
gp = 9.81 #m/s^2
pw = 1000. #kg/m^3
c0 = 24*60*60*1000. #86400 to go to mm/day
c1 = 24*60*60.

#thermodynamic
dTHa= ((-1./(gp*pw))*np.array(dtha)*dp*c0).sum(dim='plev') #advective
dTHd= ((-1./(gp*pw))*np.array(dthd)*dp*c0).sum(dim='plev') #divergent
dTH = dTHa + dTHd

#dynamic
dDYa = ((-1./(gp*pw))*np.array(ddya)*dp*c0).sum(dim='plev') #advective
dDYd = ((-1./(gp*pw))*np.array(ddyd)*dp*c0).sum(dim='plev') #divergent
dDY  = dDYa + dDYd

#nonlinear term
dNL = ((-1./(gp*pw))*np.array(dnl)*dp*c0).sum(dim='plev') 

#transient eddies
dTE = ((-1./(gp*pw))*np.array(dte)*dp*c0).sum(dim='plev') 

#Surface term
dS = dPE - (dTH + dDY + dTE + dNL)

#Reconstructed d(P-E) to check the budget
dPE_rec = dTH + dDY + dTE + dNL + dS

#write files

lat = ua_cntrl.lat
lon = ua_cntrl.lon
plev= ua_cntrl.plev

ds = xr.Dataset(
    data_vars=dict(      
        dTHa=(['lat','lon'], dTHa),
        dTHd=(['lat','lon'], dTHd),
        dTH=(['lat','lon'], dTH),        
        dDYa=(['lat','lon'], dDYa),
        dDYd=(['lat','lon'], dDYd),
        dDY=(['lat','lon'], dDY),
        dPE=(['lat','lon'], dPE),
        dNL=(['lat','lon'], dNL),
        dTE=(['lat','lon'], dTE),
        dS=(['lat','lon'], dS),
        dPE_rec=(['lat','lon'], dPE_rec),
    ),
    coords=dict(
        lon=(['lon'], lon),
        lat=(['lat'], lat),
    ),
    attrs=dict(description="moisture budget daily fields"),
)


ds.to_netcdf(dirout+'/moisture_budget_daily_DJFM_'+years+'.nc')






























