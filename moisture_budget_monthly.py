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
f0 = xr.open_dataset(dirin+'/cntrl/mon/ua_mon_DJFM_cntrl_1850-1999.nc')
f1 = xr.open_dataset(dirin+'/cntrl/mon/va_mon_DJFM_cntrl_1850-1999.nc')
f2 = xr.open_dataset(dirin+'/ho03/mon/r1/ua_day_DJFM_ho03_'+years+'.nc')
f3 = xr.open_dataset(dirin+'/ho03/mon/r1/va_day_DJFM_ho03_'+years+'.nc')
f4 = xr.open_dataset(dirin+'/cntrl/mon/hus_day_DJFM_cntrl_1850-1999.nc')
f5 = xr.open_dataset(dirin+'/ho03/mon/r1/hus_day_DJFM_ho03_'+years+'.nc')
f6 = xr.open_dataset(dirin+'/cntrl/mon/ps_day_DJFM_cntrl_1850-1999.nc')
fa = xr.open_dataset(dirin+'/cntrl/mon/pr_day_DJFM_cntrl_1850-1999.nc')
fb = xr.open_dataset(dirin+'/cntrl/mon/evspsbl_day_DJFM_cntrl_1850-1999.nc')
fc = xr.open_dataset(dirin+'/ho03/mon/r1/pr_day_DJFM_ho03_'+years+'.nc')
fd = xr.open_dataset(dirin+'/ho03/mon/r1/evspsbl_day_DJFM_ho03_'+years+'.nc')


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
dPE = pe_diff.copy()

#divergence
div_cntrl = ua_cntrl.copy()
div_diff  = ua_cntrl.copy()

div_cntrl = mpcalc.divergence(ua_cntrl, va_cntrl)
div_diff  = mpcalc.divergence(ua_diff, va_diff)
 
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

#DPEmean = anomalous divergence of vertically integrated humidity flux (u*q)
hus_ua_flux_cntrl = (ua_cntrl*hus_cntrl*dp*c1).sum(dim='plev')
hus_va_flux_cntrl = (va_cntrl*hus_cntrl*dp*c1).sum(dim='plev')
hus_ua_flux_exp   = (ua_hos*hus_hos*dp*c1).sum(dim='plev')
hus_va_flux_exp   = (va_hos*hus_hos*dp*c1).sum(dim='plev')

PEmean_cntrl= (-1/(gp*pw))*mpcalc.divergence(hus_ua_flux_cntrl,hus_va_flux_cntrl)*1000.
PEmean_exp  = (-1/(gp*pw))*mpcalc.divergence(hus_ua_flux_exp,hus_va_flux_exp)*1000.

dPEmean = PEmean_exp - PEmean_cntrl

#Transient eddies+nonlinear terms
dTE = dPE - np.array(dPEmean)

#Surface term
dS = dPE - (dTH + dDY + dTE)

#Reconstructed d(P-E) to check the budget
dPE_rec = dTH + dDY + dTE + dS

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


ds.to_netcdf(dirout+'/moisture_budget_monthly_DJFM_'+years+'.nc')




