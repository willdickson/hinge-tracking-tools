import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from hinge_tracking_tools import find_period

data_file = 'data/scutellum_tracking_data.hdf5'

fs = 10000.0
dt = 1.0/fs

sg_win = 21
sg_ord = 3

# Process scutellum data
scutellum_obj_name = 'obj2'
scutellum_ref_name = 'obj3'

data = h5py.File(data_file, 'r')
scutellum_obj = data[scutellum_obj_name][...]
scutellum_ref = data[scutellum_ref_name][...]

x_scutellum = scutellum_obj[:,0]
y_scutellum = scutellum_obj[:,1]
x_scutellum_ref = scutellum_ref[:,0]
y_scutellum_ref = scutellum_ref[:,1]
dx_scutellum = x_scutellum - x_scutellum_ref
dy_scutellum = y_scutellum - y_scutellum_ref
dx_scutellum = dx_scutellum - np.mean(dx_scutellum)
dy_scutellum = dy_scutellum - np.mean(dy_scutellum)
dy_scutellum_filt = sig.savgol_filter(dy_scutellum, sg_win, sg_ord)
frame = np.arange(len(dy_scutellum))
t = dt*frame

# Process scutum data
scutum_obj_l_name = 'obj4'
scutum_obj_r_name = 'obj5'
scutum_obj_l = data[scutum_obj_l_name][...]
scutum_obj_r = data[scutum_obj_r_name][...]
x_scutum_l = scutum_obj_l[:,0]
y_scutum_l = scutum_obj_l[:,1]
x_scutum_r = scutum_obj_r[:,0]
y_scutum_r = scutum_obj_r[:,1]
dx_scutum = x_scutum_l - x_scutum_r
dx_scutum = dx_scutum - np.mean(dx_scutum)
dx_scutum_filt = sig.savgol_filter(dx_scutum, sg_win, sg_ord)

# Find wing beat period
wb_period = find_period(t, dx_scutum_filt)
wb_freq = 1.0/wb_period
print(f'wb freq: {wb_freq} (Hz)')

# Cross correlate signals to find phase lag
xcorr = sig.correlate(dx_scutum_filt, dy_scutellum_filt)
lag = sig.correlation_lags(len(dx_scutum_filt), len(dy_scutellum_filt))
t_lag = dt*lag
deg_lag = 360.0*t_lag/wb_period

# Truncate cross correlation to less than one period
mask = np.logical_and(t_lag > -0.25*wb_period, t_lag < 0.25*wb_period)
xcorr = xcorr[mask]
t_lag = t_lag[mask]
deg_lag = deg_lag[mask]
peaks, _ = sig.find_peaks(xcorr)

print(f'len(peaks) = {len(peaks)}')
print(f'phase lag: {deg_lag[peaks[0]]} (deg)')
print(deg_lag[peaks[0]])

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t, dy_scutellum,'b')
ax[0].plot(t, dy_scutellum_filt,'g')
ax[0].set_ylabel('scutelllum pos (pixel)')
ax[0].grid(True)

ax[1].plot(t, dx_scutum,'b')
ax[1].plot(t, dx_scutum_filt,'g')
ax[1].set_ylabel('scutum pos (pixel)')
ax[1].grid(True)
ax[1].set_xlabel('time (sec)')

fig, ax = plt.subplots(1,1)
ax.plot(deg_lag, xcorr, '.-')
ax.plot(deg_lag[peaks], xcorr[peaks], 'or')
ax.set_xlabel('lag (deg)')
ax.set_ylabel('xcorr')
ax.grid(True)

plt.show()
