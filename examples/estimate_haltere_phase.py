import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from hinge_tracking_tools import find_period

def load_scutum_data(filename, filt_win=21, filt_ord=3):
    data = h5py.File(filename, 'r')
    obj1 = data['obj1'][...]
    obj2 = data['obj2'][...]
    obj3 = data['obj3'][...]
    
    x1 = obj1[:,0]
    y1 = obj1[:,1]
    
    x2 = obj2[:,0]
    y2 = obj2[:,1]
    
    x3 = obj3[:,0]
    y3 = obj3[:,1]
    
    d12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    d13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
    d12 = d12 - np.median(d12)
    d13 = d13 - np.median(d13)
    
    d12_filt = sig.savgol_filter(d12,filt_win,filt_ord)
    d13_filt = sig.savgol_filter(d13,filt_win,filt_ord)
    return d12_filt, d13_filt

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    scutum_data_filename = 'data/scutum_tracking_data.hdf5'
    haltere_angle_filename = 'data/haltere_angle.npy'
    dt = 1.0/10000.0

    d12, d13 = load_scutum_data(scutum_data_filename)
    angle = np.load(haltere_angle_filename)
    frame = np.arange(len(angle))
    t = frame*dt

    d12 = d12[:len(angle)]
    d13 = d13[:len(angle)]
    
    d13 = d13/(0.5*(d13.max() - d13.min()))
    angle = angle/(0.5*(angle.max() - angle.min()))

    period = find_period(t,d13)
    freq = 1.0/period

    xcorr = sig.correlate(d13,-angle)
    lag = sig.correlation_lags(len(d13), len(angle))
    t_lag = dt*lag
    deg_lag = 360.0*t_lag/period

    mask = np.logical_and(t_lag > -0.25*period, t_lag < 0.25*period)
    xcorr = xcorr[mask]
    t_lag = t_lag[mask]
    deg_lag = deg_lag[mask]

    phase_lag_ind = xcorr.argmax()
    phase_lag_deg = deg_lag[phase_lag_ind]
    phase_lag_xcorr = xcorr[phase_lag_ind]

    print()
    print(f'period:    {period:0.5f} (s)')
    print(f'frequency: {freq:0.2f} (Hz)')
    print(f'phase:     {phase_lag_deg:0.2f} (deg)')
    print()

    fig, ax = plt.subplots(1,1)
    d13_line, = ax.plot(t, d13, color='b')
    haltere_line, = ax.plot(t, -angle, color='g')
    ax.grid(True)
    ax.set_xlabel('frame')
    fig.legend((d13_line, haltere_line), ('scutum', 'haltere'), loc='upper right')

    fig, ax = plt.subplots(1,1)
    ax.plot(deg_lag, xcorr,'b')
    ax.plot([phase_lag_deg], [phase_lag_xcorr], 'or')
    ax.set_xlabel('lag (deg)')
    ax.set_ylabel('xcorr')
    ax.grid(True)

    plt.show()
    
