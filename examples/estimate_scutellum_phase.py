import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from hinge_tracking_tools import find_period 
from hinge_tracking_tools import get_windowed_phase_lag 
from hinge_tracking_tools import  normalize

def load_scutum_data(filename, param):
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
    d12 = d12 - np.mean(d12)
    d13 = d13 - np.mean(d13)
    
    filt_win = param['savgol_win']
    filt_ord = param['savgol_ord']
    d12_filt = sig.savgol_filter(d12,filt_win,filt_ord)
    d13_filt = sig.savgol_filter(d13,filt_win,filt_ord)

    d12_filt = d12_filt - np.mean(d12_filt)
    d13_filt = d13_filt - np.mean(d13_filt)
    return d12_filt, d13_filt


def load_scutellum_data(filename, param):
    data = h5py.File(filename, 'r')
    obj = data['obj2'][...]
    x = obj[:,0]
    y = obj[:,1]
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Find frequencies and get filtered data
    f, psd = sig.welch(y, fs, nperseg=param['psd_nperseg'])
    peaks, _ = sig.find_peaks(psd, height=param['psd_peaks_thresh'])
    f_prob = f[peaks[0]] # low frequency wobble
    f_flap = f[peaks[1]] # flapping frequency
    f_filt = f_prob + param['hp_cutoff_frac']*(f_flap - f_prob)
    sos = sig.butter(param['hp_order'], f_filt, btype='hp', fs=fs, output='sos')
    y_hp = sig.sosfiltfilt(sos,y)
    y_hp_sg = sig.savgol_filter(y_hp, param['savgol_win'], param['savgol_ord'])

    # Load referece object and computer differential data
    ref = data['obj3'][...]
    x_ref = ref[:,0]
    y_ref = ref[:,1]
    x_diff = x - x_ref
    y_diff = y - y_ref
    x_diff = x_diff - np.mean(x_diff)
    y_diff = y_diff - np.mean(y_diff)

    y_diff_filt = sig.savgol_filter(y_diff, param['savgol_win'], param['savgol_ord'])
    y_diff_filt = y_diff_filt - np.mean(y_diff_filt)

    return y, y_hp, y_hp_sg, y_diff, y_diff_filt


if __name__ == '__main__':

    fs = 10000.0
    dt = 1.0/fs
    n_rm = 200

    scutum_data_filename = 'data/scutum_tracking_data.hdf5'
    scutum_proc_param = {
        'savgol_win': 21, 
        'savgol_ord': 3,
        }
    _, scutum_pos = load_scutum_data(scutum_data_filename, scutum_proc_param)

    scutellum_data_filename = 'data/scutellum_tracking_data.hdf5'
    scutellum_proc_param = {
            'fs': fs, 
            'psd_nperseg':  5*1024,
            'psd_peaks_thresh': 1.0e-1,
            'hp_cutoff_frac': 0.6,
            'hp_order': 10,
            'savgol_win': 21,
            'savgol_ord': 3,
            }

    scutellum_data = load_scutellum_data(scutellum_data_filename, scutellum_proc_param)
    pos, pos_hp, pos_hp_sg, pos_diff, pos_diff_filt = scutellum_data
    scutellum_pos = pos_diff_filt 

    # Remove points form start and finish to remove end effects
    scutum_pos = scutum_pos[n_rm:-n_rm]
    scutellum_pos = scutellum_pos[n_rm:-n_rm]

    # normalize to range -1, 1
    scutum_pos = normalize(scutum_pos)
    scutellum_pos = normalize(scutellum_pos)

    # Get frame #s and time points
    frame = np.arange(len(scutum_pos)) + n_rm
    t = frame*dt

    wb_period = find_period(t,scutum_pos)
    wb_freq = 1.0/wb_period

    # Cross correlate signals to find phase lag
    xcorr = sig.correlate(scutum_pos, scutellum_pos)
    lag = sig.correlation_lags(len(scutum_pos), len(scutellum_pos))
    t_lag = dt*lag
    deg_lag = 360.0*t_lag/wb_period

    # Truncate cross correlation to less than one period
    mask = np.logical_and(t_lag > -0.4*wb_period, t_lag < 0.4*wb_period)
    xcorr = xcorr[mask]
    t_lag = t_lag[mask]
    deg_lag = deg_lag[mask]

    # Find peak in cross correlation to get phase lead/lag
    phase_lag_ind = xcorr.argmax()
    phase_lag_deg = deg_lag[phase_lag_ind]
    phase_lag_xcorr = xcorr[phase_lag_ind]

    print()
    print(f'period:    {wb_period:0.5f} (s)')
    print(f'frequency: {wb_freq:0.2f} (Hz)')
    print(f'phase:     {phase_lag_deg:0.2f} (deg)')
    print()

    xcorr_win = int(15*wb_period/dt)
    t_varying, phase_varying = get_windowed_phase_lag(
            t,
            scutum_pos, 
            scutellum_pos, 
            xcorr_win, 
            dt, 
            wb_period,
            disp=False
            )

    fig, ax = plt.subplots(1,1)
    ax.plot(deg_lag, xcorr,'b')
    ax.plot([phase_lag_deg], [phase_lag_xcorr], 'or')
    ax.set_xlabel('lag (deg)')
    ax.set_ylabel('xcorr')
    ax.set_title('overall phase lag')
    ax.grid(True)

    # Plot raw signals
    fig, ax = plt.subplots(2,1,sharex=True)
    scutum_line, = ax[0].plot(t, scutum_pos, 'b')
    scutellum_line, = ax[0].plot(t, scutellum_pos, 'g')
    ax[0].set_ylabel('position (normalized)')
    ax[0].grid(True)
    ax[0].legend((scutum_line, scutellum_line), ('scutum', 'scutellum'), loc='upper right')

    ax[1].plot(t_varying, phase_varying, 'b')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('phase lag (deg)')
    ax[1].grid(True)
    ax[1].set_ylim(1.1*phase_varying.min(),0)
    ax[1].set_xlabel('time (sec)')
    plt.show()






