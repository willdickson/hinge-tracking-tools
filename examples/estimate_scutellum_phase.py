import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

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
    d12 = d12 - np.median(d12)
    d13 = d13 - np.median(d13)
    
    filt_win = param['savgol_win']
    filt_ord = param['savgol_ord']
    d12_filt = sig.savgol_filter(d12,filt_win,filt_ord)
    d13_filt = sig.savgol_filter(d13,filt_win,filt_ord)
    return d12_filt, d13_filt


def load_scutellum_data(filename, param):
    data = h5py.File(filename, 'r')
    obj1 = data['obj1'][...]
    x = obj1[:,0]
    y = obj1[:,1]
    x = x - np.median(x)
    y = y - np.median(y)

    # Find frequencies
    f, psd = sig.welch(y, fs, nperseg=param['psd_nperseg'])
    peaks, _ = sig.find_peaks(psd, height=param['psd_peaks_thresh'])
    f_prob = f[peaks[0]] # low frequency wobble
    f_flap = f[peaks[1]] # flapping frequency
    f_filt = f_prob + param['hp_cutoff_frac']*(f_flap - f_prob)
    sos = sig.butter(param['hp_order'], f_filt, btype='hp', fs=fs, output='sos')
    y_hp = sig.sosfiltfilt(sos,y)
    y_hp_sg = sig.savgol_filter(y_hp, 21,3)
    return y, y_hp, y_hp_sg


def find_period(t,x,guess=1/200.0, disp=False):
    dt = t[1] - t[0]
    xcorr = sig.correlate(x,x)
    lag = sig.correlation_lags(len(x), len(x))
    t_lag = dt*lag

    peaks, _  = sig.find_peaks(xcorr)
    peaks = peaks[1:-1]

    t_peaks = t_lag[peaks]
    xcorr_peaks = xcorr[peaks]

    dt_peaks = t_peaks[1:] - t_peaks[:-1]
    period = dt_peaks.mean()

    if disp:
        fig, ax = plt.subplots(1,1)
        ax.plot(t_lag,xcorr)
        ax.plot(t_peaks, xcorr_peaks, 'or')
        ax.set_xlabel('lag (sec)')
        ax.set_ylabel('xcorr')
        ax.grid(True)
        plt.show()

    return period


def normalize(x):
    delta = x.max() - x.min()
    return x/(0.5*delta)


def get_windowed_phase_lag(t,x,y,win,dt,period,disp=False):
    num = len(x)
    t_list = []
    phase_lag_list = []
    step = 2*int(period/dt)
    if disp:
        fig, ax = plt.subplots(1,1)
    for i in range(1,num-win,step):
        # Cross correlate signals to find phase lag
        twin = t[i:i+win]
        xwin = x[i:i+win]
        ywin = y[i:i+win]
        xcorr = sig.correlate(xwin, ywin)
        lag = sig.correlation_lags(len(xwin), len(ywin))
        t_lag = dt*lag
        deg_lag = 360.0*t_lag/period

        # Truncate cross correlation to less than one period
        mask = np.logical_and(t_lag > -0.5*period, t_lag < 0.5*period)
        xcorr = xcorr[mask]
        t_lag = t_lag[mask]
        deg_lag = deg_lag[mask]


        # Find peak in cross correlation to get phase lead/lag
        phase_lag_ind = xcorr.argmax()
        phase_lag_deg = deg_lag[phase_lag_ind]
        phase_lag_xcorr = xcorr[phase_lag_ind]

        if disp:
            ax.plot(deg_lag, xcorr)
            ax.plot([phase_lag_deg],[phase_lag_xcorr], 'or')
            ax.set_xlabel('lag (deg)')
            ax.set_ylabel('xcorr')
            ax.grid(True)

        phase_lag_list.append(phase_lag_deg)
        t_list.append(twin.mean())
        
    if disp:
        plt.show()
    return np.array(t_list), np.array(phase_lag_list)

# -----------------------------------------------------------------------------
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
            'savgol_org': 3,
            }
    scutellum_data = load_scutellum_data(scutellum_data_filename, scutellum_proc_param)
    scutellum_pos, scutellum_pos_hp, scutellum_pos_hp_sg = scutellum_data

    # Remove points form start and finish to remove end effects
    scutum_pos = scutum_pos[n_rm:-n_rm]
    scutellum_pos = scutellum_pos[n_rm:-n_rm]
    scutellum_pos_hp = scutellum_pos_hp[n_rm:-n_rm]
    scutellum_pos_hp_sg = scutellum_pos_hp_sg[n_rm:-n_rm]

    # normalize to range -1, 1
    scutum_pos = normalize(scutum_pos)
    scutellum_pos = normalize(scutellum_pos)
    scutellum_pos_hp = normalize(scutellum_pos_hp)
    scutellum_pos_hp_sg = normalize(scutellum_pos_hp_sg)

    # Get frame #s and time points
    frame = np.arange(len(scutum_pos)) + n_rm
    t = frame*dt

    wb_period = find_period(t,scutum_pos)
    wb_freq = 1.0/wb_period

    # Cross correlate signals to find phase lag
    xcorr = sig.correlate(scutum_pos, scutellum_pos_hp_sg)
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
            scutellum_pos_hp, 
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
    scutellum_line, = ax[0].plot(t, scutellum_pos_hp_sg, 'g')
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






