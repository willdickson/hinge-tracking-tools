import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


def find_period(t, x, guess=1/200.0, disp=False):
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


def find_frequency(t, x, guess=200.0, disp=False):
    period = find_period(t, x, guess=1.0/guess, disp=disp)
    return 1/period


def get_windowed_period(t, x, win, dt, period, disp=False):
    num = len(x)
    t_list = []
    period_list = []
    step = 2*int(period/dt)
    for i in range(1, num-win, step):
        twin = t[i:i+win]
        xwin = x[i:i+win]
        xcorr = sig.correlate(xwin, xwin)
        lag = sig.correlation_lags(len(xwin), len(xwin))
        t_lag = dt*lag
        peaks, _  = sig.find_peaks(xcorr)
        peaks = peaks[1:-1]
        t_peaks = t_lag[peaks]
        xcorr_peaks = xcorr[peaks]
        dt_peaks = t_peaks[1:] - t_peaks[:-1]
        period_list.append(dt_peaks.mean())
        t_list.append(twin.mean())
        if disp:
            fig, ax = plt.subplots(1,1)
            ax.plot(t_lag, xcorr,'b')
            ax.plot(t_lag[peaks], xcorr[peaks], 'or')
            ax.set_xlabel('t (sec)')
            ax.set_ylabel('xcorr')
            ax.grid(True)
            plt.show()
    return np.array(t_list), np.array(period_list)



def get_windowed_phase_lag(t,x,y,win,dt,period,disp=False):
    num = len(x)
    t_list = []
    phase_lag_list = []
    step = 2*int(period/dt)
    if disp:
        fig, ax = plt.subplots(1,1)
    for i in range(1, num-win, step):
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


def normalize(x):
    delta = x.max() - x.min()
    return x/(0.5*delta)
