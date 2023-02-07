import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from hinge_tracking_tools import find_period
from hinge_tracking_tools import get_windowed_period

data_file = 'data/scutum_tracking_data.hdf5'

filt_win = 21 
filt_ord = 3
fs = 10000.0
dt = 1/fs

data = h5py.File(data_file, 'r')

obj1 = data['obj1'][...]
obj2 = data['obj2'][...]
obj3 = data['obj3'][...]

x1 = obj1[:,0]
y1 = obj1[:,1]

x2 = obj2[:,0]
y2 = obj2[:,1]

x3 = obj3[:,0]
y3 = obj3[:,1]

t = dt*np.arange(len(x3))

diff_12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
diff_13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
diff_12 = diff_12 - np.mean(diff_12)
diff_13 = diff_13 - np.mean(diff_13)

diff_12_filt = sig.savgol_filter(diff_12,filt_win,filt_ord)
diff_13_filt = sig.savgol_filter(diff_13,filt_win,filt_ord)

period = find_period(t, diff_13_filt, guess=1/200.0, disp=False)
print(f'period: {period}')
print(f'frequency: {1.0/period}')

win = 5*int(fs*period)
period_data_t, period_data_x = period_list = get_windowed_period(t, diff_13_filt, win, dt, step=10, disp=False)


if 0:
    fig, ax = plt.subplots(1,1)
    ax.plot(diff_12_filt,'-b')
    ax.grid(True)
    ax.plot(diff_13_filt,'-r')
    ax.grid(True)
    ax.set_xlabel('frame')
    plt.show()

if 0:
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(diff_12, color='lightblue')
    ax[0].plot(diff_12_filt, color='blue')
    ax[0].grid(True)
    ax[0].set_ylabel('y1-y2')
    ax[1].plot(diff_13, color='lightgreen')
    ax[1].plot(diff_13_filt, color='green')
    ax[1].grid(True)
    ax[1].set_ylabel('y1-y3')
    ax[1].set_xlabel('frame')
    plt.show()

if 1:

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(diff_12, color='lightblue')
    ax[0].plot(diff_12_filt, color='blue')
    ax[0].grid(True)
    ax[0].set_ylabel('y1-y2')
    ax[1].plot(diff_13, color='lightgreen')
    ax[1].plot(diff_13_filt, color='green')
    ax[1].grid(True)
    ax[1].set_ylabel('y1-y3')
    ax[0].set_title('scutum motion')
    plt.show()

if 1:
    fig, ax = plt.subplots(1,1)
    ax.plot(period_data_t, 1.0/period_data_x, 'b')
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('period (sec)')
    ax.grid(True)
    plt.show()

