import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

data_file = 'data/scutellum_tracking_data.hdf5'
fs = 10000.0
dt = 1.0/fs
psd_nperseg = 5*1024
psd_peaks_thresh = 1.0e-1
obj_name_list = ['obj1', 'obj2']

for obj_name 

data = h5py.File(data_file, 'r')
obj = data['obj1'][...]

x = obj[:,0]
y = obj[:,1]
x = x - np.median(x)
y = y - np.median(y)

frame = np.arange(len(x))
t = dt*frame

f, psd = sig.welch(y, fs, nperseg=psd_nperseg)
peaks, _ = sig.find_peaks(psd, height=psd_peaks_thresh)

f_prob = f[peaks[0]]
f_flap = f[peaks[1]]
f_filt = f_prob + 0.6*(f_flap - f_prob)
sos = sig.butter(10, f_filt, btype='hp', fs=fs, output='sos')
y_filt = sig.sosfiltfilt(sos,y)

y_filt_sg = sig.savgol_filter(y_filt, 21,3)

print()
print(f'peak 0: {f_prob} (Hz)')
print(f'peak 1: {f_flap} (Hz)')
print(f'f_hp:   {f_filt} (Hz)')
print()

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t, y,'b')
ax[0].set_ylabel('pos (pixel)')
ax[0].grid(True)
ax[1].plot(t, y_filt, 'b')
ax[1].plot(t, y_filt_sg, 'g')
ax[1].set_ylabel('pos hp (pixel)')
ax[1].grid(True)
ax[1].set_xlabel('time (sec)')

fig, ax = plt.subplots(1,1)
ax.semilogy(f, psd)
ax.semilogy(f[peaks], psd[peaks], 'or')
ax.set_ylim([1e-7, 1e2])
ax.set_xlim([0, 1000])
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('PSD [V**2/Hz]')
ax.grid(True)
plt.show()
