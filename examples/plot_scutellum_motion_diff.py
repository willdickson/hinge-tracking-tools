import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

data_file = 'data/scutellum_tracking_data.hdf5'
fs = 10000.0
dt = 1.0/fs
sg_win = 21
sg_ord = 3
obj_name = 'obj2'
ref_name = 'obj3'

data = h5py.File(data_file, 'r')
obj = data[obj_name][...]
ref = data[ref_name][...]

x = obj[:,0]
y = obj[:,1]
x_ref = ref[:,0]
y_ref = ref[:,1]
x = x - x_ref
y = y - y_ref

x = x - np.median(x)
y = y - np.median(y)

y_filt = sig.savgol_filter(y, sg_win, sg_ord)

frame = np.arange(len(x))
t = dt*frame

fig, ax = plt.subplots(1,1, sharex=True)
ax.plot(t, y,'b')
ax.plot(t, y_filt,'g')
ax.set_ylabel('pos (pixel)')
ax.grid(True)
ax.set_xlabel('time (sec)')

plt.show()
