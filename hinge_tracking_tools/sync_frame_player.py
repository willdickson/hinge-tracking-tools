import sys
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from collections import OrderedDict


class SyncFramePlayer:

    def __init__(self, data, options=None):
        self.data = data 
        if options is None:
            self.options = {}
        else:
            self.options = options
        self.key_actions = {
                ord(',') : self.step_backward,
                ord('.') : self.step_forward,
                ord('-') : self.decrease_step,
                ord('=') : self.increase_step,
                }
        self.cap_dict = None
        self.num_frame = None
        self.pos_dict = None 
        self.step = 1

    def run(self):
        self.cap_dict = OrderedDict() 
        self.pos_dict = OrderedDict()
        num_frame_list = []
        for name, info in self.data.items():
            self.cap_dict[name] = cv2.VideoCapture(info['videofile'])
            self.pos_dict[name] = int(self.cap_dict[name].get(cv2.CAP_PROP_POS_FRAMES))
            num_frame = int(self.cap_dict[name].get(cv2.CAP_PROP_FRAME_COUNT))
            num_frame_list.append(num_frame)
        self.num_frame = min(num_frame_list)

        while True:
            frame_dict = OrderedDict([(name,None) for name in self.cap_dict])
            for name, cap in self.cap_dict.items():
                self.pos_dict[name] = int(self.cap_dict[name].get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame_bgr = cap.read()
                if not ret:
                    raise RuntimeError('failed to read frame')
                frame_dict[name] = frame_bgr
            if any([v is None for v in frame_dict.values()]):
                continue 

            nrow, ncol,  nchan, dtype = get_frame_info(frame_dict)
            subframe_ind = get_subframe_indices(frame_dict)
            merged_frame = np.zeros((nrow,ncol,nchan), dtype=dtype)
            for name, frame in frame_dict.items():
                n, m = subframe_ind[name]
                merged_frame[:frame.shape[1],n:m] = frame

            print(f'pos: {[(k,v) for k,v in self.pos_dict.items()]}')
            cv2.imshow('merged frame', merged_frame)
            if self.options.get('step_through',False):
                wait_dt = 0
            else:
                wait_dt = self.options.get('wait_dt', 1)
            key = cv2.waitKey(wait_dt)
            print(key)
            if key == ord('q'):
                break
            else:
                try:
                    self.key_actions[key]()
                except KeyError:
                    pass

    def no_step(self):
        for name, cap in self.cap_dict.items():
            self.pos_dict[name] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos_dict[name])


    def step_forward(self):
        print('step forward')
        for name, cap in self.cap_dict.items():
            self.pos_dict[name] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + self.step-1
            if self.pos_dict[name] >= self.num_frame:
                self.pos_dict[name] = self.num_frame - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos_dict[name])

    def step_backward(self): 
        print('step backward')
        for name, cap in self.cap_dict.items():
            self.pos_dict[name] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2
            self.pos_dict[name] = max(self.pos_dict[name],0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos_dict[name])

    def decrease_step(self):
        self.no_step()
        self.step = max(1, self.step-1)
        print(f'step = {self.step}')

    def increase_step(self):
        self.no_step()
        self.step += 1
        print(f'step = {self.step}')


# -----------------------------------------------------------------------------------

def get_frame_info(frame_dict):
    ncol = sum([frame.shape[1] for name, frame in frame_dict.items()])
    nrow = max([frame.shape[0] for name, frame in frame_dict.items()])
    nchan = list(frame_dict.values())[0].shape[2]
    dtype = list(frame_dict.values())[0].dtype
    return nrow, ncol, nchan, dtype

def get_subframe_indices(frame_dict):
    name_list = [name for name, _ in frame_dict.items()]
    ind = [frame.shape[1] for _, frame in frame_dict.items()]
    ind.insert(0,0)
    ind = np.array(ind)
    ind = ind.cumsum()
    ind_pairs = list(zip(ind[:-1], ind[1:]))
    return OrderedDict(zip(name_list, ind_pairs))














