import sys
import cv2
import pickle
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal as sig
import sklearn.svm as svm


class AssistedTracker:

    def __init__(self, input_file, output_file, no_save=False):
        self.input_file = input_file
        self.output_file = output_file
        self.no_save = no_save

        self.frame_pos = -1
        self.num_frame = -1
        self.cap = None
        self.done = False
        self.frame_bgr = None
        self.window_name = 'frame_bgr'

        self.interpolate = False
        self.interp_kind = 'quadratic'

        self.filter = False
        self.filter_win = 9 
        self.filter_ord = 3

        self.project = False
        self.check_project = False
        self.project_svr_num_lag = 7 
        self.project_svr_kernel = 'rbf'
        self.project_svr_c = 2000.0
        self.project_svr_epsilon = 1.0
        self.project_point = None

        self.key_action_dict = collections.OrderedDict([
                (ord('e'), self.exit_loop),
                (ord('>'), self.step_forward),
                (ord('.'), self.step_forward),
                (ord('<'), self.step_backward),
                (ord(','), self.step_backward),
                (ord('a'), self.add_points),
                (ord('s'), self.save_data),
                (ord('v'), self.view_raw_data),
                (ord('p'), self.toggle_project) ,
                (ord('c'), self.toggle_check_project),
                (ord('i'), self.toggle_interpolate),
                (ord('f'), self.toggle_filter),
                (ord('d'), self.display_tracking_data),
                (ord('r'), self.remove_point),
                (ord('b'), self.backup_data),
                (ord('1'), self.goto_first),
                (ord('0'), self.goto_last),
                (      83, self.shift_point_right),
                (      81, self.shift_point_left),
                (      82, self.shift_point_up),
                (      84, self.shift_point_down),
                (ord('h'), self.print_help),
                ])

        self.key_help_dict = collections.OrderedDict([ 
                ('e', 'exit program') ,
                ('>', '1 step forward'),
                ('<', '1 step backward'),
                ('a', 'append point'),
                ('s', 'save tracking data') ,
                ('v', 'view raw data'),
                ('p', 'toggle project next point'),
                ('c', 'toggle check projections'),
                ('i', 'toggle interpolate missing') ,
                ('f', 'toggle filter (savgol) data'),
                ('d', 'display tracking data'),
                ('r', 'remove tracking point'),
                ('b', 'backup data'),
                ('0', 'goto first tracked frame'),
                ('1', 'goto last tracked frame'),
                ('h', 'show help'),
                ])
        self.load_track_data()

    def load_track_data(self):
        if pathlib.Path('.', self.output_file).exists():
            with open(self.output_file,'rb') as f:
                self.track_data = pickle.load(f)
        else:
            self.track_data = {}

    def save_tracking_data(self,filename):
        if self.no_save:
            return
        with open(filename, 'wb') as f:
            pickle.dump(self.track_data, f)

    def on_mouse_event(self, event, x, y, flags, param):
        match event:
            case cv2.EVENT_LBUTTONUP:
                self.track_data[self.frame_pos] = (float(x),float(y))
                self.frame_bgr = np.copy(self.frame_bgr_orig)
                self.draw_pt(x,y,pt_type='data')

    def draw_pt(self, x, y, pt_type='data'):
        match pt_type:
            case 'data':
                color = (255,0,0)
            case 'proj':
                color = (0,255,255)
            case 'interp':
                color = (0,255,0)
        cv2.circle(self.frame_bgr, (int(x), int(y)), 3, color, thickness=2)
        cv2.imshow(self.window_name, self.frame_bgr)

    def max_tracked_frame(self):
        if self.track_data:
            value = max([n for n in self.track_data])
        else:
            value = -1 
        return value

    def min_tracked_frame(self):
        if self.track_data:
            value = min([n for n in self.track_data])
        else:
            value = -1 
        return value

    def track_data_as_array(self):
        track_tuples = [(i,pos) for i, pos in self.track_data.items()]
        track_tuples.sort()
        ind = np.array([i for i, pos in track_tuples])
        x = np.array([pos[0] for i, pos in track_tuples])
        y = np.array([pos[1] for i, pos in track_tuples])
        return ind, x, y

    def get_num_missing(self):
        data_ind, data_x, data_y = self.track_data_as_array()
        return len(get_missing_ind(data_ind))

    def get_interpolated_data(self, ind=None): 
        data_ind, data_x, data_y = self.track_data_as_array()
        if ind is None: 
            missing = get_missing_ind(data_ind)
            ind_interp = np.array(missing,dtype=np.int64)
        else:
            ind_interp = np.array(ind)
        interp_func_x = interp.interp1d(data_ind, data_x, self.interp_kind)
        interp_func_y = interp.interp1d(data_ind, data_y, self.interp_kind)
        x = interp_func_x(ind_interp)
        y = interp_func_y(ind_interp)
        return ind_interp, x, y

    def get_merged_data(self):
        """ Get merged data, tracking + interpolated.
        """
        ind, xval, yval = self.track_data_as_array()
        ind_interp, xval_interp, yval_interp = self.get_interpolated_data()
        ind_all = np.arange(ind.min(), ind.max()+1)
        xval_all = np.zeros(ind_all.shape)
        yval_all = np.zeros(ind_all.shape)
        xval_all[ind] = xval
        yval_all[ind] = yval
        xval_all[ind_interp] = xval_interp
        yval_all[ind_interp] = yval_interp
        return ind_all, xval_all, yval_all

    def get_filtered_data(self):
        ind, xval, yval = self.get_merged_data()
        xval_filt = sig.savgol_filter(xval, self.filter_win, self.filter_ord)
        yval_filt = sig.savgol_filter(yval, self.filter_win, self.filter_ord)
        return ind, xval_filt, yval_filt

    @property
    def project_condition(self):
        max_ind = self.max_tracked_frame()
        return (self.frame_pos - max_ind) == 1 and self.project

    @property
    def check_project_condition(self):
        max_ind = self.max_tracked_frame()
        return (self.frame_pos - max_ind) <= 0 and self.check_project


    def run(self):
        self.cap = cv2.VideoCapture(self.input_file)
        self.num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.track_data:
            last_pos = self.max_tracked_frame()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, last_pos)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        while not self.done:

            self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if self.frame_pos != last_pos:
                print(f'frame#: {self.frame_pos}/{self.num_frame}')

            last_pos = self.frame_pos
            ret, self.frame_bgr_orig = self.cap.read()
            if not ret:
                break
            self.frame_bgr = np.copy(self.frame_bgr_orig)

            have_pt = False
            try:
                x, y = self.track_data[self.frame_pos]
                have_pt = True
            except KeyError:
                pass

            if have_pt:
                # Draw current data point
                self.draw_pt(x,y,'data')
            else:
                # Interpolate missing data point
                if self.interpolate:
                    min_ind = self.min_tracked_frame()
                    max_ind = self.max_tracked_frame()
                    if self.frame_pos > min_ind and self.frame_pos < max_ind:
                        _, x, y = self.get_interpolated_data([self.frame_pos])
                        self.draw_pt(x[0],y[0],'interp')

            # Reset projected values and check conditions for projection
            self.project_point = None
            if self.project_condition or self.check_project_condition:
                ind, xval, yval = self.get_merged_data()
                #ind, xval, yval = self.get_filtered_data()
                regr_x = svm.SVR(
                        kernel=self.project_svr_kernel, 
                        C=self.project_svr_c, 
                        epsilon=self.project_svr_epsilon
                        )
                regr_y = svm.SVR(
                        kernel=self.project_svr_kernel, 
                        C=self.project_svr_c, 
                        epsilon=self.project_svr_epsilon
                        )

                # Create training data for SVR predictor
                lag = self.project_svr_num_lag
                train_inp = np.zeros((ind.size-lag,2*lag))
                for i in range(lag):
                    train_inp[:,i] = xval[i:i-lag]
                    train_inp[:,i+lag] = yval[i:i-lag]
                train_out_x = xval[lag:]
                train_out_y = yval[lag:]

                # Fit SVR to training data
                regr_x.fit(train_inp, train_out_x)
                regr_y.fit(train_inp, train_out_y)

                # Create input for SVR prediction
                n = self.frame_pos
                if n >= lag:
                    pred_inp = np.zeros((2*lag,))
                    for i in range(lag):
                        pred_inp[i] = xval[n+i-lag]
                        pred_inp[i+lag] = yval[n+i-lag]

                    # Predict x and y location of point
                    x_pred = regr_x.predict([pred_inp])
                    y_pred = regr_y.predict([pred_inp])
                    self.draw_pt(x_pred[0], y_pred[0], 'proj')
                    if self.project_condition:
                        self.project_point = (x_pred[0], y_pred[0])

            cv2.imshow(self.window_name, self.frame_bgr)
            key = cv2.waitKey(0)
            print(f'key = {key}')
            try:
                self.key_action_dict[key]()
            except KeyError:
                self.no_step()

    def exit_loop(self):
        self.done = True
        self.save_data()

    def step_forward(self):
        print('step forward')
        self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if self.frame_pos == self.num_frame:
            self.frame_pos -= 1
            print('no more frames')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)

    def step_backward(self): 
        print('step backward')
        self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2
        self.frame_pos = max(self.frame_pos,0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)

    def display_tracking_data(self):
        print('plotting data')
        self.no_step()
        if self.track_data:
            ind, xval, yval = self.track_data_as_array()

            fig1, ax1 = plt.subplots(2,1,sharex=True)
            ax1[0].plot(ind, xval, 'ob') 
            ax1[0].grid(True)
            ax1[0].set_ylabel('x')
            ax1[1].plot(ind, yval, 'ob')
            ax1[1].grid(True)
            ax1[1].set_ylabel('y')
            ax1[1].set_xlabel('frame')

            fig2, ax2 = plt.subplots(1,1)
            ax2.plot(xval, yval, 'ob')
            ax2.grid(True)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')

            #ok_to_filt = False 
            if self.interpolate:
                ind_interp, xval_interp, yval_interp = self.get_interpolated_data()
                ax1[0].plot(ind_interp, xval_interp, 'og')
                ax1[1].plot(ind_interp, yval_interp, 'og')
                ax2.plot(xval_interp, yval_interp, 'og')

            if self.filter:
                ind_filt, xval_filt, yval_filt = self.get_filtered_data()
                ax1[0].plot(ind_filt, xval_filt, 'b')
                ax1[1].plot(ind_filt, yval_filt, 'b')
                ax2.plot(xval_filt, yval_filt, 'b')

            have_cur_pos = True
            try:
                xcur, ycur = self.track_data[self.frame_pos]
            except KeyError:
                have_cur_pos = False

            if have_cur_pos:
                ax1[0].plot([self.frame_pos],[xcur], 'or')
                ax1[1].plot([self.frame_pos],[ycur], 'or')
                ax2.plot([xcur],[ycur], 'or')

            plt.show()

    def save_data(self):
        self.no_step()
        self.save_tracking_data(self.output_file)
        print(f'saved data to {self.output_file}')

    def add_points(self):
        self.no_step()
        if self.interpolate:
            ind_interp, xval_interp, yval_interp = self.get_interpolated_data()
            for ind, xval, yval in zip(ind_interp, xval_interp, yval_interp):
                self.track_data[ind] = (xval, yval) 
                print(f'added point = ({xval}, {yval})')

        if self.project_point is not None:
            self.track_data[self.frame_pos] = self.project_point
            print(f'added point = {self.project_point}')


    def view_raw_data(self):
        self.no_step()
        min_frame = self.min_tracked_frame()
        max_frame = self.max_tracked_frame()
        for k in range(min_frame,max_frame+1):
            try:
                p = self.track_data[k]
            except KeyError:
                p = None
            if p is not None:
                print(f'frame: {k}, ({p[0]}, {p[1]})')
            else:
                print(f'frame: {k}')

    def toggle_project(self):
        self.no_step()
        self.project = not self.project
        print(f'project = {self.project}')

    def toggle_check_project(self):
        self.no_step()
        self.check_project = not self.check_project
        print(f'check_project = {self.check_project}')

    def toggle_interpolate(self):
        self.no_step()
        self.interpolate = not self.interpolate
        print(f'interpolate = {self.interpolate}')

    def toggle_filter(self):
        self.no_step()
        self.filter = not self.filter
        print(f'filter = {self.filter}')

    def remove_point(self):
        self.no_step()
        try:
            del self.track_data[self.frame_pos]
        except KeyError:
            pass
        print('removed point')

    def backup_data(self):
        self.no_step()
        output_path = pathlib.Path(self.output_file)
        backup_path = pathlib.Path(
                output_path.parent, 
                f'{output_path.stem}_backup{output_path.suffix}'
                )
        backup_file = backup_path.as_posix()
        self.save_tracking_data(backup_file)
        print(f'data backed up to {backup_file}')

    def goto_first(self):
        min_frame = self.min_tracked_frame()
        self.frame_pos = max(min_frame-1, 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)

    def goto_last(self):
        max_frame = self.max_tracked_frame()
        self.frame_pos = min(max_frame, self.num_frame-2)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)

    def shift_point_left(self):
        self.no_step()
        have_point = True
        try:
            x, y = self.track_data[self.frame_pos]
        except KeyError:
            have_point = False
        if have_point:
            self.track_data[self.frame_pos] = x-1, y

    def shift_point_right(self):
        self.no_step()
        have_point = True
        try:
            x, y = self.track_data[self.frame_pos]
        except KeyError:
            have_point = False
        if have_point:
            self.track_data[self.frame_pos] = x+1, y

    def shift_point_up(self):
        self.no_step()
        have_point = True
        try:
            x, y = self.track_data[self.frame_pos]
        except KeyError:
            have_point = False
        if have_point:
            self.track_data[self.frame_pos] = x, y-1

    def shift_point_down(self):
        self.no_step()
        have_point = True
        try:
            x, y = self.track_data[self.frame_pos]
        except KeyError:
            have_point = False
        if have_point:
            self.track_data[self.frame_pos] = x, y+1

    def no_step(self): 
        self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_pos = max(self.frame_pos-1, 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)

    def print_help(self):
        self.no_step()
        print()
        print('command keys')
        print('-'*60)
        for k, v in self.key_help_dict.items():
            print(f'{k} = {v}')
        print()
        print('state')
        print('-'*60)
        print(f'project       = {self.project}')
        print(f'check project = {self.check_project}')
        print(f'interpolate   = {self.interpolate}')
        print(f'filter        = {self.filter}')
        print(f'tracking pts  = {len(self.track_data)}')
        print(f'missing pts   = {self.get_num_missing()}')
        print(f'number frames = {self.num_frame}')
        print()

# Utility functions
# -----------------------------------------------------------------------------

def get_missing_ind(ind):
    ind_missing = []
    for n in range(min(ind), max(ind)):
        if not n in ind:
            ind_missing.append(n)
    return ind_missing


