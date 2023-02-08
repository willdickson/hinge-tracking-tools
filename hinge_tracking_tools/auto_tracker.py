import sys
import cv2
import h5py
import numpy as np


class AutoTracker:

    def __init__(self, input_file, output_file, track_config, options={}, save_data=False):
        self.input_file = input_file
        self.output_file = output_file
        self.track_config = track_config
        self.save_data = save_data
        self.options = options

    def run(self):
        tracking_data = {name : [] for name in self.track_config}
        cap = cv2.VideoCapture(self.input_file)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f'frame#: {frame_pos}/{num_frame}')
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            for obj_name, track_info in self.track_config.items():
                c0, c1 = track_info['roi']['col']
                r0, r1 = track_info['roi']['row']
                roi = frame[r0:r1,c0:c1]
                if track_info.get('gaussianBlur', False):
                    roi = cv2.GaussianBlur(roi,(3,3),0)
                min_val = float(roi.min())
                max_val = float(roi.max())
                threshold = int(min_val + track_info['threshold']*(max_val - min_val))
                ret, roi_thresh = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)

                mask_thresh = roi_thresh > 0
                indx = np.arange(roi_thresh.shape[1])
                indy = np.arange(roi_thresh.shape[0])
                indx, indy = np.meshgrid(indx, indy)
                if track_info['track_type'] == 'top':
                    ymin = indy[mask_thresh].min()
                    mask_min = np.logical_and(indy == ymin, mask_thresh)
                    roi_objx = int(indx[mask_min].mean())
                    roi_objy = int(indy[mask_min].mean())
                elif track_info['track_type'] == 'bottom':
                    ymax = indy[mask_thresh].max()
                    mask_max = np.logical_and(indy == ymax, mask_thresh)
                    roi_objx = int(indx[mask_max].mean())
                    roi_objy = int(indy[mask_max].mean())
                elif track_info['track_type'] == 'left':
                    xmin = indx[mask_thresh].min()
                    mask_min = np.logical_and(indx == xmin, mask_thresh)
                    roi_objx = indx[mask_min].mean()
                    roi_objy = indy[mask_min].mean()
                elif track_info['track_type'] == 'right':
                    xmax = indx[mask_thresh].max()
                    mask_max = np.logical_and(indx == xmax, mask_thresh)
                    roi_objx = indx[mask_max].mean()
                    roi_objy = indy[mask_max].mean()

                objx = roi_objx + c0
                objy = roi_objy + r0
                tracking_data[obj_name].append((objx, objy))

                color = track_info['color']
                thickness = track_info['thickness']

                if self.options.get('show_roi_rect', False):
                    cv2.rectangle(frame_bgr, (c0,r0), (c1,r1), color, thickness=thickness) 
                    cv2.circle(roi, (int(roi_objx), int(roi_objy)), 1, (255,0,0), thickness=1)
                
                if self.options.get('show_tracking', False):
                    cv2.circle(frame_bgr, (int(objx), int(objy)), 2, color, thickness=thickness)

                if self.options.get('show_roi_images', False):
                    cv2.imshow(f'roi {obj_name}', roi)
                    cv2.imshow(f'roi thresh {obj_name}', roi_thresh)

            cv2.imshow('frame_bgr', frame_bgr)

            if self.options.get('step_through', False):
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == 83:
                    print('fwd')
                    frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if frame_pos == num_frame:
                        print('no more frames')
                elif key == 81:
                    frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES) - 2
                    frame_pos = max(frame_pos,0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    print('bck')
            else:
                if cv2.waitKey(self.options.get('wait_dt',1)) == ord('q'):
                    break

        if self.save_data:
            h5file = h5py.File(self.output_file, 'w')
            for name, xy in tracking_data.items():
                h5file.create_dataset(name, data=np.array(xy))
            h5file.close()

        cap.release()
        cv2.destroyAllWindows()

        


