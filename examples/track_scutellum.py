from hinge_tracking_tools import AutoTracker

input_file = 'movies/camera_2.avi'
output_file = 'data/scutellum_tracking_data.hdf5'
save_data = False 

track_config = { 
        'obj1' : {
            'roi' : {
                'col' : [356, 366],
                'row' : [80, 105],
                },
            'color' : (255,0,0),
            'thickness' : 2,
            'threshold' : 0.4,
            'track_type' : 'bottom',
            'gaussianBlur': True,
            },
        }

options = {
        'show_roi_rect': False,
        'show_roi_images': False,
        'show_tracking': True,
        'wait_dt': 5,
        }

tracker = AutoTracker(input_file, output_file, track_config, options=options,save_data=save_data)
tracker.run()
