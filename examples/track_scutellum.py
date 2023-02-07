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
        'obj2' : {
            'roi' : {
                'col' : [377, 397],
                'row' : [95, 125],
                },
            'color' : (0,255,0),
            'thickness' : 2,
            'threshold' : 0.15,
            'track_type' : 'bottom',
            'gaussianBlur': True,
            },
        'obj3' : {
            'roi' : {
                'col' : [385, 415],
                'row' : [260, 285],
                },
            'color' : (0,0,255),
            'thickness' : 2,
            'threshold' : 0.15,
            'track_type' : 'bottom',
            'gaussianBlur': True,
            },
        }

options = {
        'show_roi_rect': True,
        'show_roi_images': True,
        'show_tracking': True,
        'wait_dt': 2,
        'step_through': False,
        }

tracker = AutoTracker(input_file, output_file, track_config, options=options,save_data=save_data)
tracker.run()
