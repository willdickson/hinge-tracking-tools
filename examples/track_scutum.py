from hinge_tracking_tools import AutoTracker

input_file = 'movies/camera_1.avi'
output_file = 'data/scutum_tracking_data.hdf5'
save_data = False

track_config = { 
        'obj1' : {
            'roi' : {
                'col' : [247, 264],
                'row' : [354, 368],
                },
            'color' : (255,0,0),
            'thickness' : 2,
            'threshold' : 0.15,
            'track_type' : 'top',
            'gaussianBlur' : True,
            },
        'obj2' : {
            'roi' : {
                'col' : [250, 270],
                'row' : [340, 355],
                },
            'color' : (0,255,0),
            'thickness' : 2,
            'threshold' : 0.20,
            'track_type' : 'left',
            'gaussianBlur' : True,
            },
        'obj3' : {
            'roi' : {
                'col' : [270, 290],
                'row' : [300, 315],
                },
            'color' : (0,0,255),
            'thickness' : 2,
            'threshold' : 0.20,
            'track_type' : 'left',
            'gaussianBlur' : True,
            },
        }

options = {
        'show_roi_rect': False,
        'show_roi_images': False,
        'show_tracking': True,
        'wait_dt': 1,
        }

tracker = AutoTracker(input_file, output_file, track_config, options=options, save_data=save_data)
tracker.run()
