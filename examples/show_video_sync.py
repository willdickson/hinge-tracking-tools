from collections import OrderedDict
from hinge_tracking_tools import SyncFramePlayer

data = OrderedDict([
        ('side', {
            'videofile' : 'movies/camera_1.avi',
            'datafile'  : 'data/scutum_tracking_data.hdf5',
            }),
        ('top', {
            'videofile' : 'movies/camera_2.avi',
            'datafile'  : 'data/scutellum_tracking_data.hdf5',
            }),
        ])

options = {
        'step_through' : True,
        'wait_dt'      : 10,
        }

player = SyncFramePlayer(data, options=options)
player.run()



