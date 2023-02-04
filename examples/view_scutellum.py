from hinge_tracking_tools import AssistedTracker


input_file = 'movies/camera_2.avi'
output_file = 'data/scutellum_tracking_data.pkl'
no_save = True 

tracker = AssistedTracker(input_file, output_file, no_save=no_save)
tracker.run()

