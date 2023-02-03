from hinge_tracking_tools import AssistedTracker


input_file = 'movies/camera_1.avi'
output_file = 'data/haltere_tracking_data.pkl'
no_save = False

tracker = AssistedTracker(input_file, output_file, no_save=no_save)
tracker.run()

