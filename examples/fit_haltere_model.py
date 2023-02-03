from hinge_tracking_tools import HaltereModel

data_file = 'data/haltere_tracking_data.pkl'
solution_file = 'data/haltere_model_fit.npy'
method = 'shgo_differential_evolution'
fit = False
save = False

model = HaltereModel(data_file, solution_file=solution_file, method=method)
if fit:
    model.fit()
model.plot_solution()
model.plot_haltere_angle()
if save:
    model.save_haltere_angle('haltere_angle.npy')

