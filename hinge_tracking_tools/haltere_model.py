import sys
import pickle
import pathlib
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt


class HaltereModel:

    NumParam = 7
    NumArcPtsFit = 200
    NumArcPtsFind = 1000
    OptInitDefault = {
            'amp': {
                'init': np.deg2rad(100.0),
                'range': {
                    'min': np.deg2rad(20.0),
                    'max': np.deg2rad(200.0),
                    },
                },
            'ang_x' : {
                'init': np.deg2rad(0.0),
                'range': {
                    'min': np.deg2rad(-60.0), 
                    'max': np.deg2rad( 60.0),
                    },
                },
            'ang_y' : {
                'init': np.deg2rad(0.0),
                'range': {
                    'min': np.deg2rad(-60.0), 
                    'max': np.deg2rad( 60.0),
                    },
                },
            'ang_z' : {
                'init': np.deg2rad(0.0),
                'range': {
                    'min': np.deg2rad(-60.0), 
                    'max': np.deg2rad( 60.0),
                    },
                },
            }
    OptParam = { 
            'differential_evolution': { 
                'disp': True, 
                'popsize': 20, 
                'workers': 8, 
                'updating': 'deferred', 
                #'recombination': 0.7,
                #'mutation': (0.6, 1.5),
                'tol': 0.005, 
                }, 
            'shgo' : {
                'disp': False,
                },
            'shgo_differential_evolution': {},
            }

    ParamOrder = ['cx', 'cy', 'r', 'amp', 'ang_x', 'ang_y', 'ang_z']
    ParamAngle = ['amp', 'ang_x', 'ang_y', 'ang_z']
    AutoRangeParam = 2.0


    SolutionFileDefault = 'haltere_model_fit.npy'

    def __init__(self, data_file, solution_file=None, opt_init=None, method='differential_evolution'):
        self.opt_init = {}
        self.solution = {}
        self.tracking_data = {}
        self.load_tracking_data(data_file)
        self.load_solution(solution_file)
        self.autoset_opt_init(opt_init)
        self.method = method

    def autoset_opt_init(self, opt_init):
        self.opt_init = dict(self.OptInitDefault) 
        if opt_init is not None:
            self.opt_init.update(opt_init)

        # Get tracking data bounding box, center and extent
        data_bbox = self.get_tracking_data_bbox()
        cx = 0.5*(data_bbox['x']['max'] + data_bbox['x']['min'])
        cy = 0.5*(data_bbox['y']['max'] + data_bbox['y']['min'])
        dx = data_bbox['x']['max'] - data_bbox['x']['min']
        dy = data_bbox['y']['max'] - data_bbox['y']['min']

        # Set model fitting parameters
        if 'cx' not in self.opt_init:
            self.opt_init['cx'] = {}
            self.opt_init['cx']['init'] = cx
            self.opt_init['cx']['range'] = {
                    'min': cx - self.AutoRangeParam*dx,
                    'max': cx + self.AutoRangeParam*dx,
                    }

        if 'cy' not in self.opt_init:
            self.opt_init['cy'] = {}
            self.opt_init['cy']['init'] = cy
            self.opt_init['cy']['range'] = {
                    'min': cy - self.AutoRangeParam*dy,
                    'max': cy + self.AutoRangeParam*dy,
                    }

        if 'r' not in self.opt_init:
            data_radius = self.get_tracking_data_radius()
            self.opt_init['r'] = {} 
            self.opt_init['r']['init'] = data_radius
            self.opt_init['r']['range'] = {
                    'min': (1.0/self.AutoRangeParam)*data_radius,
                    'max': self.AutoRangeParam*data_radius,
                    }

    def get_tracking_data_bbox(self):
        bbox = {
                'x': { 'min': 0.0, 'max': 0.0 },
                'y': { 'min': 0.0, 'max': 0.0 },
                }
        if self.tracking_data:
            bbox['x']['min'] = self.tracking_data['x'].min()
            bbox['x']['max'] = self.tracking_data['x'].max()
            bbox['y']['min'] = self.tracking_data['y'].min()
            bbox['y']['max'] = self.tracking_data['y'].max()
        return bbox


    def get_tracking_data_radius(self):
        diam = 0.0
        if self.tracking_data:
            x = self.tracking_data['x']
            y = self.tracking_data['y']
            point_list = list(zip(x, y))
            for x0, y0 in point_list[:-1]:
                for x1, y1 in point_list[1:]:
                    dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
                    if dist > diam:
                        diam = dist
        return 0.5*diam 


    def load_tracking_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Get frame, x, y data and sort by frame
        frame = [k for k in data]
        x = [v[0] for (k,v) in data.items()]
        y = [v[1] for (k,v) in data.items()]
        framexy = list(zip(frame,x,y))
        framexy.sort()
        frame = [f for (f,x,y) in framexy]
        x = [x for (f,x,y) in framexy]
        y = [y for (f,x,y) in framexy]

        # Turn into arrays and save in dict
        frame = np.array(frame)
        x = np.array(x)
        y = np.array(y)
        xy = np.array([x, y])
        self.tracking_data = {  
                'file' : filename,
                'frame': frame, 
                'x' : x, 
                'y' : y, 
                'xy': xy, 
                }


    def load_solution(self, solution_file):
        if solution_file is None:
            self.solution_file = self.SolutionFileDefault
        else:
            self.solution_file = solution_file
        solution_file_path = pathlib.Path(self.solution_file)
        if solution_file_path.exists():
            with open(self.solution_file,'rb') as f:
                self.solution = pickle.load(f)


    def save_solution(self):
        if self.solution:
            with open(self.solution_file,'wb') as f:
                pickle.dump(self.solution, f)


    def cost_func(self, vals):
        arc_x, arc_y, _ = get_arc_points(*vals, num_pts=self.NumArcPtsFit)
        arc_xy = np.array([arc_x, arc_y])
        tracking_xy = self.tracking_data['xy']
        return get_pointset_score(arc_xy, tracking_xy)


    def get_bounds(self):
        bounds = []
        for i, name in enumerate(self.ParamOrder):
            lb = self.opt_init[name]['range']['min']
            ub = self.opt_init[name]['range']['max']
            bounds.append((lb,ub))
        return bounds


    def get_x0(self):
        x0 = np.zeros((self.NumParam,))
        for i, name in enumerate(self.ParamOrder):
            x0[i] = self.opt_init[name]['init']
        return x0


    def fit(self):
        bounds = self.get_bounds()
        if self.method == 'differential_evolution':
            options = self.OptParam['differential_evolution']
            res = opt.differential_evolution(self.cost_func, bounds, **options)
        elif self.method == 'shgo':
            options = self.OptParam['shgo']
            res = opt.shgo(self.cost_func, bounds, options=options)
        elif self.method == 'shgo_differential_evolution':
            print()
            print('Running SHGO optimization to get x0 for differential evolution')
            print()
            options = self.OptParam['shgo']
            res = opt.shgo(self.cost_func, bounds, options=options)
            print()
            print('Running differential evolution')
            print()
            options = self.OptParam['differential_evolution']
            options['x0'] = res.x
            res = opt.differential_evolution(self.cost_func, bounds, **options)
        else:
            raise ValueError('unknown optimiztion method')

        self.solution = {'vals': res.x, 'cost': res.fun}
        self.save_solution()
        self.print_solution()
        self.plot_solution()


    def plot_solution(self):
        if self.solution:
            sol = self.solution['vals']
            sol_cx = sol[0]
            sol_cy = sol[1]
            fit_x, fit_y, _  = get_arc_points(*sol, num_pts=self.NumArcPtsFit)
            dat_x = self.tracking_data['x']
            dat_y = self.tracking_data['y']

            fig, ax = plt.subplots(1,1)
            ax.plot(dat_x, dat_y, 'ob')
            ax.plot(fit_x, fit_y, 'g')
            ax.plot([sol_cx], [sol_cy], 'or')
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.show()

    def print_solution(self):
        if self.solution:
            vals = self.solution['vals']
            cost = self.solution['cost']
            print()
            print('solution')
            print('-'*60)
            for name, value in zip(self.ParamOrder, vals):
                if name in self.ParamAngle:
                    print(f'{name}: {np.rad2deg(value)}')
                else:
                    print(f'{name}: {value}')
            if cost is not None:
                print(f'cost: {cost}')
                print()

    def find_haltere_angle(self):
        angle = None
        sol = self.solution['vals']
        arc_x, arc_y, arc_theta = get_arc_points(*sol, num_pts=self.NumArcPtsFind)
        if self.solution:
            xvals = self.tracking_data['x']
            yvals = self.tracking_data['y']
            angle = np.zeros_like(xvals)
            for i in range(0,angle.size):
                x = xvals[i]
                y = yvals[i]
                dist = np.sqrt((x-arc_x)**2 + (y-arc_y)**2)
                imin = dist.argmin()
                angle[i] = arc_theta[imin]
        return angle


    def plot_haltere_angle(self):
        angle = self.find_haltere_angle()
        frame = self.tracking_data['frame']
        angle_deg = np.rad2deg(angle)

        fig, ax = plt.subplots(1,1)
        ax.plot(frame, angle_deg,  '.-')
        ax.set_xlabel('frame #')
        ax.set_ylabel('angle (deg)')
        ax.grid(True)
        plt.show()


    def save_haltere_angle(self, filename, units='deg'):
        angle = self.find_haltere_angle()
        frame = self.tracking_data['frame']
        if units == 'deg':
            angle = np.rad2deg(angle)
        np.save(filename, angle)


# Utility functions
# -----------------------------------------------------------------------------

def get_arc_points(cx, cy, r, amp, ang_x, ang_y, ang_z, num_pts=200):
    s = np.linspace(-0.5,0.5,num_pts)
    theta = amp*s
    x = r*np.cos(theta) 
    y = r*np.sin(theta)
    z = np.zeros_like(x)
    Rx = np.array([
        [1.0,          0.0,           0.0], 
        [0.0, np.cos(ang_x), -np.sin(ang_x)], 
        [0.0, np.sin(ang_x),  np.cos(ang_x)],
        ])
    Ry = np.array([
        [np.cos(ang_y), 0.0, -np.sin(ang_y)], 
        [         0.0, 1.0,           0.0],
        [np.sin(ang_y), 0.0,  np.cos(ang_y)],
        ])
    Rz = np.array([
        [np.cos(ang_z), -np.sin(ang_z), 0.0], 
        [np.sin(ang_z),  np.cos(ang_z), 0.0],
        [         0.0,           0.0, 1.0],
        ])
    xyz = np.array([x,y,z])
    xyz = np.dot(Rz, np.dot(Ry,np.dot(Rx,xyz)))
    return xyz[0,:] + cx, xyz[1,:] + cy, theta


def get_pointset_score(a,b):
    """
    Compute the score for two pointsets a and b. 

    a = 2 x n matrix 
    b = 2 x k matrix

    where column of the datasets are considered to be 2d points.
    """

    # Get distance from pointset a to pointset b
    dist_ab = np.zeros((a.shape[1],))
    for i in range(a.shape[1]):
        dist_ab[i] = ((b - a[:,[i]])**2).sum(axis=0).min()
    dist_ab_max = dist_ab.max()

    # Get distance form pointsset b to pointset a
    dist_ba = np.zeros((b.shape[1],))
    for i in range(b.shape[1]):
        dist_ba[i] = ((a - b[:,[i]])**2).sum(axis=0).min()
    dist_ba_max = dist_ba.max()
    return max(dist_ab_max, dist_ba_max)


def arc_score_example(): 
    cx = 100.0
    cy = 150.0
    r  = 20.0
    amp = np.deg2rad(120.0)
    ang_x = np.deg2rad(0.0)
    ang_y = np.deg2rad(0.0)
    ang_z = np.deg2rad(0.0)
    x0, y0, _ = get_arc_points(cx, cy, r, amp, ang_x, ang_y, ang_z)

    amp = np.deg2rad(120.0)
    ang_x = np.deg2rad(20.0)
    ang_y = np.deg2rad(20.0)
    ang_z = np.deg2rad(0.0)
    x1, y1, _ = get_arc_points(cx, cy, r, amp, ang_x, ang_y, ang_z)

    xy0 = np.array([x0,y0])
    xy1 = np.array([x1,y1])

    score = get_pointset_score(xy0, xy1)
    print(f'score: {score}')

    fig, ax = plt.subplots(1,1)
    ax.plot(x0,y0,'b')
    ax.plot(x1,y1,'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.axis('equal')
    plt.show()











