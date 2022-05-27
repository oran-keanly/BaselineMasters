from scipy.interpolate import UnivariateSpline
from scipy import optimize
from scipy.interpolate import Rbf
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time

class Curvature:
    """
    Class for computing curvature of ordered list of points on a plane
    """
    def __init__(self, trace, interpolate=True):

        self.trace = np.array(trace)
        self.interpolate = interpolate
        self.curvature = None

    @staticmethod
    def _get_twice_triangle_area(a, b, c):

        if np.all(a == b) or np.all(b == c) or np.all(c == a):
            exit('CURVATURE:\nAt least two points are at the same position')

        twice_triangle_area = (b[0] - a[0])*(c[1] - a[1]) - (b[1]-a[1]) * (c[0]-a[0])

        if twice_triangle_area == 0:
            warnings.warn('Collinear consecutive points found: '
                          '\na: {}\t b: {}\t c: {}'.format(a, b, c))

        return twice_triangle_area

    def _get_menger_curvature(self, a, b, c):

        menger_curvature = (2 * self._get_twice_triangle_area(a, b, c) /
                            (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)))
        return menger_curvature

    def calculate_curvature(self, interpolation_target_n=500):
        if self.interpolate:
            self.trace = self.interpolate_trace(self.trace, interpolation_target_n)

        self.curvature = np.zeros(len(self.trace) - 2)
        for point_index in range(len(self.curvature)):
            triplet = self.trace[point_index:point_index+3]
            self.curvature[point_index-1] = self._get_menger_curvature(*triplet)
        return self.curvature

    def plot_curvature(self):
        fig, _ = plt.subplots(figsize=(8, 7))
        _.plot(self.trace[1:-1, 0], self.curvature, 'r-', lw=2)
        _.set_title('Corresponding Menger\'s curvature'.format(len(self.curvature)))
        plt.show()
        fig.savefig(os.path.join('images', 'Curvature.png'))
        return _
    
    def interpolate_trace(self, trace, target_n=500):

        n_trace_points = len(trace)
        x_points = [x[0] for x in trace]
        y_points = [y[1] for y in trace]

        positions = np.arange(n_trace_points)  # strictly monotonic, number of points in single trace
        interpolation_base = np.linspace(0, n_trace_points, target_n)

        # Radial basis function interpolation 'quintic': r**5 where r is the distance from the next point
        # Smoothing is set to length of the input data
        rbf_x = Rbf(positions, x_points, smooth=len(positions), function='quintic')
        rbf_y = Rbf(positions, y_points, smooth=len(positions), function='quintic')

        # Interpolate based on the RBF model
        x_interpolated = rbf_x(interpolation_base)
        y_interpolated = rbf_y(interpolation_base)

        interpolated_trace = np.array([[x, y] for x, y in zip(x_interpolated, y_interpolated)])

        return interpolated_trace



class GradientCurvature:

    def __init__(self, trace, interpolate=True, plot_derivatives=True):
        self.trace = trace
        self.plot_derivatives = plot_derivatives
        self.interpolate = interpolate
        self.curvature = None

    def _get_gradients(self):
        self.x_trace = [x[0] for x in self.trace]
        self.y_trace = [y[1] for y in self.trace]

        x_prime = np.gradient(self.x_trace)
        y_prime = np.gradient(self.y_trace)
        x_bis = np.gradient(x_prime)
        y_bis = np.gradient(y_prime)

#         if self.plot_derivatives:
#             plt.subplot(211)
#             plt.plot(x_prime, label='x\'')
#             plt.plot(y_prime, label='y\'')
#             plt.title('First spatial derivative')
#             plt.legend()
#             plt.subplot(212)
#             plt.plot(x_bis, label='x\'\'')
#             plt.plot(y_bis, label='y\'\'')
#             plt.title('Second spatial derivative')
#             plt.legend()

        return x_prime, y_prime, x_bis, y_bis

    def calculate_curvature(self, interpolation_target_n=500):
        if self.interpolate:
            self.trace = self.interpolate_trace(self.trace, interpolation_target_n)
        x_prime, y_prime, x_bis, y_bis = self._get_gradients()
        curvature = x_prime * y_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2)) - \
            y_prime * x_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2))  # Numerical trick to get accurate values
        self.curvature = curvature
        return curvature

    def interpolate_trace(self, trace, target_n=500):
        n_trace_points = len(trace)
        x_points = [x[0] for x in trace]
        y_points = [y[1] for y in trace]

        positions = np.arange(n_trace_points)  # strictly monotonic, number of points in single trace
        interpolation_base = np.linspace(0, n_trace_points, target_n)

        # Radial basis function interpolation 'quintic': r**5 where r is the distance from the next point
        # Smoothing is set to length of the input data
        rbf_x = Rbf(positions, x_points, smooth=len(positions), function='quintic')
        rbf_y = Rbf(positions, y_points, smooth=len(positions), function='quintic')

        # Interpolate based on the RBF model
        x_interpolated = rbf_x(interpolation_base)
        y_interpolated = rbf_y(interpolation_base)

        interpolated_trace = np.array([[x, y] for x, y in zip(x_interpolated, y_interpolated)])

        return interpolated_trace
