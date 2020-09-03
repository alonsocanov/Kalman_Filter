from kalman_filter import KalmanFilter
from kalman_filter_extended import KalmanFilterExt
import unittest
import numpy as np


class TestKF(unittest.TestCase):

    def test_construct_pos_and_vel(self):
        x = np.array([0.2, 0.1, 0.3])
        v = np.array([2.3, .7, .4])
        accel = np.array([3, 1, 2])
        kf = KalmanFilter(x_0=x, v_0=v, acceleration=accel)
        print('Initial position:\n', kf.pos)
        print('Initial velocity:\n', kf.vel)
        print('Acceleration:\n', kf.acceleration)

    def test_predict(self):
        print()
        print('Test uncertany in prediction')
        x = np.array([0.2, 0.1, 0.3])
        v = np.array([2.3, .7, .4])
        accel = np.array([3, 1, 2])
        dt = 0.1
        accel_var = np.array([1, 1, 1, .3, .3, .3])
        kf = KalmanFilter(x_0=x, v_0=v, acceleration=accel)
        for i in range(10):
            kf.predict(delta_t=dt, acceleration_variance=accel_var)
            print(''.join(['Det_', str(i), ':']), np.linalg.det(kf.cov))

    def test_update(self):
        print()
        print('Test update')
        x = np.array([0.2, 0.1, 0.3])
        v = np.array([2.3, .7, .4])
        accel = np.array([3, 1, 2])
        meas = np.array([2, 3, 5, 1, 2, 4])
        meas_var = np.array([.01, .01, .01, .01, .01, .01])
        kf = KalmanFilter(x_0=x, v_0=v, acceleration=accel)
        kf.update(measurements=meas, measurements_var=meas_var)
        print('Determinant:', np.linalg.det(kf.cov))


if __name__ == '__main__':
    unittest.main()
