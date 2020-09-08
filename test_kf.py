from kalman_filter import KalmanFilter
# from kalman_filter_extended import KalmanFilterExt
import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestKF(unittest.TestCase):

    def test_construct_pos_and_vel(self):
        x = np.array([0.2, 0.1, 0.3])
        v = np.array([2.3, .7, .4])
        accel = np.array([0.0, 0.0, 0.0])
        accel_var = .5
        kf = KalmanFilter(x_0=x, v_0=v, accel=accel, accel_var=accel_var)
        print('Initial position:\n', kf.pos)
        print('Initial velocity:\n', kf.vel)
        print('Acceleration:\n', kf.acceleration)

    def test_predict(self):
        print()
        print('Test uncertany in prediction')
        x = np.array([0.2, 0.1, 0.3])
        v = np.array([2.3, .7, .4])
        accel = np.array([0.0, 0.0, 0.0])
        accel_var = .5
        dt = 0.1
        kf = KalmanFilter(x_0=x, v_0=v, accel=accel, accel_var=accel_var)
        for i in range(10):
            kf.predict(delta_t=dt)
            print(''.join(['Det_', str(i), ':']), np.linalg.det(kf.cov))

    def test_update(self):
        print()
        print('Test update')
        x = np.array([0.2, 0.1, 0.3])
        v = np.array([2.3, .7, .4])
        accel = np.array([0.0, 0.0, 0.0])
        accel_var, meas_var = .5, .5
        meas = np.array([2, 3, 5, 1, 2, 4])
        kf = KalmanFilter(x_0=x, v_0=v, accel=accel, accel_var=accel_var)
        kf.update(measurements=meas, measurements_var=meas_var)
        print('Determinant:', np.linalg.det(kf.cov))

    def test_kamaln(self):
        print()
        print('Test Kalman')
        x = np.array([10, 0.5])
        v = np.array([20., .7])
        accel = np.array([1., 0.2])
        accel_var = 1.9
        meas_var = 3.1

        kf = KalmanFilter(x_0=x, v_0=v, accel=accel, accel_var=accel_var)

        plt.figure()
        delta_t = 0.01
        num_steps = 10000

        mus = []
        covs = []

        for i in range(num_steps):
            covs.append(kf.cov)
            mus.append(kf.mean)

            x += delta_t * v + delta_t ** 2 * accel
            v += delta_t * accel
            meas_x = np.append(x, v)

            kf.predict(delta_t=delta_t)
            if i != 0 and i % 1000 == 0:
                kf.update(measurements=meas_x, measurements_var=meas_var)

        covs = np.array(covs)[:, :, 0]
        mus = np.array(mus)[:, :, 0]
        uncertanty_pos = mus + 2 * np.sqrt(covs)
        uncertanty_neg = mus - 2 * np.sqrt(covs)

        plt.subplot(2, 1, 1)
        plt.title('Position')
        plt.plot(mus[:, 0], mus[:, 1], 'b')
        plt.plot(uncertanty_pos[:, 0], uncertanty_pos[:, 1], 'r--')
        plt.plot(uncertanty_neg[:, 0], uncertanty_neg[:, 1], 'r--')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.subplot(2, 1, 2)
        plt.title('Velocity')
        plt.plot(mus[:, 2], mus[:, 3], 'b')
        plt.plot(uncertanty_pos[:, 2], uncertanty_pos[:, 3], 'r--')
        plt.plot(uncertanty_neg[:, 2], uncertanty_neg[:, 3], 'r--')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.show()


if __name__ == '__main__':
    unittest.main()
