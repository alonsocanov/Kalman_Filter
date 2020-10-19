from kalman_filter import KalmanFilter
import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class TestKF(unittest.TestCase):

    def test_construct_pos_and_vel(self):
        print()
        print('Test construction of kalman')
        x = [0.2, 0.1]
        v = np.array([2.3, .7])
        var = 1.0
        kf = KalmanFilter(x_0=x, v_0=v, var=var)
        print('Initial position:\n', kf.pos)
        print('Initial velocity:\n', kf.vel)
        print('Variance:\n', kf.var)

    def test_check_predict_construction(self):
        print()
        print('Test construction of prediction')
        x = np.array([0.2, 0.1])
        v = np.array([2.3, .7])
        control = np.array([0.2, 0.1])
        var = .5
        dt = 0.1
        kf = KalmanFilter(x_0=x, v_0=v, var=var)
        kf.predict(delta_t=dt, control=control, var=var)

    def test_predict(self):
        print()
        print('Test uncertany in prediction')
        x = np.array([0.2, 0.1])
        v = np.array([2.3, .7])
        control = np.array([0.2, 0.1])
        var = .5
        dt = .5
        kf = KalmanFilter(x_0=x, v_0=v, var=var)
        for i in range(100):
            kf.predict(delta_t=dt, control=control, var=var)
            if i % 10 == 0:
                print(''.join(['Det_', str(i), ':']), np.linalg.det(kf.cov))

    def test_update(self):
        print()
        print('Test update')
        x = np.array([0.2, 0.1])
        v = np.array([2.3, .7])
        control = np.array([0.2, 0.1])
        var = .5
        dt = .5
        meas_var = .5
        meas = np.array([0.2, 0.1, 2.5, .7])
        kf = KalmanFilter(x_0=x, v_0=v, var=var)
        kf.update(measurements=meas, var=meas_var)
        print('Determinant:', np.linalg.det(kf.cov))
    
    def test_voltage(self):
        print()
        print('Test Voltage')
        t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        voltage = [0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45]
        v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        v_est = list()

        kf = KalmanFilter([0], [0], 10)
        for i in range(len(voltage)):
            meas = [voltage[i], v[0]]
            kf.update(meas, var=.1)
            v_est.append(kf.pos[0])
            kf.predict(1, np.array([0]), 1)

        fig = plt.figure(num=1, figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Voltage')
        ax.plot(t, voltage,'black', linewidth=1.0)
        ax.plot(t, v_est, linewidth=1.0)
        ax.legend(['$v$', '$\hat{v}$'])
        ax.set_xlabel('x')
        ax.set_ylabel('volatge')
        # plt.show()

    
    def test_kalman(self):
        x = np.array([0, 0]).reshape(-1, 1)
        v = np.array([10, 15]).reshape(-1, 1)
        dt = .1
        num_steps = 500
        x_plot = x
        v_plot = v
        t_plot = [0]
        t = 0
        kf = KalmanFilter(x-30, v+20, 10)
        k_pos = kf.pos.reshape(-1, 1)
        k_vel = kf.vel.reshape(-1, 1)
        for i in range(num_steps):
            random = (np.random.rand(2, 1) - .5) * 2
            x = x + random + (v + random) * dt
            t += dt


            meas = np.append(x, v + random)
            kf.update(meas, 5)
            kf.predict(dt, np.array([0, 0]), 10)

            x_plot = np.append(x_plot, x + random, axis=1)
            v_plot = np.append(v_plot, v + random, axis=1)
            t_plot.append(t)
            k_pos = np.append(k_pos, kf.pos.reshape(-1, 1), axis=1)
            k_vel = np.append(k_vel, kf.vel.reshape(-1, 1), axis=1)
                

        fig = plt.figure(num=1, figsize=(10, 7))
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title('Position')
        ax.plot(x_plot[0, :], x_plot[1, :], 'black', linewidth=3.0)
        ax.plot(k_pos[0, :], k_pos[1, :], linewidth=1.5)
        ax.legend(['$pos$', '$\hat{pos}$'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(2, 1, 2)
        ax.set_title('Velocity')
        ax.plot(t_plot, v_plot[0, :], 'black', linewidth=3.0)
        ax.plot(t_plot, v_plot[1, :], 'black', linewidth=3.0)
        ax.plot(t_plot, k_vel[0, :], linewidth=1.5)
        ax.plot(t_plot, k_vel[1, :], linewidth=1.5)

        ax.legend(['$\dot{x}$', '$\dot{y}$', '$\hat{\dot{x}}$', '$\hat{\dot{y}}$'])
        ax.set_xlabel('time')
        ax.set_ylabel('velocity')

        plt.show()
    
    

if __name__ == '__main__':
    unittest.main()

    
