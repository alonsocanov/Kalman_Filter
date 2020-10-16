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
            kf.predict(1, [0,0], 1)

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
            x = x + v * dt
            t += dt

            meas = np.append(x, v)
            kf.update(meas, 5)
            kf.predict(dt, np.array([0, 0]), 10)

            x_plot = np.append(x_plot, x, axis=1)
            v_plot = np.append(v_plot, v, axis=1)
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
    
            


    # def test_kamaln(self):
    #     print()
    #     print('Test Kalman')
    #     x = np.array([.1, 0.05])
    #     v = np.array([.02, .07])
    #     accel = np.array([0.1, 0.1])
    #     var = .1
    #     meas_var = .1

    #     kf = KalmanFilter(x_0=x, v_0=v, accel=accel)

    #     plt.figure()
    #     delta_t = 0.01
    #     num_steps = 10000

    #     mus = []
    #     covs = []
    #     time = []
    #     meas = []
    #     t = 0
    #     for i in range(num_steps):
    #         covs.append(kf.cov)
    #         mus.append(kf.mean)
    #         # print(mus)

    #         x += delta_t * v + delta_t ** 2 * accel
    #         v += 0.5 * delta_t * accel
    #         meas_x = np.append(x, v)

    #         if i != 0 and i % 1000 == 0:
    #             kf.update(measurements=meas_x, measurements_var=meas_var)
    #             meas.append(meas_x)
    #         else:
    #             kf.predict(delta_t=delta_t, var=var)

    #         time.append(t)
    #         t += delta_t

    #     covs = np.array(covs)[:, :, 0]
    #     mus = np.array(mus)[:, :, 0]
    #     meas = np.array(meas)

    #     uncertanty_pos = mus + 2 * np.sqrt(covs)
    #     uncertanty_neg = mus - 2 * np.sqrt(covs)

    #     fig = plt.figure(num=1, figsize=(10, 10))

    #     ax = fig.add_subplot(2, 2, 1)
    #     ax.set_title('2D Position')

    #     ax.plot(mus[:, 0], mus[:, 1], 'b', linewidth=3.0)
    #     ax.plot(uncertanty_pos[:, 0], uncertanty_pos[:, 1], 'r--')
    #     ax.plot(uncertanty_neg[:, 0], uncertanty_neg[:, 1], 'r--')
    #     ax.plot(meas[:, 0], meas[:, 1], 'g*')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')

    #     ax = fig.add_subplot(2, 2, 2, projection='3d')
    #     ax.set_title('3D Position')
    #     ax.plot3D(mus[:, 0], mus[:, 1], time, 'blue', linewidth=3.0)
    #     ax.plot3D(uncertanty_neg[:, 0], uncertanty_neg[:, 1], time, 'r--')
    #     ax.plot3D(uncertanty_pos[:, 0], uncertanty_pos[:, 1], time, 'r--')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('time')

    #     ax = fig.add_subplot(2, 2, 3)
    #     ax.set_title('2D Velocity')
    #     ax.plot(mus[:, 2], mus[:, 3], 'b', linewidth=3.0)
    #     ax.plot(uncertanty_pos[:, 2], uncertanty_pos[:, 3], 'r--')
    #     ax.plot(uncertanty_neg[:, 2], uncertanty_neg[:, 3], 'r--')
    #     ax.plot(meas[:, 2], meas[:, 3], 'g*')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')

    #     ax = fig.add_subplot(2, 2, 4, projection='3d')
    #     ax.set_title('3D Velocity')
    #     ax.plot3D(mus[:, 2], mus[:, 3], time, 'blue', linewidth=3.0)
    #     ax.plot3D(uncertanty_pos[:, 2], uncertanty_pos[:, 3], time, 'r--')
    #     ax.plot3D(uncertanty_neg[:, 2], uncertanty_neg[:, 3], time, 'r--')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('time')

    #     plt.show()


if __name__ == '__main__':
    unittest.main()

    
