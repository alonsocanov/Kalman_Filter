import numpy as np


class KalmanFilter:
    def __init__(self, x_0: np.array, v_0: np.array, acceleration: np.array) -> None:
        self._x = np.append(x_0, v_0).reshape(-1, 1)
        self._len_state = self._x.shape[0]
        # State vector
        self._accel = acceleration.reshape(-1, 1)
        # Covarianve error matrix
        self._P = np.eye(self._len_state)

    def predict(self, delta_t: float, acceleration_variance: np.array) -> None:
        state_div = self._len_state // 2
        # State transition matrix
        F = np.eye(self._len_state)
        F[:state_div, state_div:] = np.eye(state_div) * delta_t
        # Control input matrix
        B = np.zeros((self._len_state, state_div))
        B[:state_div, :] = np.eye(state_div) * 0.5 * delta_t**2
        B[state_div:, :] = np.eye(state_div) * delta_t
        # Predict estimate
        x = F.dot(self._x) + B.dot(self._accel)
        # Process noise covariance matrix
        Q = np.eye(self._len_state)
        Q[:state_div, :] *= .25 * delta_t**4
        Q[state_div:, :] *= delta_t**2
        Q *= acceleration_variance.reshape(-1, 1)**2
        # Predicted covariance error
        P = F.dot(self._P).dot(F.T) + Q
        self._P = P
        self._x = x

    def update(self, measurements: np.array, measurements_var: np.array) -> None:
        # Observation vector
        z = measurements.reshape(-1, 1)
        # Mesurement noise covariance matrix
        R = np.eye(self._len_state) * measurements_var.reshape(-1, 1)**2
        # Measurement transition matrix
        H = np.eye(self._len_state)
        # Measurement residual
        y = z - H.dot(self._x)
        # Meassurement error
        S = R + H.dot(self._P).dot(H.T)
        # Kalman gain
        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        # Update state estimate
        x = self._x + K.dot(y)
        # Update error covariance
        P = (np.eye(self._len_state) - K.dot(H)).dot(self._P)

        self._x = x
        self._P = P

    @ property
    def cov(self) -> np.array:
        return self._P

    @ property
    def mean(self) -> np.array:
        return self._x

    @ property
    def pos(self) -> np.array:
        return self._x[:self._len_state // 2, 0]

    @ property
    def vel(self) -> np.array:
        return self._x[self._len_state // 2:, 0]

    @ property
    def acceleration(self) -> np.array:
        return self._accel
