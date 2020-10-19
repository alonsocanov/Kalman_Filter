import sys
import numpy as np


class KalmanFilter:
    def __init__(self, x_0:np.ndarray, v_0:np.ndarray, var=10.0) -> None:
        if isinstance(x_0, list):
            x_0 = np.array(x_0, dtype=np.float32)
        if isinstance(v_0, list):
            v_0 = np.array(v_0, dtype=np.float32)
        self._x = np.append(x_0, v_0).reshape(-1, 1)
        self._len_state = self._x.shape[0]


        # State vector
        if isinstance(var, float) or isinstance(var, int):
            self._var = var
        elif isinstance(var, list):
            self._var = np.array(var, dtype=np.float32).reshape(-1, 1)
        elif isinstance(var, np.ndarray):
            self._var = var.astype(np.float32).reshape(-1, 1)
        
        # Covarianve error matrix
        try:
            self._P = np.eye(self._len_state) * self._var
        except ValueError:
            print('Shape error:', self._len_state, '!=', self._var.shape[0])
            self._P = np.eye(self._len_state)
            print('Taking identity matrix of shape:', self._P.shape)
            

    def predict(self, delta_t: float, control:np.ndarray, var: int) -> None:
        state_div = self._len_state // 2
        # State transition matrix
        F = np.eye(self._len_state)
        F[:state_div, state_div:] = np.eye(state_div) * delta_t
        # Control input matrix
        B = np.zeros((self._len_state, state_div))
        B[:state_div, :] = np.eye(state_div) * 0.5 * delta_t**2
        B[state_div:, :] = np.eye(state_div) * delta_t
        if isinstance(control, list):
            control = np.array(control)
        
        control = control.reshape(-1, 1)    
        # Predict estimate
        try:
            x = F.dot(self._x) + B.dot(control)
        except ValueError:
            print('Dimensions dont agree for equation x = Fx + Bu')
            print('dim(F)={0}, dim(x)={1}, dim(B)={2}, dim(u)={3}'.format(F.shape, self._x.shape, B.shape, control.shape))
            sys.exit(1)
        # Process noise covariance matrix
        G = np.ones((self._len_state, 1), dtype=np.float32)
        G[:state_div, 0] *= delta_t**2 * .5
        G[state_div:, 0] *= delta_t**2
        Q = G.dot(G.T) * var
        # Predicted covariance error
        try:
            P = F.dot(self._P).dot(F.T) + Q
        except ValueError:
            print('Dimensions dont agree for equation P = FPF.T + Q')
            print('dim(P)={0}, dim(F)={1}, dim(Q)={2}'.format(P.shape, F.shape, Q.shape))
            sys.exit(1)
        self._x = x
        self._P = P

    def update(self, measurements: np.array, var) -> None:
        # Observation vector
        if isinstance(measurements, list):
            measurements = np.array(measurements)
        z = measurements.reshape(-1, 1)

        if isinstance(var, list):
            var = np.array(var)
        # Mesurement noise covariance matrix
        try:
            R = np.eye(self._len_state) * var
        except ValueError:
            print('Dimensions dont agree for equation R = I var')
            if isinstance(var, np.ndarray):
                print('dim(I)={0}, dim(var)={2}'.format((self._len_state, self._len_state), var.shape))
            sys.exit(1)
        # Measurement transition matrix
        H = np.eye(self._len_state)
        # Measurement residual
        try:
            y = z - H.dot(self._x)
        except ValueError:
            print('Dimensions dont agree for equation y = z - Hx')
            print('dim(z)={0}, dim(H)={1}, dim(x)={2}'.format(z.shape, H.shape, self._x.shape))
            sys.exi(1)

        # Meassurement error
        try:
            S = R + H.dot(self._P).dot(H.T)
        except ValueError:
            print('Dimensions dont agree for equation S = R + HPH.T')
            print('dim(R)={0}, dim(H)={1}, dim(P)={2}'.format(R.shape, H.shape, self._P.shape))
            sys.exit(1)
        # Kalman gain
        try:
            K = self._P.dot(H.T).dot(np.linalg.inv(S))
        except ValueError:
            print('Dimensions dont agree for equation K = PH.T/S')
            print('dim(P)={0}, dim(H)={1}, dim(S)={2}'.format(self._P.shape, H.shape, S.shape))
            sys.exit(1)
            
        # Update state estimate
        try:
            x = self._x + K.dot(y)
        except ValueError:
            print('Dimensions dont agree for equation x = x + Ky')
            print('dim(x)={0}, dim(K)={1}, dim(y)={2}'.format(self._x.shape, K.shape, y.shape))
            sys.exit(1)
        # Update error covariance
        try:
            P = (np.eye(self._len_state) - K.dot(H)).dot(self._P)
        except ValueError:
            print('Dimensions dont agree for equation P = (I - KH)P')
            print('dim(I)={0}, dim(K)={1}, dim(H)={2}, dim(P)={3}'.format((self._len_state, self._len_state), K.shape, H.shape, P.shape))
            sys.exit(1)

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
    def var(self) -> np.array:
        return self._var


'''
class KalmanFilterExt:
    def __init__(self, x_0:np.ndarray, v_0:np.ndarray, var=10.0) -> None:
        if isinstance(x_0, list):
            x_0 = np.array(x_0, dtype=np.float32)
        if isinstance(v_0, list):
            v_0 = np.array(v_0, dtype=np.float32)
        self._x = np.append(x_0, v_0).reshape(-1, 1)
        self._len_state = self._x.shape[0]


        # State vector
        if isinstance(var, float) or isinstance(var, int):
            self._var = var
        elif isinstance(var, list):
            self._var = np.array(var, dtype=np.float32).reshape(-1, 1)
        elif isinstance(var, np.ndarray):
            self._var = var.astype(np.float32).reshape(-1, 1)
        
        # Covarianve error matrix
        try:
            self._P = np.eye(self._len_state) * self._var
        except ValueError:
            print('Shape error:', self._len_state, '!=', self._var.shape[0])
            self._P = np.eye(self._len_state)
            print('Taking identity matrix of shape:', self._P.shape)

    def predict(self, delta_t: float, control:np.ndarray, var: int) -> None:
        state_div = self._len_state // 2
        # State transition matrix
        F = np.eye(self._len_state)
        F[:state_div, state_div:] = np.eye(state_div) * delta_t
        
        # Control input matrix
        B = np.zeros((self._len_state, state_div))
        B[:state_div, :] = np.eye(state_div) * 0.5 * delta_t**2
        B[state_div:, :] = np.eye(state_div) * delta_t
        if isinstance(control, list):
            control = np.array(control)
        
        control = control.reshape(-1, 1)    
        # Predict estimate
        try:
            x = F.dot(self._x) + B.dot(control)
        except ValueError:
            print('Dimensions dont agree for equation x = Fx + Bu')
            print('dim(F)={0}, dim(x)={1}, dim(B)={2}, dim(u)={3}'.format(F.shape, self._x.shape, B.shape, control.shape))
            sys.exit(1)
        # Process noise covariance matrix
        G = np.ones((self._len_state, 1), dtype=np.float32)
        G[:state_div, 0] *= delta_t**2 * .5
        G[state_div:, 0] *= delta_t**2
        Q = G.dot(G.T) * var
        # Predicted covariance error
        try:
            P = F.dot(self._P).dot(F.T) + Q
        except ValueError:
            print('Dimensions dont agree for equation P = FPF.T + Q')
            print('dim(P)={0}, dim(F)={1}, dim(Q)={2}'.format(P.shape, F.shape, Q.shape))
            sys.exit(1)
        self._x = x
        self._P = P

    def update(self, measurements: np.array, var) -> None:
        # Observation vector
        if isinstance(measurements, list):
            measurements = np.array(measurements)
        z = measurements.reshape(-1, 1)

        if isinstance(var, list):
            var = np.array(var)
        # Mesurement noise covariance matrix
        try:
            R = np.eye(self._len_state) * var
        except ValueError:
            print('Dimensions dont agree for equation R = I var')
            if isinstance(var, np.ndarray):
                print('dim(I)={0}, dim(var)={2}'.format((self._len_state, self._len_state), var.shape))
            sys.exit(1)
        # Measurement transition matrix
        H = np.eye(self._len_state)
        # Measurement residual
        try:
            y = z - H.dot(self._x)
        except ValueError:
            print('Dimensions dont agree for equation y = z - Hx')
            print('dim(z)={0}, dim(H)={1}, dim(x)={2}'.format(z.shape, H.shape, self._x.shape))
            sys.exi(1)

        # Meassurement error
        try:
            S = R + H.dot(self._P).dot(H.T)
        except ValueError:
            print('Dimensions dont agree for equation S = R + HPH.T')
            print('dim(R)={0}, dim(H)={1}, dim(P)={2}'.format(R.shape, H.shape, self._P.shape))
            sys.exit(1)
        # Kalman gain
        try:
            K = self._P.dot(H.T).dot(np.linalg.inv(S))
        except ValueError:
            print('Dimensions dont agree for equation K = PH.T/S')
            print('dim(P)={0}, dim(H)={1}, dim(S)={2}'.format(self._P.shape, H.shape, S.shape))
            sys.exit(1)
            
        # Update state estimate
        try:
            x = self._x + K.dot(y)
        except ValueError:
            print('Dimensions dont agree for equation x = x + Ky')
            print('dim(x)={0}, dim(K)={1}, dim(y)={2}'.format(self._x.shape, K.shape, y.shape))
            sys.exit(1)
        # Update error covariance
        try:
            P = (np.eye(self._len_state) - K.dot(H)).dot(self._P)
        except ValueError:
            print('Dimensions dont agree for equation P = (I - KH)P')
            print('dim(I)={0}, dim(K)={1}, dim(H)={2}, dim(P)={3}'.format((self._len_state, self._len_state), K.shape, H.shape, P.shape))
            sys.exit(1)

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
    def var(self) -> np.array:
        return self._var
'''