import numpy as np
import control
from scipy import signal
import slycot
import math

"""
This module impelments an SE2 based rover controler.
"""
import math
import numpy as np


def matrix_exp(A, n=30):
    s = np.zeros((3, 3))
    A_i = np.eye(3)
    for i in range(n):
        s = s + A_i/math.factorial(i)
        A_i = A_i@A
    return s


def check_shape(a, shape):
    if np.shape(a) != shape:
        raise IOError(str(np.shape(a)) + '!=' + str(shape))


def wrap(x):
    return np.where(np.abs(x) >= np.pi, (x + np.pi) % (2 * np.pi) - np.pi, x)


class LieGroup:
    
    def __repr__(self):
        return repr(self.matrix)

    def __mul__(self, other):
        return NotImplementedError('')

    
class LieAlgebra:
    
    def __repr__(self):
        return repr(self.wedge)

    def __mul__(self, other):
        return NotImplementedError('')


class Vector:
    
    def __repr__(self):
        return repr(self.matrix)

    def __mul__(self, other):
        return NotImplementedError('')


class R2(Vector):
    
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    @property
    def matrix(self):
        return np.array([[self.x], [self.y]])

    def __neg__(self):
        return R2(x=-self.x, y=-self.y)
    
    def __add__(self, other):
        return R2(x=self.x + other.x, y=self.y + other.y)

    @classmethod
    def from_vector(cls, a):
        a = a.reshape(-1)
        return cls(x=a[0], y=a[1])

    
class so2(LieAlgebra):
    
    def __init__(self, theta):
        self.theta = np.reshape(wrap(theta), ())
    
    @property
    def wedge(self):
        return np.array([
            [0, -self.theta],
            [self.theta, 0]
        ])
    
    @property
    def vee(self):
        return np.array([self.theta])

    @property
    def exp(self):
        return SO2(theta=self.theta)
    

class SO2(LieGroup):
    
    def __init__(self, theta):
        self.theta = np.reshape(wrap(theta), ())
    
    @classmethod
    def one(cls):
        return cls(theta=0)

    @property
    def inv(self):
        return SO2(theta=-self.theta)

    @property
    def matrix(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]
        ])
    
    @property
    def params(self):
        return np.array([self.theta])

    @property
    def log(self):
        return so2(self.theta)
    
    @classmethod
    def from_matrix(cls, a):
        check_shape(a, (2, 2))
        return cls(theta=np.arctan2(a[1, 0], a[0, 0]))

    def __matmul__(self, other):
        if isinstance(other, R2):
            return R2.from_vector(self.matrix@other.matrix)
        elif isinstance(other, SO2):
            return SO2(theta=self.theta + other.theta)


class se2(LieAlgebra):
    
    def __init__(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def __neg__(self):
        return se2(-self.x, -self.y, -self.theta)

    @property
    def wedge(self):
        return np.array([
            [0, -self.theta, self.x],
            [self.theta, 0, self.y],
            [0, 0, 0]
        ])
    

    def __add__(self, other):
        return se2(x=self.x + other.x, y=self.y + other.y, theta=self.theta + other.theta)
    
    def __sub__(self, other):
        return se2(x=self.x - other.x, y=self.y - other.y, theta=self.theta - other.theta)
    
    @property
    def vee(self):
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_vector(cls, a):
        a = a.reshape((3, 1))
        return cls(x=a[0], y=a[1], theta=a[2])
    
    @classmethod
    def from_matrix(cls, a):
        check_shape(a, (3, 3))
        return cls(x=a[0, 2], y=a[1, 2], theta=a[1, 0])

    @property
    def ad_matrix(self):
        x, y, theta = self.x, self.y, self.theta
        return np.array([
            [0, -theta, y],
            [theta, 0, -x],
            [0, 0, 0]
        ])

    def __matmul__(self, other):
        return se2.from_vector(self.ad_matrix@other.vee)

    @property
    def exp(self):
        theta = self.theta
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/12, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V = np.array([[a, -b], [b, a]])
        p = V@np.array([self.x, self.y])
        return SE2(x=p[0], y=p[1], theta=self.theta)

    def __rmul__(self, scalar):
        s = np.reshape(scalar, ())
        return se2(x=self.x*s, y=self.y*s, theta=self.theta*s)


class SE2(LieGroup):
    
    def __init__(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap(float(theta))
    
    @classmethod
    def one(cls):
        return cls(x=0, y=0, theta=0)

    @property
    def params(self):
        return np.array([self.x, self.y, self.theta])
    
    @property
    def matrix(self):
        x, y, theta = self.x, self.y, self.theta
        return np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])

    @property
    def R(self):
        return SO2(theta=self.theta)

    @property
    def p(self):
        return R2(x=self.x, y=self.y)
    
    @property
    def inv(self):
        p = -(self.R.inv@self.p)
        return SE2(x=p.x, y=p.y, theta=-self.theta)

    def __matmul__(self, other: 'SE2'):
        p = self.R@other.p + self.p
        return SE2(x=p.x, y=p.y, theta=self.theta + other.theta)

    @classmethod
    def from_matrix(cls, a: np.array):
        check_shape(a, (3, 3))
        return SE2(theta=np.arctan2(a[1, 0], a[0, 0]),
                   x=a[0, 2], y=a[1, 2])
    
    @classmethod
    def from_vector(cls, a):
        a = a.reshape((3, 1))
        return cls(x=a[0], y=a[1], theta=a[2])

    @property
    def Ad_matrix(self):
        x, y, theta = self.x, self.y, self.theta
        return np.array([
            [np.cos(theta), -np.sin(theta), y],
            [np.sin(theta), np.cos(theta), -x],
            [0, 0, 1]
        ])
    
    def Ad(self, v: 'se2'):
        v2 = self.Ad_matrix@v.vee
        return se2(x=v2[0], y=v2[1], theta=v2[2])

    @property
    def log(self):
        x, y, theta = self.x, self.y, self.theta
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/12, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V_inv = np.array([
            [a, b],
            [-b, a]
        ])/(a**2 + b**2)
        p = V_inv@np.array([x, y])
        return se2(x=p[0], y=p[1], theta=theta)     

def solve_control_gain(v, omega):
    A = -se2(v, 0, omega).ad_matrix
    B = np.array([[1, 0], [0, 0], [0, 1]])
    Q = 10*np.eye(3)  # penalize state
    R = 1*np.eye(2)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R)  # rescale K, set negative feedback sign
    K = -K
    return B, K #, A+B@K, B@K

def se2_diff_correction(e: se2): # U
    x = e.x
    y = e.y
    theta = e.theta
    with np.errstate(divide='ignore',invalid='ignore'):
        a = np.where(abs(theta) > 1e-1, -theta*np.sin(theta)/(2*(np.cos(theta) - 1)), 1 - theta**2/12 - theta**4/720)
        b = np.where(abs(theta) > 1e-1, -(theta*x*np.sin(theta) + (1 - np.cos(theta))*(theta*y - 2*x))/(2*theta*(1 - np.cos(theta))), -y/2 + theta*x/12 - theta**3*x/720)
        c = np.where(abs(theta) > 1e-1, -(theta*y*np.sin(theta) + (1 - np.cos(theta))*(-theta*x - 2*y))/(2*theta*(1 - np.cos(theta))), x/2 + theta*y/12 + theta**3*y/720)
    return -np.array([
        [a, theta/2, b],
        [-theta/2, a, c],
        [0, 0, 1]
    ])

def se2_diff_correction_inv(e: se2): # U inverse
    x = e.x
    y = e.y
    theta = e.theta
    with np.errstate(divide='ignore',invalid='ignore'):
        a = np.where(abs(theta) > 1e-4, np.sin(theta)/theta, 1 - theta**2/6 + theta**4/120)
        b = np.where(abs(theta) > 1e-4, (1  - np.cos(theta))/theta, theta/2 - theta**3/24)
        c = np.where(abs(theta) > 1e-4, -(x*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2) + y*(2*np.cos(theta) - np.cos(2*theta)/2 - 3/2))/(theta**2*(1 - np.cos(theta))), y/2 + theta*x/6 - theta**2*y/24 - theta**3*x/120 + theta**4*y/720)
        d = np.where(abs(theta) > 1e-4, -(x*(-2*np.cos(theta) + np.cos(2*theta)/2 + 3/2) + y*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2))/(theta**2*(1 - np.cos(theta))), -x/2 + theta*y/6 + theta**2*x/24 - theta**3*y/120 - theta**4*x/720)
    return -np.array([
        [a, -b, c],
        [b, a, d],
        [0, 0, 1]
    ])


def control_law(B, K, e:se2):
    L = np.diag([1, 1, 1])
    BK = L@se2_diff_correction_inv(e)@B@K@e.vee #L@se2_diff_correction_inv(e)@B@K@e.vee # controller input
    # print(se2_diff_correction_inv(e))
    return BK

def maxw(sol, x):
    U1 = np.eye(2) # multiply singular val of U
    U2 = np.array([
        [0],
        [0]])
    P = sol['P']
    P1 = P[:2, :]
    P2 = P[2, :]
    mu1 = sol['mu1'] 
    mu2 = sol['mu2']
    alpha = sol['alpha'].real
    
    w1 = (U1.T@P1@x + x.T@P1.T@U1)/(2*alpha*mu1) #disturbance for x y
    w2 = (U2.T@P1 + P2)@x/(alpha*mu2) #disturbance theta
    
    return w1, w2

def compute_control(t, x_vect, ref_data, freq_d, w1, w2, dist, sol, use_approx):
    # reference data
    ref_x = ref_data['x']
    ref_y = ref_data['y']
    ref_theta = ref_data['theta']
    ref_omega = ref_data['omega']
    ref_V = ref_data['V']

    # reference at time t
    r_x = float(ref_x(t))
    r_y = float(ref_y(t))
    r_omega = float(ref_omega(t))
    r_theta = float(ref_theta(t))
    r_V = float(ref_V(t))

    # Lie group
    X_r = SE2(r_x, r_y, r_theta) # reference
    X = SE2(x=x_vect[0], y=x_vect[1], theta=x_vect[2])
    e = se2(x=x_vect[3], y=x_vect[4], theta=x_vect[5]) # log error
    eta = X.inv@X_r # error in SE2
    chi = eta.log # error in se2

    v_r = se2(x=r_V, y=0, theta=r_omega) #np.array([r_V, 0, r_omega])
    
    # sine wave
    if dist == 'sine':
        w = se2(x=np.cos(2*np.pi*freq_d*t)*w1, y=np.sin(2*np.pi*freq_d*t)*w1, theta=np.cos(2*np.pi*freq_d*t)*w2).vee
    
    # square wave
    elif dist == 'square':
        w = se2(x=signal.square(2*np.pi*freq_d*t+np.pi/2)*w1, y=signal.square(2*np.pi*freq_d*t)*w1, theta=signal.square(2*np.pi*freq_d*t)*w2).vee
    
    # maximize dV
    elif dist == 'maxdV':
        er = chi.vee
        w1, w2 = maxw(sol, er)
        print('w1', w1, 'w2', w2)
        w = se2(w1[0], w1[1], w2[0]).vee
        print('w', w)

    B, K = solve_control_gain(1, 0)

    u = control_law(B, K, chi)
    us = se2.from_vector(u)
    v = v_r.vee + u + w
    vx = v[0]
    vy = v[1]
    omega = v[2]

    # log error dynamics
    U = se2_diff_correction(e)
    if use_approx:
        # these dynamics don't hold exactly unless you can move sideways
        e_dot = se2.from_vector((-v_r.ad_matrix + B@K)@e.vee + U@w.vee)
    else:
        # these dynamics, always hold
        e_dot = -v_r@e + se2.from_vector(U@(us.vee + w))

    return vx, vy, omega, e_dot