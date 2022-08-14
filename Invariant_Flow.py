import pyhull
import math
import numpy as np
from pytope import Polytope
import picos
import scipy.optimize
import control
import slycot

from IntervalHull import qhull2D, minBoundingRect
from SE2Lie import *
from rover_planning import *

# Singular value function for U matrix
def sv1(theta):
    return (np.sqrt(-2/(np.cos(theta)-1))*abs(theta))/2

def sv2(x, y, theta):
    v = x**2 + y**2
    value = np.sqrt(2*v*(-theta**2*(np.cos(theta)-1)-theta*(2*np.sin(theta)-np.sin(2*theta))-4*np.cos(theta)+np.cos(2*theta)+3)/
                        (theta**2*(np.cos(theta)-1)**2))/2
    if value > 0.84:
        value= 0.84
    return value

# LMI solver
def solve_lmi(alpha, A1, A2, A3, A4, verbosity=0):
    
    prob = picos.Problem()
    P = picos.SymmetricVariable('P', (3, 3))
    P1 = P[:2, :]
    P2 = P[2, :]
    mu1 = picos.RealVariable('mu_1')
    mu2 = picos.RealVariable('mu_2')
    gam = mu1 + mu2
    block_eq1 = picos.block([
         [A1.T*P + P*A1 + alpha*P, P1.T, P2.T],
         [P1, -alpha*mu1*np.eye(2), 0],
         [P2, 0, -alpha*mu2]])
    block_eq2 = picos.block([
         [A2.T*P + P*A2 + alpha*P, P1.T, P2.T],
         [P1, -alpha*mu1*np.eye(2), 0],
         [P2, 0, -alpha*mu2]])
    block_eq3 = picos.block([
         [A3.T*P + P*A3 + alpha*P, P1.T, P2.T],
         [P1, -alpha*mu1*np.eye(2), 0],
         [P2, 0, -alpha*mu2]])
    block_eq4 = picos.block([
         [A4.T*P + P*A4 + alpha*P, P1.T, P2.T],
         [P1, -alpha*mu1*np.eye(2), 0],
         [P2, 0, -alpha*mu2]])
    prob.add_constraint(block_eq1 << 0) # dV < 0
    prob.add_constraint(block_eq2 << 0)
    prob.add_constraint(block_eq3 << 0)
    prob.add_constraint(block_eq4 << 0)
    prob.add_constraint(P >> 1)
    prob.add_constraint(mu1 >> 0)
    prob.add_constraint(mu2 >> 0)
    prob.set_objective('min', mu1 + mu2)
    try:
        prob.solve(options={'verbosity': verbosity})
        cost = gam.value
    except Exception as e:
        print(e)
        cost = -1
    return {
        'cost': cost,
        'prob': prob,
        'mu1': mu1.value,
        'mu2': mu2.value,
        'P': np.array(P.value),
        'alpha':alpha,
        'gam': gam
    }


# Feedback Control 
def solve_control_gain_polytopic():
    A = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]])
    B = np.array([[1, 0], [0, 0], [0, 1]])
    Q = 10*np.eye(3)  # penalize state
    R = 1*np.eye(2)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R)
    K = -K  # rescale K, set negative feedback sign
    A0 = np.zeros((3,3)) + B@K
    return K, B, A0

# 
def find_se2_invariant_set(omega1, omega2, v1, v2, verbosity=0):
    dA1 = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]])
    dA2 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]])
    A0 = solve_control_gain_polytopic()[2]
    A1 = A0 + omega1*dA1 + v1*dA2
    A2 = A0 + omega1*dA1 + v2*dA2
    A3 = A0 + omega2*dA1 + v1*dA2
    A4 = A0 + omega2*dA1 + v2*dA2
    
    # these are the two parts of U(x), split ast U(x) = [U1, U2], where the first impacts the u, v and the last impacts the w disturbance
    # these are the zero order terms of the taylor expansion below
    # TODO: could add polytopic system with bounded input disturbance, U(x) is actually a function of the state not a constant, so this 
    # is an under approximation as is
    U1 = np.eye(2)
    U2 = np.array([
        [0],
        [0]])
    
    # we use fmin to solve a line search problem in alpha for minimum gamma
    if verbosity > 0:
        print('line search')
    
    # we perform a line search over alpha to find the largest convergence rate possible
    alpha_1 = np.real(np.max(np.array([np.linalg.eig(A1)[0],np.linalg.eig(A2)[0],np.linalg.eig(A3)[0],np.linalg.eig(A4)[0]])))
    print(alpha_1)
    # alpha_opt = scipy.optimize.fmin(lambda alpha: solve_lmi(alpha, A1, A2, A3, A4)['cost'], x0=1)
    alpha_opt = scipy.optimize.fminbound(lambda alpha: solve_lmi(np.real(alpha), A1, A2, A3, A4, verbosity=verbosity)['cost'], x1=0.001, x2=-alpha_1, disp=True if verbosity > 0 else False)
    
    sol = solve_lmi(alpha_opt, A1, A2, A3, A4)
    print(sol)
    prob = sol['prob']
    if prob.status == 'optimal':
        P = prob.variables['P'].value
        mu1 =  prob.variables['mu_1'].value
        mu2 =  prob.variables['mu_2'].value
        if verbosity > 0:
            print(sol)
    else:
        raise RuntimeError('Optimization failed')
        
    return sol

# Create Ellipsoid Invariant Set
def se2_lie_algebra_invariant_set_points(sol, t, w1_norm, w2_norm, e0): # w1_norm (x-y direc): wind speed
    P = sol['P']
    beta = (e0.T@P@e0) # V0
#     print(e0)
#     print('V0', beta)
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu1']*w1_norm**2 + sol['mu2']*w2_norm**2)*(1-np.exp(-sol['alpha']*t))) # V(t)
#     print('val', val)
    
    # 1 = xT(P/V(t))x, equation for the ellipse
    evals, evects = np.linalg.eig(P/val)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 25
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    return R@points

def iteration(w1, w2, e0, t, sol): # singlar value do the iteration
    x = 2
    y = 2
    theta = np.pi

    # since sv1 only depends on theta, we use sv1 to find the tolerance of theta first
    for j in np.linspace(0,100,101):
        v1 = sv1(theta)
        w1_a = v1*w1

        points = se2_lie_algebra_invariant_set_points(sol, t, w1_a, w2, e0) #Lie Algebra
        max_theta = points[2,:].max()
        min_theta = abs(points[2,:].min())
        theta_bound = max(max_theta, min_theta)
        v2 = sv1(theta_bound)

        if v1 > v2:
            if v1 - v2 > .1:
                theta = theta_bound
            else:
                break
        else:
            theta = theta_bound
    
    # use the previous result of theta to do the iteration of x and y      
    for k in np.linspace(0,100,101):
        v3 = sv2(x, y, theta)
        w1_b = w1_a + v2*w2

        points = se2_lie_algebra_invariant_set_points(sol, t, w1_b, w2, e0) #Lie Algebra
        max_x = points[0,:].max()
        min_x = abs(points[0,:].min())
        x_bound = np.round(max(max_x, min_x),6)
        max_y = points[1,:].max()
        min_y = abs(points[1,:].min())
        y_bound = np.round(max(max_y, min_y),6)
        v4 = sv2(x_bound, y_bound, theta)
        
        if v3 > v4:
            if v3 - v4 > 1e-3:
                x = x_bound
                y = y_bound
            else:
                break
        else:
            x = x_bound
            y = y_bound
            # the bound shouldn't over twice of the disturbance magnitude
            # if it over twice of the disturbance magnitude -> the guessing max singular value is too large
        #     if x_bound <=2*w1 and y_bound <=2*w1: 
        #         break
        #     # if over twice of the disturbance magnitude, decrease x/y for 0.1 until it reach 0.1
        #     elif x_bound > 2*w1 and x > 0.1:
        #         x = x-0.1
        #         if x < 0.1:
        #             x = 0.1
        #     elif y_bound > 2*w1 and y > 0.1:
        #         y = y-0.1
        #         if y < 0.1:
        #             y = 0.1
        # elif x == 0.1 or y == 0.1:
        #     break
        # else: # if the x/y is outside the x_bound/y_bound, replace the x/y with x_bound/y_bound
        #     if x > x_bound:
        #         x = x_bound
        #     if y > y_bound:
        #         y = y_bound
    i = np.array([x,y,theta])
    return i

def rotate_point(point, angle):
    new_point = np.array([point[0] * np.cos(angle) - point[1] * np.sin(angle),
                       point[0] * np.sin(angle) + point[1] * np.cos(angle)])
    return new_point 

def flowpipes(planner, n, e0, w1, w2):
    
    ref_data = planner.compute_ref_data()
    ref_omega = ref_data['omega']
    ref_theta = ref_data['theta']
    x_r = ref_data['way_points'][0,:]
    y_r = ref_data['way_points'][1,:]
    
    sol = find_se2_invariant_set(-np.pi/2, np.pi/2)
    zeta = iteration(w1, w2, e0, 0, sol)
    v1 = sv1(zeta[2])
    v2 = sv2(zeta[0], zeta[1],zeta[2])
    w1_new = v1*w1 + v2*w2

    # print(x_r.shape)
    nom = np.array([x_r,y_r]).T
    # print(e_theta.shape, nom.shape)
    flowpipes = []
    intervalhull = []
    t_vect = []
    Rx1 = []
    Rx2 = []
    Ry1 = []
    Ry2 = []
    
    steps = int(len(x_r)/n)
    if len(x_r)%n != 0:
        n = n+1
        
    for i in range(n):
        # print(i)
        a = steps*i
        if i < n-1:
            b = steps*(i+1)
            nom_i = nom[a:b,:] # steps*2
        else:
            b = len(x_r)
            nom_i = nom[a-1:b,:]
        # Get interval hull
        hull_points = qhull2D(nom_i)
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
            
        t = 0.05*a
        t_vect.append(t)
        ang_list = []
        for k in range(a, b):
            angle = ref_theta(0.05*k)
            ang_list.append(angle)
        
        # omega_list = []
        # for k in range(a, b):
        #     omega = ref_omega(0.05*k)
        #     omega_list.append(omega)
        # omega1 = np.min(omega_list)
        # omega2 = np.max(omega_list)
        # if np.min(omega_list) < 0:
        #      omega1 = -np.pi/2
        # else:
        #     omega1 = np.pi/2
        # sol = find_se2_invariant_set(omega1, 0)
        
        # zeta = iteration(w1, w2, e0, t, sol)
        # v1 = sv1(zeta[2])
        # v2 = sv2(zeta[0], zeta[1],zeta[2])
        # w1_new = v1*w1 + v2*w2
        
        # invariant set in se2
        points = se2_lie_algebra_invariant_set_points(sol, t, w1_new, w2, e0)
        
        # exp map (invariant set in Lie group)
        inv_points = np.zeros((3,points.shape[1]))
        for i in range(points.shape[1]):
            exp_points = se2(points[0,i], points[1,i], points[2,i]).exp
            inv_points[:,i] = np.array([exp_points.x, exp_points.y, exp_points.theta])
            
        inv_set = [[],[]]
        for theta in ang_list:
            inv_set1 = rotate_point(inv_points, theta)
            inv_set = np.append(inv_set, inv_set1, axis = 1) 
        set_bound = rotate_point(inv_set, -ref_theta(0.05*(a+b)/2))
        
        max_x = set_bound[0,:].max()
        min_x = abs(set_bound[0,:].min())
        x_bound1 = np.round(min(max_x, min_x),6)
        x_bound2 = np.round(max(max_x, min_x),6)
        Rx1.append(x_bound1)
        Rx2.append(x_bound2)
        max_y = set_bound[1,:].max()
        min_y = set_bound[1,:].min()
        y_bound1 = np.round(min(max_y, min_y),6)
        y_bound2 = np.round(max(max_y, min_y),6)
        Ry1.append(min_y)
        Ry2.append(max_y)

            
        P2 = Polytope(inv_set.T) 
        
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        
        P = P1 + P2 # sum
        p1_vertices = P1.V
        p_vertices = P.V
        p_vertices = np.append(p_vertices, p_vertices[0].reshape(1,2), axis = 0) # add the first point to last, or the flow pipes will miss one line
        flowpipes.append(p_vertices)
        intervalhull.append(P1.V)
    return n, flowpipes, intervalhull, nom, t_vect, Rx1, Rx2, Ry1, Ry2

def hj_invariant_set_points(r): # w1_norm (x-y direc): wind speed
    
    angle = np.linspace(0, 2*np.pi, 200)
    x = r * np.cos(angle) 
    y = r * np.sin(angle) 
    points = np.array([x,y])
    # R = r*np.eye(3,3)
    # # draw sphere
    # points = []
    # n = 25
    # for u in np.linspace(0, 2*np.pi, n):
    #     for v in np.linspace(0, 2*np.pi, 2*n):
    #         points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    # for v in np.linspace(0, 2*np.pi, 2*n):
    #     for u in np.linspace(0, 2*np.pi, n):
    #         points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    # points = np.array(points).T
    return points

def flowpipeshj(x_r, y_r, n, e0, w1, w2, r):
    
    nom = np.array([x_r,y_r]).T
    flowpipes = []
    intervalhull = []
    steps = int(len(x_r)/n)
    if len(x_r)%n != 0:
        n = n+1

    for i in range(n):
        if i < n-1:
            nom_i = nom[steps*i:steps*(i+1),:] # steps*2
        else:
            nom_i = nom[steps*i:len(x_r),:]
        # Get interval hull
        hull_points = qhull2D(nom_i)
        # Reverse order of points, to match output from other qhull implementations
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
        
        hj_points = hj_invariant_set_points(r)
        inv_points = hj_points[:2,:]
        
        P2 = Polytope(inv_points.T)
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        
        P = P1 + P2 # sum
        p1_vertices = P1.V
        p_vertices = P.V
        p_vertices = np.append(p_vertices, p_vertices[0].reshape(1,2), axis = 0) # add the first point to last, or the flow pipes will miss one line
        flowpipes.append(p_vertices)
        intervalhull.append(P1.V)
    return n, flowpipes, intervalhull, nom