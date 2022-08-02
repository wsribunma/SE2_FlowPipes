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

def svd(theta):
    return (np.sqrt(-2/(np.cos(theta)-1))*abs(theta))/2

def solve_lmi(alpha, A1, A2, U1, U2, verbosity=0):
    
    prob = picos.Problem()
    P = picos.SymmetricVariable('P', (3, 3))
    P1 = P[:2, :]
    P2 = P[2, :]
    mu1 = picos.RealVariable('mu_1')
    mu2 = picos.RealVariable('mu_2')
    gam = mu1 + mu2
    block_eq1 = picos.block([
         [A1.T*P + P*A1 + alpha*P, P1.T*U1, P1.T*U2 + P2.T],
         [U1.T*P1, -alpha*mu1*np.eye(2), 0],
         [U2.T*P1 + P2, 0, -alpha*mu2]])
    block_eq2 = picos.block([
         [A2.T*P + P*A2 + alpha*P, P1.T*U1, P1.T*U2 + P2.T],
         [U1.T*P1, -alpha*mu1*np.eye(2), 0],
         [U2.T*P1 + P2, 0, -alpha*mu2]])
    prob.add_constraint(block_eq1 << 0) # dV < 0
    prob.add_constraint(block_eq2 << 0)
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
        'P': np.round(np.array(P.value), 3),
        'alpha':alpha,
        'gam': gam
    }


def solve_control_gain():
    A = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]])
    B = np.array([[1, 0], [0, 0], [0, 1]])
    Q = 10*np.eye(3)  # penalize state
    R = 1*np.eye(2)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R)
    K = -K  # rescale K, set negative feedback sign
    A0 = A + B@K
    return K, B, A0


def find_se2_invariant_set(verbosity=0):
    dA = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]])    
    A0 = solve_control_gain()[2]
    A1 = A0 + (0)*dA
    A2 = A0 + (75*np.pi/180)*dA

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
    alpha_1 = np.max(np.array([np.linalg.eig(A1)[0],np.linalg.eig(A2)[0]]))
    alpha_opt = scipy.optimize.fminbound(lambda alpha: solve_lmi(alpha, A1, A2, U1, U2, verbosity=verbosity)['cost'], x1=0.001, x2=-alpha_1, disp=True if verbosity > 0 else False)
    
    sol = solve_lmi(alpha_opt, A1, A2, U1, U2)
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

def rotate_point(point, angle):
    new_point = np.array([point[0] * np.cos(angle) - point[1] * np.sin(angle),
                       point[0] * np.sin(angle) + point[1] * np.cos(angle)])
    return new_point

def flowpipes2(x_r, y_r, n, e0, w1, w2, sol):
    
    nom = np.array([x_r,y_r]).T
    flowpipes = []
    intervalhull = []
    cp = []
    ag = []
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
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)

        ang_list = []
        for j in range(len(nom_i)-1):
            x = nom_i[j+1][0]-nom_i[j][0]
            y = nom_i[j+1][1]-nom_i[j][1]
            l = np.sqrt(x**2+y**2)
            angle = np.arccos(x/l)
            ang_list.append(angle)
        min_ang = min(np.abs(ang_list))
        max_ang = max(np.abs(ang_list))
        
        x = nom_i[-1][0]-nom_i[0][0]
        y = nom_i[-1][1]-nom_i[0][1]
        l = np.sqrt(x**2+y**2)
        angle = np.arccos(x/l)
        if np.around(y, decimals = 1) < 0:
            angle = 2*np.pi - angle

        t = 0.05*i*steps
        # invariant set in se2
        if max_ang == 0:
            max_ang = np.pi/20
        points = se2_lie_algebra_invariant_set_points(sol, t, w1*svd(max_ang), w2, e0)
        
        # exp map (invariant set in Lie group)
        inv_points = np.zeros((3,points.shape[1]))
        for i in range(points.shape[1]):
            exp_points = se2(points[0,i], points[1,i], points[2,i]).exp
            inv_points[:,i] = np.array([exp_points.x, exp_points.y, exp_points.theta])
            
        inv_set = rotate_point(inv_points, ang_list[0])
        for theta in ang_list:
            inv_set1 = rotate_point(inv_points, theta)
            inv_set = np.append(inv_set,inv_set1,axis = 1)    
        P2 = Polytope(inv_set.T) 
        
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        
        P = P1 + P2 # sum
        p1_vertices = P1.V
        p_vertices = P.V
        p_vertices = np.append(p_vertices, p_vertices[0].reshape(1,2), axis = 0) # add the first point to last, or the flow pipes will miss one line
        flowpipes.append(p_vertices)
        intervalhull.append(P1.V)
        cp.append(center_point)
        ag.append(angle)
    return n, flowpipes, intervalhull, nom

def hj_invariant_set_points(r): # w1_norm (x-y direc): wind speed
    
    R = r*np.eye(3,3)
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
    #u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:40j]
    #x = np.cos(u) * np.sin(v)
    #y = np.sin(u) * np.sin(v)
    #z = np.cos(v)
    #points = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    return R@points

def flowpipeshj(x_r, y_r, n, e0, w1, w2, r):
    
    nom = array([x_r,y_r]).T
    flowpipes = []
    intervalhull = []
    cp = []
    ag = []
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
        cp.append(center_point)
        ag.append(angle)
    return n, flowpipes, intervalhull, nom