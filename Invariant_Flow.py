import numpy as np
from pytope import Polytope # not sure if it works for 3D
import picos
import scipy.optimize
import control
import itertools

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
    return value
    
def solve_lmi(alpha, A, verbosity=0):
    
    prob = picos.Problem()
    P = picos.SymmetricVariable('P', (3, 3))
    P1 = P[:2, :]
    P2 = P[2, :]
    mu1 = picos.RealVariable('mu_1')
    mu2 = picos.RealVariable('mu_2')
    gam = mu1 + mu2
    for Ai in A:

        block_eq1 = picos.block([
            [Ai.T*P + P*Ai + alpha*P, P1.T, P2.T],
            [P1, -alpha*mu1*np.eye(2), 0],
            [P2, 0, -alpha*mu2]])
    
        prob.add_constraint(block_eq1 << 0) # dV < 0
    # prob.add_constraint(block_eq2 << 0)
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


def solve_control_gain(vr):
    A = np.array([
        [0, 0, 0],
        [0, 0, vr],
        [0, 0, 0]])
    B = np.array([[1, 0], [0, 0], [0, 1]])
    Q = 10*np.eye(3)  # penalize state
    R = 1*np.eye(2)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R)
    K = -K  # rescale K, set negative feedback sign
    A0 = B@K
    return K, B, A0


def find_se2_invariant_set(omega1, omega2, v1, v2, e, case, verbosity=0):
    dA = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]])    
    A0 = solve_control_gain(v2)[2]

    iterables2 =[[omega1, omega2]]
    omega = []
    for m in itertools.product(*iterables2):
        m = np.array(m)
        omega.append(m)

    if case=='no_side':
        a1 = []
        a2 = []
        a3 = []
        a4 = []
        a5 = []
        a6 = []
        for i in range(e.shape[1]):
            err = se2(e[0,i], e[1,i], e[2,i])
            U = se2_diff_correction(err)
            U_inv = se2_diff_correction_inv(err)
            L = np.diag([1, 0, 1])
            dA = (U@L@U_inv - np.eye(3))@A0
            # print(dA)
            a1.append(dA[0,0])
            a2.append(dA[0,1])
            a3.append(dA[0,2])
            a4.append(dA[1,0])
            a5.append(dA[1,1])
            a6.append(dA[1,2])
        iterables = [[np.min(a1), np.max(a1)],[np.min(a2), np.max(a2)],[np.min(a3), np.max(a3)],[np.min(a4), np.max(a4)],[np.min(a5), np.max(a5)],[np.min(a6)+v1, np.max(a6)+v2]]
        # iterables = [[0],[0],[0],[-0.01, 0.01],[-0.01, 0.01],[v1, v2]]
    
    else:
        iterables = [[0],[0],[0],[0],[0],[v1, v2]]
    v = []
    for n in itertools.product(*iterables):
        n = np.array(n)
        v.append(n)
    # print(iterables)
    A = []
    eig = []
    for omegai in omega:
        for vi in v:
            Ai = A0 + omegai*dA + np.array([[vi[0], vi[1], vi[2]],[vi[3], vi[4], vi[5]],[0, 0,0]])
            A.append(Ai)
            eig.append(np.linalg.eig(Ai)[0])
    
    # we use fmin to solve a line search problem in alpha for minimum gamma
    if verbosity > 0:
        print('line search')
    
    # we perform a line search over alpha to find the largest convergence rate possible
    # alpha_1 = np.linalg.norm(np.max(eig))
    alpha_1 = np.real(np.max(eig))
    #print(alpha_1)
    alpha_opt = scipy.optimize.fminbound(lambda alpha: solve_lmi(alpha, A, verbosity=verbosity)['cost'], x1=0.001, x2=-alpha_1, disp=True if verbosity > 0 else False)
    #print(alpha_opt1)
    #alpha_opt2 = scipy.optimize.fminbound(lambda alpha: solve_lmi(alpha, A1, A2, U1, U2, verbosity=verbosity)['cost'], x1=0.001, x2=-alpha_2, disp=True if verbosity > 0 else False)
    #print(alpha_opt2)
    #alpha_opt = np.min(np.array([alpha_opt1, alpha_opt2]))
    
    # if the alpha optimization fail, pick a fixed value for alpha.
    sol = solve_lmi(alpha_opt, A)
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
# invariany set for 6D
def se2_lie_algebra_invariant_set_points(sol, t, w1_norm, w2_norm, beta): # w1_norm (x-y direc): wind speed
    P = sol['P']
    # V = xTPx scalar
    # beta = (e0.T@P@e0) # V0
    #print('V0', beta)
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu1']*w1_norm**2 + sol['mu2']*w2_norm**2)*(1-np.exp(-sol['alpha']*t))) # V(t)
    #print('val', val)
    
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
    return R@points, val

def iteration(w1, w2, u12, u3, beta, t, sol): # singlar value do the iteration
    x = 2
    y = 2
    theta = np.pi

    # since sv1 only depends on theta, we use sv1 to find the tolerance of theta first
    for j in np.linspace(0,100,101):
        v1 = sv1(theta)
        w1_a = v1*w1

        points, _ = se2_lie_algebra_invariant_set_points(sol, t, w1_a+u12, w2+u3, beta) #Lie Algebra
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

        points, _ = se2_lie_algebra_invariant_set_points(sol, t, w1_b+u12, w2+u3, beta) #Lie Algebra
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

    return sv1(theta)*w1 + sv2(x, y, theta)*w2


def rotate_point(point, angle):
    new_point = np.array([point[0] * np.cos(angle) - point[1] * np.sin(angle),
                       point[0] * np.sin(angle) + point[1] * np.cos(angle)])
    return new_point 

def flowpipes(planner, n, beta, w1, w2, sol):
    
    ref_data = planner.compute_ref_data()
    
    ref_omega = ref_data['omega']
    ref_theta = ref_data['theta']
    ref_V = ref_data['V']
    x_r = ref_data['way_points'][0,:]
    y_r = ref_data['way_points'][1,:]
    
    w1_new = iteration(w1, w2, 0, 0, beta, 0, sol)

    # print(x_r.shape)
    nom = np.array([x_r,y_r]).T
    # print(e_theta.shape, nom.shape)
    flowpipes = []
    intervalhull = []
    t_vect = []
    Ry1 = []
    Ry2 = []
    # lyap = []
    
    steps0 = int(len(x_r)/n)
    
    a = 0    
    for i in range(n):
        if i < len(x_r)%n:
            steps = steps0 + 1
        else:
            steps = steps0
        b = a + steps
        if i == n-1:
            nom_i = nom[a:len(x_r)+1,:]
            # dnom = nom[len(x_r)-1,:] - nom[len(x_r)-2,:]
            # nom_last = nom[len(x_r)-1,:] + dnom.reshape(1,2)
            # nom_i = np.append(nom_i, nom_last, axis=0)
        else:
            nom_i = nom[a:b+1,:]
        # Get interval hull
        hull_points = qhull2D(nom_i)
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
            
        t = 0.05*a
        t_vect.append(t)
        ang_list = []
        for k in range(a, b):
            if k == 0:
                k = 1e-3
            angle = ref_theta(0.05*k)
            ang_list.append(angle)
        
        
        # omega_list = []
        # V_list = []
        # for k in range(a, b):
        #     omega = ref_omega(0.05*k)
        #     V = ref_V(0.05*k)[0]
        #     if abs(V) < 0.1:
        #         if V < 0:
        #             V = -0.1
        #         else:
        #             V = 0.1
        #     omega_list.append(omega)
        #     V_list.append(V)
        # omega1 = np.min(omega_list)
        # omega2 = np.max(omega_list)
        # # v1 = np.min(V_list)
        # v2 = np.max(V_list)
        # if np.min(omega_list) < 0:
        #      omega1 = -np.pi/2
        # else:
        #     omega1 = np.pi/2
        # sol = find_se2_invariant_set(omega1, omega2, 18, 20) 
        
        # zeta = iteration(w1, w2, beta, 0, sol)
        # v1 = sv1(zeta[2])
        # v2 = sv2(zeta[0], zeta[1],zeta[2])
        # w1_new = v1*w1 + v2*w2
        
        # invariant set in se2
        points, val1 = se2_lie_algebra_invariant_set_points(sol, t, w1_new, w2, beta) # invariant set at t0 in that time interval
        points2, val2 = se2_lie_algebra_invariant_set_points(sol, 0.05*b, w1_new, w2, beta) # invariant set at t final in that time interval
        # val1, val2 are value of lyapunov function
        # beta = val2
        # print(beta)
        # val = val1
        
        if val2 > val1: 
            points = points2
        # lyap.append(val)
        
        # exp map (invariant set in Lie group) x, y, theta
        inv_points = np.zeros((3,points.shape[1]))
        for j in range(points.shape[1]):
            # print(points[2,j])
            # val = points2[:,j].T@sol['P']@points2[:,j]
            # lyap.append(val)
            exp_points = se2(points[0,j], points[1,j], points[2,j]).exp
            inv_points[:,j] = np.array([exp_points.x, exp_points.y, exp_points.theta])

        
        # m = np.array(lyap).argmax()
        # e0 = points2[:,m]
        
        # rotaion and sweep
        inv_set = [[],[]]
        for theta in ang_list:
            inv_set1 = rotate_point(inv_points, theta) # it only gives you x and y
            inv_set = np.append(inv_set, inv_set1, axis = 1) 
        
        set_bound = rotate_point(inv_set, -ref_theta(0.05*(a+b)/2))
        
        max_x = set_bound[0,:].max()
        min_x = set_bound[0,:].min()
        x_bound = np.sqrt(min_x**2 + max_x**2)
        max_y = set_bound[1,:].max()
        min_y = set_bound[1,:].min()
        y_bound = np.sqrt(min_y**2 + max_y**2)
        Ry1.append(max_y)
        Ry2.append(min_y)
            
        P2 = Polytope(inv_set.T) 
        
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        
        P = P1 + P2 # sum

        p1_vertices = P1.V
        p_vertices = P.V

        p_vertices = np.append(p_vertices, p_vertices[0].reshape(1,2), axis = 0) # add the first point to last, or the flow pipes will miss one line
        
        # create list for flow pipes and interval hull
        flowpipes.append(p_vertices)
        intervalhull.append(P1.V)
        
        a = b
    return flowpipes, intervalhull, nom, t_vect, Ry1, Ry2

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
    steps0 = int(len(x_r)/n)
    
    a = 0    
    for i in range(n):
        if i < len(x_r)%n:
            steps = steps0 + 1
        else:
            steps = steps0
        b = a + steps
        if i == n-1:
            nom_i = nom[a:len(x_r),:]
        else:
            nom_i = nom[a:b+1,:]
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
        a = b
    return n, flowpipes, intervalhull, nom