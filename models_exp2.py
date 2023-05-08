import numpy as np
from scipy.signal import cont2discrete
import copy

"""""
Models:
1. Full_Model: Models a single controller for both hands i.e. alike bimanual control. State estimator has full state information.
2. LH_Model: Models the hand of the left side participant. Derived from the parent class Full_Model. State estimator has full state information of self hand and partial state information of partner's hand according to the experimental feedback condition.
3. RH_Model: Models the hand of the right side participant. Derived from the parent class Full_Model. State estimator has full state information of self hand and partial state  information of partner's hand according to the experimental feedback condition.

"""""


"""""
Change log:
1. Added cost on deviation of each hand from the y-location of the target. To prevent the hands from circling around the target which is not observed in the experiments.
2. Increased cost weighting on x control signal. To prevent spread of hand positions in the lateral dimension.

"""""

class Full_Model:
    def __init__(self, params, X_eq):
        self.params = params # system, control and noise parameters
        self.dt = params['h'] # step size
        self.l = params['l'] # free length of the spring
        self.k = params['k'] # spring constant for the haptic spring
        self.c = params['c'] # damping constant on hand velocity. Models viscous properties of the muscle
        self.mh = params['mh'] # mass of the hand
        self.tau = params['tau'] # time constant of the second order muscle filter
        
        self.update_linearmodel(X_eq)
        self.setup_delayaugmented_model()
        self.set_costmatrices()
        self.set_noisevariables()

    # linearizes the continuous nonlinear model around the equilibrium point (X_eq), 
    # discretizes the continuous linearized model, and 
    # returns the  discrete time linearized state space matrices    
    def update_linearmodel(self, X_eq):
        # Continuous time linearized model

        """
        States
        [0-xc-midpoint_x, 1-yc-midpoint_y, 
        2-xlc-lcursor_x, 3-vlcx-lcursor_xvel, 4-xl-lhand_x, 5-yl-lhand_y, 
        6-vlx-l_xvel, 7-vly-l_yvel, 8-fl_x-lhand_xforce, 9-gl_x-lhand_xactiv, 10-fl_y-lhand_yforce, 11-gl_y-lhand_yactiv,  
        12-xrc-rcursor_x, 13-vrcx-rcursor_xvel, 14-xr-rhand_x, 15-yr-rhand_y, 
        16-vrx-rhand_xvel, 17-vry-rhand_yvel, 18-fr_x-rhand_xforce, 19-gr_x-rhand_xactiv, 20-fr_y-rhand_yforce, 21-gr_y-rhand_yactiv, 
        22-targ_m_x, 23-targ_m_y]

        """

        xl, yl, xr, yr = X_eq[4], X_eq[5], X_eq[14], X_eq[15]
        
        # obtain jacobian matrix for midpoint states
        self.Ac = np.block([[np.zeros(3), 0.5, np.zeros(9), 0.5, np.zeros(10)],
                       [np.zeros(7), 0.5, np.zeros(9), 0.5, np.zeros(6)]])
                
        # obtain jacobian matrix for hand states
        self.Ah_r, self._Ah_l = self.hand_linearized_model(xr, yr, xl, yl)
        self.Ah_l, self._Ah_r = self.hand_linearized_model(xl, yl, xr, yr)

        # Construct full jacobian A matrix 
        # [midpoint states, l-hand states, r-hand states, target states]
        self.A = np.block([[self.Ac], # midpoint dynamics
                           [np.zeros((10, 2)), self.Ah_l, self._Ah_r, np.zeros((10, 2))], # right hand dynamics
                           [np.zeros((10, 2)), self._Ah_l, self.Ah_r, np.zeros((10, 2))], # left hand dynamics
                           [np.zeros((2, 24))]]) # target state dynamics
        
        # Jacobian Bh matrix same for both hands
        self.Bh = np.block([[np.zeros((7, 2))],
                            [np.array([[1/self.tau, 0], [0, 0], [0, 1/self.tau]])]]) 
        
        # Jacobian Bh matrix same for both hands
        self.B = np.block([[np.zeros((2, 4))],
                           [self.Bh, np.zeros((self.Bh.shape[0], 2))],
                           [np.zeros((self.Bh.shape[0], 2)), self.Bh],
                           [np.zeros((2, 4))]])
        
        self.C = np.eye(self.A.shape[0])

        self.state_len = self.A.shape[0]
        self.control_len = self.B.shape[1]
        self.obs_len = self.C.shape[0]
        
        self.linearized_discretizedbyzoh()
        
        return self.A_ld, self.B_ld, self.C

    # Sets up the block matrix elements for the hand states under consideration. 
    # States of the hand under consideration (x, y), states of the other hand (_x, _y). Required for the update_linearmodel method
    def hand_linearized_model(self, x, y, _x, _y):
        L = ((x - _x) ** 2 + (y - _y) ** 2) ** 0.5
        
        # A matrix portion for each hand dynamics
        # states: [xc, vcx, xh, yh, vhx, vhy, fh_x, gh_x, fh_y, gh_y] f's and g's are states for the second order muscle filter
        h2_0 = -self.k / self.mh * (1 - self.l/L  + self.l*(x - _x)**2/L**3)
        h2_1 = -self.k /self.mh *self.l*(x - _x)*(y - _y)/L**3
        h2_2 = -self.c / self.mh
        h2_4 = 1 / self.mh
        h3_0 = -self.k / self.mh *self.l*(y - _y)*(x - _x)/L**3
        h3_1 = -self.k / self.mh * (1 - self.l/L  + self.l*(y - _y)**2/L**3)
        h3_3 = -self.c / self.mh
        h3_6 = 1 / self.mh
        Ah = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, h2_0, h2_1, h2_2, 0, h2_4, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, h2_0, h2_1, h2_2, 0, h2_4, 0, 0, 0], [0, 0, h3_0, h3_1, 0, h3_3, 0, 0, h3_6, 0],
                        [0, 0, 0, 0, 0, 0, -1 / self.tau, 1 / self.tau, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1 / self.tau, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, - 1 / self.tau, 1 / self.tau], [0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / self.tau]])
        
        # Interaction forces between the hands
        _h2_0 = -self.k / self.mh * (self.l / L - 1 - self.l*(x - _x)**2/L**3)
        _h2_1 = self.k / self.mh * (self.l*(x - _x)*(y - _y)/L**3)
        _h3_0 = self.k / self.mh * (self.l*(x - _x)*(y - _y)/L**3)
        _h3_1 = -self.k / self.mh * (self.l / L - 1 - self.l*(y - _y)**2/L**3)
        _Ah = np.block([[np.zeros((1, 10))],
                        [0, 0, _h2_0, _h2_1, 0, 0, 0, 0, 0, 0],
                        [np.zeros((2, 10))],
                        [0, 0, _h2_0, _h2_1, 0, 0, 0, 0, 0, 0],
                        [0, 0, _h3_0, _h3_1, 0, 0, 0, 0, 0, 0],
                        [np.zeros((4, 10))]])

        return Ah, _Ah 
    
    # Discretizes a continuous linear model using zero order hold. Required for the update_linearmodel method
    def linearized_discretizedbyzoh(self):
        self.A_ld, self.B_ld, _, _, _ = cont2discrete((self.A, self.B, self.C, np.eye(self.A.shape[0])*0), self.dt, method='zoh')
        
        return self.A_ld, self.B_ld
    
    # augmenting the matrices to accommodate delayed states
    def setup_delayaugmented_model(self):
        self.h_delay = self.params['haptic delay']
        self.c_h = np.zeros((16, self.state_len)) 
        self.c_h[np.arange(16), np.concatenate((np.arange(4, 12), np.arange(14, 22)))] = 1 # states observed through the haptic feedback channel
        self.v_delay = self.params['visual delay'] 
        self.c_v = np.zeros((20, self.state_len)) 
        self.c_v[np.arange(20), np.arange(2, 22)] = 1 # states observed through the visual feedback channel
        self.A_ldd = np.block([[self.A_ld, np.zeros((self.state_len, self.state_len * self.v_delay))],
                             [np.eye(self.v_delay * self.state_len), np.zeros((self.state_len * self.v_delay, self.state_len))]])
        self.B_ldd = np.block([[self.B_ld], [np.zeros((self.state_len * self.v_delay, self.control_len))]])
        self.C_ldd = np.block([[np.zeros((self.c_h.shape[0], self.state_len * self.h_delay)), self.c_h, np.zeros((self.c_h.shape[0], self.state_len * (self.v_delay-self.h_delay)))],
                             [np.zeros((self.c_v.shape[0], self.state_len * self.v_delay)), self.c_v]])

    # Setting up cost matrices for the iLQR
    def set_costmatrices(self):
        self.r = self.params['r'] # weighting factor for the control cost 
        self.wp = self.params['wp'] # weighting factor for position cost
        self.wv = self.params['wv'] # weighting factor for velocity cost
        self.wf = self.params['wf'] # weighting factor for force cost
        self.R = self.r / (self.params['N'] - 1) * np.identity(self.control_len) # Control cost matrix
        
        self.p = np.block([[np.concatenate(([-self.wp], np.zeros(21), [self.wp, 0]))],
                            [np.concatenate(([0, -self.wp], np.zeros(21), [self.wp]))],
                           [np.zeros((20, 2)), np.diag([0, self.wv, 0, 0, 0, self.wv, self.wf, self.wf, self.wf, self.wf,
                           0, self.wv, 0, 0, 0, self.wv, self.wf, self.wf, self.wf, self.wf]), np.zeros((20, 2))]])
        self.Q_N = 1 / 2 * self.p.T @ self.p # State Cost matrix at last step
        self.Q = np.zeros_like(self.Q_N) 
    
    # Computing iLQR feedback gains for the nominal trajectory (u, x)
    def compute_controlfeedbackgains(self, x, u, x_target):
        self.obtain_linearmodels(x)
        
        # calculate control feedback gains
        self.K = np.zeros((self.params['N'] + 1, self.B.shape[1], self.B.shape[0])) 
        self.Kv = np.zeros((self.params['N'] + 1, self.B.shape[1], self.B.shape[0])) 
        self.Ku = np.zeros((self.params['N'] + 1, self.B.shape[1], self.R.shape[0]))
        self.v =  np.zeros((self.params['N'] + 1, self.Q.shape[1]))
        # Control feedback gain backward recursion
        S = self.Q_N
        self.v[-1, :] = self.Q_N @ x[:, -1]
        for step in np.arange(self.params['N'] - 1, -1, -1):
            self.K[step] = np.linalg.inv(self.B_t[:, :, step].T @ S @ self.B_t[:, :, step] + self.R) @ \
                            self.B_t[:, :, step].T @ S @ self.A_t[:, :, step]
            self.Kv[step] = np.linalg.inv(self.B_t[:, :, step].T @ S @ self.B_t[:, :, step] + self.R) @ \
                            self.B_t[:, :, step].T
            self.Ku[step] = np.linalg.inv(self.B_t[:, :, step].T @ S @ self.B_t[:, :, step] + self.R) @ self.R
            
            self.Q = self.Q_N.copy() if step >= self.params['settle_steps'] else np.zeros_like(self.Q_N)

            self.v[step] = (self.A_t[:, :, step] - self.B_t[:, :, step] @ self.K[step, :, :]).T @ self.v[step + 1] - \
                    self.K[step].T @ self.R @ u[:, step] + self.Q @ x[:, step]
            S = self.Q + self.A_t[:, :, step].T @ S @ (self.A_t[:, :, step] - self.B_t[:, :, step] @ self.K[step])

    # Setting up measurement and control noise covariance matrices 
    def set_noisevariables(self):
        self.sig_p_scale = self.params['sig prop noise']
        self.sig_h_scale = self.params['sig haptic noise']
        self.sig_v_scale = self.params['sig visual noise']
        
        s_p = self.params['sig p']
        s_v = self.params['sig v']
        s_f = self.params['sig f']
        
        # diagonal terms of noise std deviations for states observed through haptic
        self.sig_h = self.sig_p_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f, 
                                                s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f]) 
        # diagonal terms of noise std deviations for states observed through haptic
        self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f, 
                                                s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])

        self.noise_s = np.concatenate((self.sig_h, self.sig_v)) 

        # measurement noise covariance matrix
        self.Ws = np.diag(self.noise_s ** 2)

         # initial state covariance matrix 
        self.E = 0.01 * np.eye(self.A_ldd.shape[0])

        # control noise covariance matrix 
        v_c_cov = self.params['vel cov']
        self.noise_c = np.array([0.0, 0.0,
                                0.0, v_c_cov, 0.0, 0.0, v_c_cov, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, v_c_cov, 0.0, 0.0, v_c_cov, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0])
        self.Wc = np.diag(self.noise_c ** 2)
        self.Wc_d = np.diag(np.concatenate((self.noise_c ** 2, np.zeros(self.A_ldd.shape[0] - self.state_len))))

    # Estimating state using the Extended Kalman Filter
    def estimate_state(self, x_true_next, x_est_curr, u_curr):        
        y = self.C_ldd @ x_true_next + np.random.normal(0, np.sqrt(np.diag(self.Ws)))
        
        self.update_linearmodel(X_eq=x_est_curr)
        
        # prediction steps
        x_pred = x_est_curr[:self.A_ld.shape[0]] + self.rk4_ode(x_est_curr[:self.A_ld.shape[0]], u_curr, self.nonlinear_sysode)
        x_pred_aug = self.augment_newstate(x_pred, x_est_curr)
        E_pred = self.A_ldd @ self.E @ self.A_ldd.T + self.Wc_d
        
        # update steps
        Ko = E_pred @ self.C_ldd.T @ np.linalg.inv(self.C_ldd @ E_pred @ self.C_ldd.T + self.Ws)
        x_est_next = x_pred_aug + Ko @ (y - self.C_ldd @ x_pred_aug)
        self.E = (np.eye(self.C_ldd.shape[1]) - Ko @ self.C_ldd) @ E_pred
        
        return x_est_next 
    
    # Updating the delayed states vector (x_aug) with the new state of the system (x). Required for estimate_state method
    def augment_newstate(self, x, x_aug):
        x_aug_ = copy.copy(x_aug)
        x_aug_[self.A_ld.shape[0]:] = x_aug_[:-self.A_ld.shape[0]]
        
        x_aug_[:self.A_ld.shape[0]] = x
        
        return x_aug_
    
    # Obtains linear models for all time steps of the nominal trajectory (x). Required for compute_controlfeedbackgains method
    def obtain_linearmodels(self, x):
        # obtain linear models around nominal trajectory
        self.A_t = np.zeros((self.A_ld.shape[0], self.A_ld.shape[1], self.params['N']))
        self.B_t = np.zeros((self.B_ld.shape[0], self.B_ld.shape[1], self.params['N']))
        for step in range(self.params['N']):
            self.update_linearmodel(X_eq=x[:, step])
            self.A_t[:, :, step], self.B_t[:, :, step] = self.A_ld, self.B_ld
    
    # Setting up the nonlinear dynamical equations at current state X and control U
    def nonlinear_sysode(self, X, U):
        k = self.k
        l = self.l
        c = self.c
        mh = self.mh
        tau = self.tau
        
        """
        States
        [0-xc-midpoint_x, 1-yc-midpoint_y, 
        2-xlc-lcursor_x, 3-vlcx-lcursor_xvel, 4-xl-lhand_x, 5-yl-lhand_y, 
        6-vlx-l_xvel, 7-vly-l_yvel, 8-fl_x-lhand_xforce, 9-gl_x-lhand_xactiv, 10-fl_y-lhand_yforce, 11-gl_y-lhand_yactiv,  
        12-xrc-rcursor_x, 13-vrcx-rcursor_xvel, 14-xr-rhand_x, 15-yr-rhand_y, 
        16-vrx-rhand_xvel, 17-vry-rhand_yvel, 18-fr_x-rhand_xforce, 19-gr_x-rhand_xactiv, 20-fr_y-rhand_yforce, 21-gr_y-rhand_yactiv, 
        22-targ_m_x, 23-targ_m_y]

        """

        xc, yc, xlc, vlcx, xl, yl, vlx, vly, flx, glx, fly, gly, xrc, vrcx, xr, yr, vrx, vry, frx, grx, fry, gry, x_targ, y_targ = tuple(X)
        ulx, uly, urx, ury = tuple(U)
        
        L = ((xl - xr) ** 2 + (yl - yr) ** 2) ** 0.5
        
        xc_d, yc_d = (vlcx + vrcx) / 2, (vry + vly) / 2
        xlc_d = vlcx 
        vlxc_d = -k / mh * (1 - l / L) * (xl - xr) - c / mh * vlx + flx / mh
        xl_d, yl_d = vlx, vly
        vlx_d = -k / mh * (1 - l / L) * (xl - xr) - c / mh * vlx + flx / mh
        vly_d = -k / mh * (1 - l / L) * (yl - yr) - c / mh * vly + fly / mh
        flx_d = -flx / tau + glx / tau
        glx_d = -glx / tau + ulx / tau
        fly_d = -fly / tau + gly / tau
        gly_d = -gly / tau + uly / tau
        xrc_d = vrcx
        vrxc_d = -k / mh * (1 - l / L) * (xr - xl) - c / mh * vrx + frx / mh
        xr_d, yr_d = vrx, vry
        vrx_d = -k / mh * (1 - l / L) * (xr - xl) - c / mh * vrx + frx / mh
        vry_d = -k / mh * (1 - l / L) * (yr - yl) - c / mh * vry + fry / mh
        frx_d = -frx / tau + grx / tau
        grx_d = -grx / tau + urx / tau
        fry_d = -fry / tau + gry / tau
        gry_d = -gry / tau + ury / tau
        
        return np.array([xc_d, yc_d, 
                        xlc_d, vlxc_d, xl_d, yl_d, vlx_d, vly_d, flx_d, glx_d, fly_d, gly_d,
                        xrc_d, vrxc_d, xr_d, yr_d, vrx_d, vry_d, frx_d, grx_d, fry_d, gry_d, 0, 0])

    # Forward propagating the state of the nonlinear system (sys_ode) from the initial state (x0) using the current set of optimal controls (u)
    def generate_forwarddynamics(self, u, x0, sys_ode):
        X = np.zeros((x0.shape[0], u.shape[1] + 1))
        X[:, 0] = x0
        for step in range(u.shape[1]):
            X[:, step + 1] = X[:, step] + self.rk4_ode(x=X[:, step], u=u[:, step], sys_ode=sys_ode)

        return X
    
    # Numerical integration of the nonlinear system using 4th order Runge-Kutta method. Required for generate_forwarddynamics method
    def rk4_ode(self, x, u, sys_ode):
        k1 = sys_ode(x, u)
        k1 *= self.dt
        k2 = sys_ode(x + k1 / 2, u)
        k2 *= self.dt
        k3 = sys_ode(x + k2 / 2,  u)
        k3 *= self.dt
        k4 = sys_ode(x + k3, u)
        k4 *= self.dt

        x_integrated = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x_integrated

    # Computing trajectory cost with the current trajectory (u, x)
    def compute_trajectory_cost(self, x, u):
        cost = 0
        for step in range(u.shape[1]):
            Q = copy.copy(self.Q_N) if step > self.params['settle_steps'] else np.zeros_like(self.Q_N)
            
            cost += x[:, step].T @ Q @ x[:, step] + \
                u[:, step].transpose() @ self.R @ u[:, step]

        cost += x[:, -1].T @ self.Q_N @ x[:, -1] # terminal state cost

        return cost

class LH_Model(Full_Model):
    def __init__(self, fback_type, params, X_eq):
        self.fback_type = fback_type
        self.params = params
        self.dt = params['h'] # step size
        self.l = params['l'] # free length of the spring
        self.k = params['k'] # spring constant for the spring connecting object to the hand
        self.c = params['c'] # damping constant on hand velocity. Models viscous properties of the muscle
        self.mh = params['mh'] # mass of the hand
        self.tau = params['tau'] # time constants of the second order muscle filter
        
        self.update_linearmodel(X_eq)
        self.set_costmatrices()
        self.set_noisevariables()

    def update_linearmodel(self, X_eq):
        # Continuous time linearized model
        """
        States
        [0-xc-midpoint_x, 1-yc-midpoint_y, 
        2-xlc-lcursor_x, 3-vlcx-lcursor_xvel, 4-xl-lhand_x, 5-yl-lhand_y, 
        6-vlx-l_xvel, 7-vly-l_yvel, 8-fl_x-lhand_xforce, 9-gl_x-lhand_xactiv, 10-fl_y-lhand_yforce, 11-gl_y-lhand_yactiv,  
        12-xrc-rcursor_x, 13-vrcx-rcursor_xvel, 14-xr-rhand_x, 15-yr-rhand_y, 16-vrx-r_xvel,, 17-vry-rhand_yvel, 
        18-targ_m_x, 19-targ_m_y]

        """

        xl, yl, xr, yr = X_eq[4], X_eq[5], X_eq[14], X_eq[15]
        
        # obtain jacobian matrix for midpoint states
        self.Ac = np.block([[np.zeros(3), 0.5, np.zeros(9), 0.5, np.zeros(6)],
                       [np.zeros(7), 0.5, np.zeros(9), 0.5, np.zeros(2)]])
                
        # obtain jacobian matrix for hand states
        self.Ah_r, self._Ah_l = self.hand_linearized_model(xr, yr, xl, yl)
        self.Ah_l, self._Ah_r = self.hand_linearized_model(xl, yl, xr, yr)

        # Construct full jacobian A matrix 
        # midpoint states, l-hand states, r-hand states, target states
        self.A = np.block([[self.Ac], # mass dynamics
                           [np.zeros((10, 2)), self.Ah_l, self._Ah_r[:, :6], np.zeros((10, 2))], # right hand
                           [np.zeros((6, 2)), self._Ah_l[:6, :], self.Ah_r[:6, :6], np.zeros((6, 2))], # left hand
                           [np.zeros((2, 20))]]) # target state multiplier
        
        # Jacobian Bh matrix same for both hands
        self.Bh = np.block([[np.zeros((7, 2))],
                            [np.array([[1/self.tau, 0], [0, 0], [0, 1/self.tau]])]]) 
        
        # Jacobian Bh matrix same for both hands
        self.B = np.block([[np.zeros((2, 2))],
                           [self.Bh],
                           [np.zeros((6, 2))],
                           [np.zeros((2, 2))]])
        
        self.C = np.eye(self.A.shape[0])

        self.state_len = self.A.shape[0]
        self.control_len = self.B.shape[1]
        self.obs_len = self.C.shape[0]
        
        self.linearized_discretizedbyzoh()
        
        self.setup_delayaugmented_model()
        
        return self.A_ld, self.B_ld, self.C
    
    def setup_delayaugmented_model(self):
        self.h_delay = self.params['haptic delay']
        self.v_delay = self.params['visual delay']
        if self.fback_type == 'V':
            self.c_h = np.zeros((8, self.state_len))
            self.c_h[np.arange(8), np.arange(4, 12)] = 1
            self.c_v = np.zeros((16, self.state_len))
            self.c_v[np.arange(16), np.arange(2, 18)] = 1
        elif self.fback_type == 'H':
            self.c_h = np.zeros((14, self.state_len))
            self.c_h[np.arange(14), np.arange(4, 18)] = 1
            self.c_h[[8, 9, 10, 11, 12, 13], [2, 3, 4, 5, 6, 7]] = -1
            self.c_v = np.zeros((10, self.state_len))
            self.c_v[np.arange(10), np.arange(2, 12)] = 1
        elif self.fback_type == 'VH':
            self.c_h = np.zeros((14, self.state_len))
            self.c_h[np.arange(14), np.arange(4, 18)] = 1
            self.c_h[[8, 9, 10, 11, 12, 13], [2, 3, 4, 5, 6, 7]] = -1
            self.c_v = np.zeros((16, self.state_len))
            self.c_v[np.arange(16), np.arange(2, 18)] = 1

        self.A_ldd = np.block([[self.A_ld, np.zeros((self.state_len, self.state_len * self.v_delay))],
                             [np.eye(self.v_delay * self.state_len), np.zeros((self.state_len * self.v_delay, self.state_len))]])
        self.B_ldd = np.block([[self.B_ld], [np.zeros((self.state_len * self.v_delay, self.control_len))]])
        self.C_ldd = np.block([[np.zeros((self.c_h.shape[0], self.state_len * self.h_delay)), self.c_h, np.zeros((self.c_h.shape[0], self.state_len * (self.v_delay-self.h_delay)))],
                             [np.zeros((self.c_v.shape[0], self.state_len * self.v_delay)), self.c_v]])
    
    def set_costmatrices(self):
        self.r = self.params['r'] # weighting factor for the control cost matrix
        self.wp = self.params['wp'] # weighting factor for position cost
        self.wv = self.params['wv'] # weighting factor for velocity cost
        self.wf = self.params['wf'] # weighting factor for force cost
        self.R = self.r / (self.params['N'] - 1) * np.array([[2, 0], [0, 1]]) # Control cost matrix
        
        self.p = np.block([[np.concatenate(([-self.wp], np.zeros(17), [self.wp, 0]))],
                            [np.concatenate(([0, -self.wp], np.zeros(17), [self.wp]))],
                            [np.zeros((16, 2)), np.diag([0, self.wv, 0, -self.wp/2, self.wv, self.wv, self.wf, self.wf, self.wf, self.wf,
                            0, 0, 0, 0, 0, 0]), np.zeros((16, 2))]])
        self.p[5, -1] = self.wp / 2 # setting a cost on vertical hand deviation from the target 
        self.Q_N = 1 / 2 * self.p.transpose() @ self.p # State Cost matrix at last step
        self.Q_N = np.block([[self.Q_N, np.zeros((self.state_len, self.A_ld.shape[0] - self.state_len))],
                             [np.zeros((self.A_ld.shape[0] - self.state_len, self.A_ld.shape[0]))]])
        self.Q = np.zeros_like(self.Q_N)

    def set_noisevariables(self):
  
        self.sig_p_scale = self.params['sig prop noise']
        self.sig_h_scale = self.params['sig haptic noise']
        self.sig_v_scale = self.params['sig visual noise']
        
        s_p = self.params['sig p']
        s_v = self.params['sig v']
        s_f = self.params['sig f']
        
        if self.fback_type == 'V':
            self.sig_h = self.sig_p_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])
            self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f,
                                                    s_p, s_v, s_p, s_p, s_v, s_v,])
        elif self.fback_type == 'H':
            self.sig_h = np.concatenate((self.sig_p_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f]), 
                                self.sig_h_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v])))
            self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])
        elif self.fback_type == 'VH':
            self.sig_h = np.concatenate((self.sig_p_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f]), 
                                        self.sig_h_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v])))
            self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f,
                                                        s_p, s_v, s_p, s_p, s_v, s_v])

        self.noise_s = np.concatenate((self.sig_h, self.sig_v)) # sensory feedback additive noise

        self.Ws = np.diag(self.noise_s ** 2) # sensory feedback additive noise covariance matrix
        
        self.E = 0.01 * np.eye(self.A_ldd.shape[0]) # state covariance matrix initial for estimation

        v_c_cov = self.params['vel cov']
        self.noise_c = np.array([0.0, 0.0,
                                0.0, v_c_cov, 0.0, 0.0, v_c_cov, 0, 0.0, 0.0, 0.0, 0.0,
                                0.0, v_c_cov, 0.0, 0.0, v_c_cov, v_c_cov,
                                0.0, 0.0])
        self.Wc = np.diag(self.noise_c ** 2)    
        self.Wc_d = np.diag(np.concatenate((self.noise_c ** 2, np.zeros(self.A_ldd.shape[0] - self.state_len))))

    def nonlinear_sysode(self, X, U):
        k = self.k
        l = self.l
        c= self.c
        mh = self.mh
        tau = self.tau
        
        """
        States
        [0-xc-midpoint_x, 1-yc-midpoint_y, 
        2-xlc-lcursor_x, 3-vlcx-lcursor_xvel, 4-xl-lhand_x, 5-yl-lhand_y, 
        6-vlx-l_xvel, 7-vly-l_yvel, 8-fl_x-lhand_xforce, 9-gl_x-lhand_xactiv, 10-fl_y-lhand_yforce, 11-gl_y-lhand_yactiv,  
        12-xrc-rcursor_x, 13-vrcx-rcursor_xvel, 14-xr-rhand_x, 15-yr-rhand_y, 16-vrx-r_xvel,, 17-vry-rhand_yvel, 
        18-targ_m_x, 19-targ_m_y]

        """
        
        xc, yc, xlc, vlcx, xl, yl, vlx, vly, flx, glx, fly, gly, xrc, vrcx, xr, yr, vrx, vry, x_targ, y_targ = tuple(X)
        ulx, uly = tuple(U)
        
        L = ((xl - xr) ** 2 + (yl - yr) ** 2) ** 0.5
        
        xc_d, yc_d = (vrcx + vlcx) / 2, (vry + vly) / 2
        xlc_d = vlcx 
        vlcx_d = -k / mh * (1 - l / L) * (xl - xr) - c / mh * vlx + flx / mh
        xl_d, yl_d = vlx, vly
        vlx_d = -k / mh * (1 - l / L) * (xl - xr) - c / mh * vlx + flx / mh
        vly_d = -k / mh * (1 - l / L) * (yl - yr) - c / mh * vly + fly / mh
        flx_d = -flx / tau + glx / tau
        glx_d = -glx / tau + ulx / tau
        fly_d = -fly / tau + gly / tau
        gly_d = -gly / tau + uly / tau
        xrc_d = vrcx
        vrcx_d = -k / mh * (1 - l / L) * (xr - xl) - c / mh * vrx
        xr_d, yr_d = vrx, vry
        vrx_d = -k / mh * (1 - l / L) * (xr - xl) - c / mh * vrx
        vry_d = -k / mh * (1 - l / L) * (yr - yl) - c / mh * vry
        
        return np.array([xc_d, yc_d, 
                        xlc_d, vlcx_d, xl_d, yl_d, vlx_d, vly_d, flx_d, glx_d, fly_d, gly_d,
                        xrc_d, vrcx_d, xr_d, yr_d, vrx_d, vry_d, 0, 0])

class RH_Model(Full_Model):
    def __init__(self, fback_type, params, X_eq):
        self.fback_type = fback_type
        self.params = params
        self.dt = params['h']
        self.l = params['l']
        self.k = params['k'] # spring constant for the spring connecting object to the hand
        self.c = params['c'] # damping constant on the velocity of the hand
        self.mh = params['mh'] # mass of the object, mass of the hand
        self.tau = params['tau'] # time constants of the second order muscle filter
        
        self.update_linearmodel(X_eq)
        self.set_costmatrices()
        self.set_noisevariables()

    def update_linearmodel(self, X_eq):
        """
        States
        [0-xc-midpoint_x, 1-yc-midpoint_y, 
        2-xlc-lcursor_x, 3-vlcx-lcursor_xvel, 4-xl-lhand_x, 5-yl-lhand_y, 6-vlx-l_xvel, 7-vly-l_yvel,   
        8-xrc-rcursor_x, 9-vrcx-rcursor_xvel, 10-xr-rhand_x, 11-yr-rhand_y, 
        12-vrx-r_xvel,, 13-vry-rhand_yvel, 14-fr_x, 15-gr_x, 16-fr_y, 17-gr_y, 
        18-targ_m_x, 19-targ_m_y]

        """
        xl, yl, xr, yr = X_eq[4], X_eq[5], X_eq[10], X_eq[11]
        
        # obtain jacobian matrix for mass states
        self.Ac = np.block([[np.zeros(3), 0.5, np.zeros(5), 0.5, np.zeros(10)],
                       [np.zeros(7), 0.5, np.zeros(5), 0.5, np.zeros(6)]])
                
        # obtain jacobian matrix for hand states
        self.Ah_r, self._Ah_l = self.hand_linearized_model(xr, yr, xl, yl)
        self.Ah_l, self._Ah_r = self.hand_linearized_model(xl, yl, xr, yr)

        # Construct full jacobian A matrix 
        # mass states, l-hand states, r-hand states, target states
        self.A = np.block([[self.Ac], # mass dynamics
                           [np.zeros((6, 2)), self.Ah_l[:6, :6], self._Ah_r[:6, :], np.zeros((6, 2))], # right hand
                           [np.zeros((10, 2)), self._Ah_l[:, :6], self.Ah_r, np.zeros((10, 2))], # left hand
                           [np.zeros((2, 20))]]) # target state multiplier
        
        # Jacobian Bh matrix same for both hands
        self.Bh = np.block([[np.zeros((7, 2))],
                            [np.array([[1/self.tau, 0], [0, 0], [0, 1/self.tau]])]]) 
        
        # Jacobian Bh matrix same for both hands
        self.B = np.block([[np.zeros((2, 2))],
                           [np.zeros((6, 2))],
                           [self.Bh],
                           [np.zeros((2, 2))]])
        
        self.C = np.eye(self.A.shape[0])

        self.state_len = self.A.shape[0]
        self.control_len = self.B.shape[1]
        self.obs_len = self.C.shape[0]
        
        self.linearized_discretizedbyzoh()
        
        self.setup_delayaugmented_model()
        
        return self.A_ld, self.B_ld, self.C

    def setup_delayaugmented_model(self):
        self.h_delay = self.params['haptic delay']
        self.v_delay = self.params['visual delay']
        if self.fback_type == 'V':
            self.c_h = np.zeros((8, self.state_len))
            self.c_h[np.arange(8), np.arange(10, 18)] = 1
            self.c_v = np.zeros((16, self.state_len))
            self.c_v[np.arange(16), np.arange(2, 18)] = 1
        elif self.fback_type == 'H':
            self.c_h = np.zeros((14, self.state_len))
            self.c_h[np.arange(14), np.concatenate((np.arange(2, 8), np.arange(10, 18)))] = 1
            self.c_h[[0, 1, 2, 3, 4, 5], [8, 9, 10, 11, 12, 13]] = -1
            self.c_v = np.zeros((10, self.state_len))
            self.c_v[np.arange(10), np.arange(8, 18)] = 1
        elif self.fback_type == 'VH':
            self.c_h = np.zeros((14, self.state_len))
            self.c_h[np.arange(14), np.concatenate((np.arange(2, 8), np.arange(10, 18)))] = 1
            self.c_h[[0, 1, 2, 3, 4, 5], [8, 9, 10, 11, 12, 13]] = -1
            self.c_v = np.zeros((16, self.state_len))
            self.c_v[np.arange(16), np.arange(2, 18)] = 1

        self.A_ldd = np.block([[self.A_ld, np.zeros((self.state_len, self.state_len * self.v_delay))],
                             [np.eye(self.v_delay * self.state_len), np.zeros((self.state_len * self.v_delay, self.state_len))]])
        self.B_ldd = np.block([[self.B_ld], [np.zeros((self.state_len * self.v_delay, self.control_len))]])
        self.C_ldd = np.block([[np.zeros((self.c_h.shape[0], self.state_len * self.h_delay)), self.c_h, np.zeros((self.c_h.shape[0], self.state_len * (self.v_delay-self.h_delay)))],
                             [np.zeros((self.c_v.shape[0], self.state_len * self.v_delay)), self.c_v]])

    def set_costmatrices(self):
        self.r = self.params['r'] # weighting factor for the control cost matrix
        self.wp = self.params['wp'] # weighting factor for position cost
        self.wv = self.params['wv'] # weighting factor for velocity cost
        self.wf = self.params['wf'] # weighting factor for force cost
        self.R = self.r / (self.params['N'] - 1) * np.array([[2, 0], [0, 1]]) # Control cost matrix
        
        self.p = np.block([[np.concatenate(([-self.wp], np.zeros(17), [self.wp, 0]))],
                            [np.concatenate(([0, -self.wp], np.zeros(17), [self.wp]))],
                           [np.zeros((16, 2)), np.diag([0, 0, 0, 0, 0, 0, 
                            0, self.wv, 0, -self.wp/2, self.wv, self.wv, self.wf, self.wf, self.wf, self.wf]), np.zeros((16,2))]])
        self.p[11, -1] = self.wp/2 # setting a cost on vertical hand deviation from the target 

        self.Q_N = 1 / 2 * self.p.transpose() @ self.p # State Cost matrix at last step
        self.Q_N = np.block([[self.Q_N, np.zeros((self.state_len, self.A_ld.shape[0] - self.state_len))],
                             [np.zeros((self.A_ld.shape[0] - self.state_len, self.A_ld.shape[0]))]])
        self.Q = np.zeros_like(self.Q_N)

    def set_noisevariables(self):
        # self.sig_v = self.params['sig visual noise']
        # self.noise_s = np.tile(self.sig_v, self.C_ldd.shape[0]) # sensory feedback additive noise
        
        self.sig_p_scale = self.params['sig prop noise']
        self.sig_h_scale = self.params['sig haptic noise']
        self.sig_v_scale = self.params['sig visual noise']
        
        s_p = self.params['sig p']
        s_v = self.params['sig v']
        s_f = self.params['sig f']
        
        if self.fback_type == 'V':
            self.sig_h = self.sig_h_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])
            self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, 
                                                        s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])
        elif self.fback_type == 'H':
            self.sig_h = np.concatenate((self.sig_h_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v]),
                                        self.sig_p_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])))
            self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])
        elif self.fback_type == 'VH':
            self.sig_h = np.concatenate((self.sig_h_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v]),
                                            self.sig_p_scale * np.array([s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])))
            self.sig_v = self.sig_v_scale * np.array([s_p, s_v, s_p, s_p, s_v, s_v, 
                                                        s_p, s_v, s_p, s_p, s_v, s_v, s_f, s_f, s_f, s_f])
 
        self.noise_s = np.concatenate((self.sig_h, self.sig_v)) # sensory feedback additive noise

        self.Ws = np.diag(self.noise_s ** 2) # sensory feedback additive noise covariance matrix
        
        self.E = 0.01 * np.eye(self.A_ldd.shape[0]) # state covariance matrix initial for estimation

        v_c_cov = self.params['vel cov']
        self.noise_c = np.array([0.0, 0.0,
                                0.0, v_c_cov, 0.0, 0.0, v_c_cov, v_c_cov,
                                0.0, v_c_cov, 0.0, 0.0, v_c_cov, 0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0])
        self.Wc = np.diag(self.noise_c ** 2)    
        self.Wc_d = np.diag(np.concatenate((self.noise_c ** 2, np.zeros(self.A_ldd.shape[0] - self.state_len))))

    def nonlinear_sysode(self, X, U):
        k = self.k
        l = self.l
        c = self.c
        mh = self.mh
        tau = self.tau
        
        """
        States
        [0-xc-midpoint_x, 1-yc-midpoint_y, 
        2-xlc-lcursor_x, 3-vlcx-lcursor_xvel, 4-xl-lhand_x, 5-yl-lhand_y, 6-vlx-l_xvel, 7-vly-l_yvel,   
        8-xrc-rcursor_x, 9-vrcx-rcursor_xvel, 10-xr-rhand_x, 11-yr-rhand_y, 
        12-vrx-r_xvel,, 13-vry-rhand_yvel, 14-fr_x, 15-gr_x, 16-fr_y, 17-gr_y, 
        18-targ_m_x, 19-targ_m_y]

        """

        xc, yc, xlc, vlcx, xl, yl, vlx, vly, xrc, vrcx, xr, yr, vrx, vry, frx, grx, fry, gry, x_targ, y_targ = tuple(X)
        urx, ury = tuple(U)
        
        L = ((xl - xr) ** 2 + (yl - yr) ** 2) ** 0.5
        
        xc_d, yc_d = (vrcx + vlcx) / 2, (vry + vly) / 2
        xlc_d = vlcx 
        vlcx_d = -k / mh * (1 - l / L) * (xl - xr) - c / mh * vlx
        xl_d, yl_d = vlx, vly
        vlx_d = -k / mh * (1 - l / L) * (xl - xr) - c / mh * vlx
        vly_d = -k / mh * (1 - l / L) * (yl - yr) - c / mh * vly
        xrc_d = vrcx
        vrcx_d = -k / mh * (1 - l / L) * (xr - xl) - c / mh * vrx + frx / mh
        xr_d, yr_d = vrx, vry
        vrx_d = -k / mh * (1 - l / L) * (xr - xl) - c / mh * vrx + frx / mh
        vry_d = -k / mh * (1 - l / L) * (yr - yl) - c / mh * vry + fry / mh
        frx_d = -frx / tau + grx / tau
        grx_d = -grx / tau + urx / tau
        fry_d = -fry / tau + gry / tau
        gry_d = -gry / tau + ury / tau
        
        return np.array([xc_d, yc_d, 
                        xlc_d, vlcx_d, xl_d, yl_d, vlx_d, vly_d,
                        xrc_d, vrcx_d, xr_d, yr_d, vrx_d, vry_d, frx_d, grx_d, fry_d, gry_d, 0, 0])
