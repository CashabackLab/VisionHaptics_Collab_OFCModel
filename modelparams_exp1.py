"""
Model parameters for Experiment 1 models. 
"""
import numpy as np
import os

data_save_directory = os.getcwd() + r'\\Data' 

h = 0.01 # time step in s
N = int(3.5 / h) # total number of simulation time steps

target_coordinates = (0.15, 0) # coordinates of the target

"""
mh: mass of the hand (kg),
k: spring constant of haptic spring connecting the participant hands (N/m)
c: damping constant modeling the viscous properties of muscles and limbs (Ns/ m)
tau: time constant of the second order filter modeling activation to force (s)
l: resting length of the spring (m)
r: control cost weighting
wp: position cost weighting
wv: velocity cost weighting
wf: force cost weighting
sig_u: standard deviation of process or control noise
sig_p: standard deviation of position measurement noise
sig_v: standard deviation of velocity measurement noise
sig_f: standard deviation of force measurement noise
sig prop noise: scaling of measurement noise standard deviation for states derived from proprioception
sig haptic noise: scaling of measurement noise standard deviation for states derived from partner haptic
sig vision noise: scaling of measurement noise standard deviation for states derived from partner vision
haptic delay: time delay of haptic partner feedback or self propcioceptive feedback
visual delay: time delay of visual partner feedback 
vel cov: scaling for velocity covariance in process noise covariance matrix
pos cov: scaling for position covariance in process noise covariance matrix
settle_steps: step count to settle at target
"""

params = {'h': h, 'N': N,
        'mh': 1.0, 'k': 1, 'c': 0.15, 'tau': 0.04, 'l': 0.1,
        'r': 0.1, 'wp': 1, 'wv': 0.2, 'wf': 0.01,
        'sig_u': 0.005, 'sig p': 0.01, 'sig v': 0.1, 'sig f': 1,
        'sig prop noise': 0.4, 'sig haptic noise': 0.8, 'haptic delay': 5, 'sig visual noise':0.2, 'visual delay': 10,
        'vel cov': 0.01, 'pos cov': 0.01,
        'settle_steps': 150}

process_noise = params['sig_u'] * np.array([0.0, 0.0,
                                      0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                                      0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  
                                      0.0, 0.0]) # process additive noise standard deviations

hand_x_init = [-params['l']/2, params['l']/2] # hand initial x coordinates

perturb_location = 0.85 * target_coordinates[0]
perturb_mag = 0.1
perturb_std = 25
perturb_status = 1
def force_perturbation(step):
    force = perturb_status * perturb_mag * np.exp(-(step * 1000 * h - 3 * perturb_std) ** 2 / perturb_std ** 2)
    
    return force



