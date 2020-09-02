import numpy as np
import sys
sys.path.append('/home/prakash/gps/python/gps/Linear EM with GPS')
from plot_error_ellipse import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import pickle
import random
file = '/home/prakash/Desktop/New Data Points/Real system data/ILQG_0.65_MSN/'
#fig_1 = plt.figure()
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
fig_cost, axs_cost = plt.subplots(1, 2, figsize=(9, 3))
""" ax = fig_1.add_subplot(113)
ax_u_1 = fig_1.add_subplot(213)
ax_u_2 = fig_1.add_subplot(313) """
em_iteration = [0,1]
length = len(em_iteration)
T=30
dX=6
dU =2 
real_state = np.zeros(( len(em_iteration), T,dX ))
real_actions = np.zeros(( len(em_iteration), T,dU ))
color=['red','blue','green','magenta','black']
for index,em_iteration in enumerate(em_iteration):
    state_file = open(file+ 'real_system_state%s.pkl'%em_iteration,'rb')
    real_state[index,:,:] = pickle.load(state_file)
    state_file.close()
    control_file = open(file+ 'real_system_control%s.pkl'%em_iteration,'rb')
    real_actions[index,:,:] = pickle.load(control_file)
    control_file.close()

    axs[0].plot(real_state[index, :,0],real_state[index,:,1],c=color[index] )
    axs[1].plot(real_actions[index,:,0], c=color[index])
    axs[2].plot(real_actions[index,:,1],c=color[index])
#print real_actions

""" plt.show()
plt.close() """


#####
####
#

beta=0.25  
Q_dim=6
gamma=1
q_dim=2
Q_control=np.identity(2)
small_fac=1e-5

target_state= np.array([5, 20, 0,0,0,0])
Q_new=np.ones((6,1))
q_new=np.ones((2,1))
Q =  np.identity(Q_dim)
q = small_fac* np.identity(q_dim)
Quad_reward_orig=np.zeros(( length, T))

""" R_t_orig=np.zeros((T,1))
R_t_orig_quad=np.zeros((T,1)) """
for em_iter in range( length ):
    for t in range (T):
        ####
        #### Quadratic reward
        ####
        Quad_reward_orig[em_iter, t]=  ( np.dot((real_state[em_iter,t,:6]-target_state),np.dot(Q,np.transpose((real_state[em_iter,t,:6]-target_state)))) \
            + np.dot(real_actions[em_iter,t,:2],np.dot(Q_control, np.transpose(  real_actions[em_iter,t,:2]  )) ) )
    axs_cost[0].plot( np.cumsum (Quad_reward_orig[ em_iter,: ]), c =color[em_iter]  )
    axs_cost[1].plot((Quad_reward_orig[ em_iter,: ]), c =color[em_iter] ,marker = '*' )
#axs_cost[1].set_yscale('log')
plt.show()
plt.close()