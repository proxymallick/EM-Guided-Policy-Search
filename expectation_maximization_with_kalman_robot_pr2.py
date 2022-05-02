from __future__ import division
import sys
sys.path.append('/home/prakash/gps/python/gps/Linear EM with GPS/EM Control PR2 Robot')
from scipy import optimize
import csv
import numpy as np
import pandas as pd
from scipy import random, linalg
from scipy.stats import multivariate_normal
from numpy.linalg import multi_dot
import math
import random
import time,timeit
Nfeval = 0
counter=29
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from temporary_robust_kalman import robust_kf_ks_time_vary_last_night_edited, robust_kf_ks_time_vary
from results_time_varying_pr2 import print_results_time_vary
#from temporary_robust_kalman import robust_kf_ks_time_vary
from temporary_robust_kalman_time_varying_pr2 import *
from numba import jit, cuda ,njit
from timeit import default_timer as timer  
file_name='/home/prakash/gps/python/gps/dataset and pkl file/dataset and pkl file/After matrix Augmentation/'
import seaborn as sns
#random.seed(100)

#############################################################
### Nearest correlation amtrix approximation by Rebonato and Jackel (1999)
#############################################################

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

#############################################################
### Nearest correlation amtrix approximation by Higham (2000)
#############################################################

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def is_pd(K):
    try:
        np.linalg.cholesky(K)
        print "Matrix is positive definite ------ Can be used"
        return 1 
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0

def robust_cholesky(A):
    A = np.triu(A); n = np.shape(A)[0]; tol = n*np.spacing(1)
    if A[0,0]<=tol :
        A[0,0:n]=0
    else:
        A[0,0:n] = A[0,0:n]/ np.sqrt(A[0,0])
    for k in range(1,n):
        #A[k,k:n]=A[k,k:n] -  np.dot(np.transpose(A[k-1,k]) ,  A[k-1,k:n])
        A[k,k:n]=A[k,k:n] -  np.dot(np.transpose(A[ :k,k]) ,  A[:k,k:n])
        if A[k,k]<=tol:
            A[k,k:n]=0
        else:
            A[k,k:n]=A[k,k:n]/np.sqrt(A[k,k])
    return A

def EM_kalman_time_varying_case(argument, T,data_K,data_k,data_sigma,data_X,r,Adyn,Bdyn,Sigmadyn,A_rew,B_rew,Sig_rew,t_s,\
    initial_state_mu,multiply_factor_for_exponent,mean_reward ,file_name ,dX,dU):
    #############################################################
    ### Define the Dynamics parameters initialized after learning the transition dynamics
    #############################################################

    T_temp=T
    data_sig=np.zeros((T,dU,dU))
    for t in range(T):
        data_sig[t,:,:]= np.dot( data_sigma[t,:,:], np.transpose(data_sigma[t,:,:]) )



    if np.isnan(Sigmadyn).any():
        print "nans appearing in the Sigmadyn"

    if np.isnan(Sig_rew).any():
        print "nans appearing in the Sig_rew"

    print data_K.shape,data_k.shape,data_sig.shape

    reward_dim=1
    action_dim=dU
    state_dim=dX
    gamma=1
    A_x_dyn=Adyn[:,:,:state_dim]

    B_u_dyn=Adyn[:,:,state_dim:state_dim+action_dim] # np.random.randn(state_dim,2)
    covdyn= Sigmadyn#+ np.transpose(Sigmadyn) )
    """ for i in range(T):
        A_x_dyn[i,:,:]= 0.5 * ( A_x_dyn + np.transpose(A_x_dyn) )
        covdyn[i,]= 0.5 * (Sigmadyn+ np.transpose(Sigmadyn) )  """
    #####
    ## Check if the matrix is posive definite or not
    #####
    #print Sig_rew

    
    print "shape of the A_xdyn, B_u_dyn, cov_dyn reward is ",A_x_dyn.shape,B_u_dyn.shape,covdyn.shape
    
    covrew=(Sig_rew)

    mean_noise=np.zeros((state_dim,))
    mean_noise_reward=np.zeros((reward_dim,))

    w_t=np.zeros((T,state_dim))
    v_t=np.zeros((T,1))
    #############################################################
    ### Define the Reward parameters initialized after learning the reward dynamics
    #############################################################

    R_x_dyn=A_rew[:,:,:state_dim]
    R_u_dyn=A_rew[:,:,state_dim:state_dim+action_dim] # np.random.randn(state_dim,2)  
    print "shape of the R_xdyn, R_u_dyn, cov_rew reward is ",R_x_dyn.shape,R_u_dyn.shape,covrew.shape
    
    #############################################################
    ### Define the policy parameters compatible with the GPS
    #############################################################
    
    k1= data_K[0,:,:]
    k2= data_k[0,:].reshape(dU,)
    policy_sigma= data_sig[0,:,:].reshape(dU,dU)
    #print policy_sigma
    #dd
    for i in range (T):
        np.all(np.linalg.eigvals(data_sig[i,:,:]) > 0)
    print "k1 shape and k2 shape is ",k1.shape,k2.shape

    #############################################################
    ### Kalman filter and smoother definations
    #############################################################
    s_hat= np.zeros((T,state_dim,1)) #
    A_kal=np.zeros((T,state_dim,state_dim))
    B_kal=np.zeros((T,state_dim,action_dim))
    covdyn_kal=np.zeros((T,state_dim,state_dim))
    w_t_kal=np.zeros((T,state_dim))

    #s_hat = np.absolute(np.random.randn(state_dim,1))

    #############################################################
    ## x(t+1) = A(t)x(t) + B(t)u(t) + w(t),   [w(t)]    (    [ Q(t)    S(t) ] )   >
    ##                                        [    ] ~ N( 0, [              ] )   >  0 
    ##   y(t) = C(t)x(t) + D(t)u(t) + v(t),   [v(t)]    (    [ S^T(t)  R(t) ] )   >
    ############################################################# 
    

    ###############
    ##            Varies from 0 - 28
    ##########

    for t in range (T-1):
        w_t[t,:] = np.random.multivariate_normal(mean_noise,covdyn[t,:,:])
        v_t[t,:] = np.random.normal(mean_noise_reward,covrew[t,:,:])
        A_kal[t,:,:]=A_x_dyn[t,:,:]#-np.dot(s_hat[t,:,:],(np.linalg.solve(covrew[t,:,:],R_x_dyn[t,:,:])))
        B_kal[t,:,:]=B_u_dyn[t,:,:]#-np.dot(s_hat[t,:,:],(np.dot(np.linalg.inv(covrew[t,:,:]),R_u_dyn[t,:,:])))
        covdyn_kal[t,:,:]=covdyn[t,:,:]#-np.dot(s_hat[t,:,:],np.dot(np.linalg.inv(covrew[t,:,:]),np.transpose(s_hat[t,:,:])))
        np.linalg.cholesky(covrew[t,:,:])
        np.linalg.cholesky(covdyn[t,:,:])
        np.linalg.cholesky(covdyn_kal[t,:,:])
        w_t_kal[t,:] = np.random.multivariate_normal(mean_noise,covdyn_kal[t,:,:])
    print "shape of w_t is", w_t.shape
    print "shape of v_t is", v_t.shape
    print "shape of A_kal is", A_kal.shape
    print "shape of B_kal is", B_kal.shape    
    

    p_1_n=1.e-6*np.identity(state_dim)

    ####  DEFINE X U AND R AND ASSIGN SPACE
    x_sim=np.zeros((T,state_dim))
    u_sim=np.zeros((T,action_dim))
    reward=np.zeros((T,))
    x_sim[0,:]=  initial_state_mu#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
    u_sim[0,:]=np.random.multivariate_normal((np.dot(k1,x_sim[0,:])+k2).reshape(dU,),policy_sigma,1)
    reward[0] =  (np.dot(R_x_dyn[0,:,:],x_sim[0,:].reshape(state_dim,1))+ np.dot(R_u_dyn[0,:,:],u_sim[0,:].reshape(action_dim,1)) +v_t[0,:] ).reshape(1,)
    #############################################################
    ### Simulate the state space the trick is that use the rewards but d
    ### dont use the true states or the true actions which is dependent 
    ### on the true states-- 
    ### Note: - Also the reward calculated after simulating the state space
    ### are the approxmation of the reward distribution
    #############################################################
    for t in range(1,T):
        
        x_sim[t,:]= np.dot(A_kal[t-1,:,:],x_sim[t-1,:]) + np.dot(B_kal[t-1,:,:],u_sim[t-1,:])  +w_t_kal[t-1,:]
        u_sim[t,:]=np.random.multivariate_normal((np.dot(data_K[t,:,:],x_sim[t,:])+data_k[t,:].reshape(action_dim,)),data_sig[t,:,:],1)    
        reward[t] =  (np.dot(R_x_dyn[t-1,:,:],x_sim[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn[t,:,:],u_sim[t,:].reshape(action_dim,1)) +v_t[t,:] ).reshape(1,)

    #print reward
    
    print x_sim[:,:2]

    #plot_ellipse(x_sim , covdyn_kal[:,0:2,0:2] )
    
    
    """ x_est_smooth,cov_smooth,M = Kalman_filter_smoother(T,state_dim,action_dim,A_kal\
                                        ,B_kal,data_K,data_k,data_sig,data_X,s_hat,covrew\
                                                ,reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn) """
    global x_est_smooth,cov_smooth,M,x_est,p
    x_est_smooth,cov_smooth,M,x_est,p= robust_kf_ks_time_vary_change(T,state_dim,action_dim,A_kal\
                                        ,B_kal,data_K,data_k,data_sig,data_X,s_hat,covrew\
                                                ,reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn,u_sim,x_sim,dU,dX)

    file_x_smooth = open(file_name+'file_x_smooth%s.pkl'%argument[1],'wb')
    pickle.dump( x_est_smooth , file_x_smooth)
    file_x_smooth.close()

    file_cov_smooth = open(file_name+'file_cov_smooth%s.pkl'%argument[1],'wb')
    pickle.dump( cov_smooth, file_cov_smooth )
    file_cov_smooth.close()
    print x_est_smooth[:,:2]
    #plot_ellipse(x_est_smooth , cov_smooth[:,0:2,0:2] )
    """ def write_csv(data):
        with open('/home/prakash/gps/python/gps/csv_em_updated_params.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data) """
    def f(k):  
        x_1_n=x_est_smooth[0,:].reshape(dX,1)
        
        mu_1=initial_state_mu.reshape(dX,1)


        x_sim_inside=np.zeros((T,state_dim))
        u_sim_inside=np.zeros((T,action_dim))
        reward_inside=np.zeros((T,))

        #############################################################
        ### Expectation expression for the log likelihood
        #############################################################
        temp = np.dot(x_1_n, np.transpose(x_1_n) ) + p_1_n - np.dot(x_1_n,np.transpose(mu_1)) -  np.dot( mu_1,np.transpose(x_1_n) ) \
            - np.dot( mu_1,np.transpose(mu_1) )
        Expec_log_joint_1st_term=-.5 *( np.trace(np.dot( np.linalg.inv( p_1_n ), temp )))  -.5 * np.linalg.det(p_1_n)
        con_term = 0
        
        #print "shape of the A_total is",A_total.shape
        Expec_log_joint_sum=0 
        for t in range (T-1):
            """ if t==0:
                x_sim_inside[0,:]=  initial_state_mu#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
            else:
                x_sim_inside[t,:]= np.dot(A_kal[t,:,:],x_sim[t-1,:]) + np.dot(B_kal[t,:,:],u_sim[t-1,:]) + np.dot(s_hat[t,:,:], np.dot(np.linalg.inv(covrew[t,:]),reward[t-1])).reshape(1,state_dim) +w_t_kal[t,:]
            u_sim_inside[t,:]=np.random.multivariate_normal((np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))\
                                                                                                              ,np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) ),1)    
            reward[t] =  (np.dot(R_x_dyn[t,:,:],x_sim[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn[t,:,:],u_sim[t,:].reshape(2,1)) +v_t[t,:] ).reshape(1,)
            """
            
            sigma_total=  np.vstack((np.hstack (( Sigmadyn[t,:,:] , s_hat[t,:,:] )) ,np.hstack( ( np.transpose(s_hat[t,:,:])  ,covrew[t,:]))))  
            #print "shape of the sigma total is ",sigma_total.shape

            A_total= np.vstack((np.hstack((A_x_dyn[t,:,:],B_u_dyn[t,:,:])) , np.hstack((R_x_dyn[t,:,:],R_u_dyn[t,:,:])) ))
            
            x_t_smooth = x_est_smooth[t,:].reshape(dX,1)
            x_t_plus_1_smooth = x_est_smooth[t+1,:].reshape(dX,1)
            x_t_x_t=np.dot(x_t_smooth,np.transpose(x_t_smooth))+cov_smooth[t,:,:]
            x_t_plus_1_r_t_transpose= np.dot(np.dot(x_t_plus_1_smooth, np.transpose( x_t_smooth ) ) + M[t+1,:,:] , np.transpose(R_x_dyn[t,:,:])  + \
                np.dot( np.transpose(k[:(dU*dX)].reshape(dU,dX)) , np.transpose(R_u_dyn[t,:,:]) ) \
            ) + np.dot( x_t_plus_1_smooth ,   np.dot( np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,1)) , np.transpose(R_u_dyn[t,:,:]) )  )
            u_t_x_t_transpose = np.transpose  (np.dot(x_t_x_t , np.transpose(k[:(dU*dX)].reshape(dU,dX)) )   +\
                           np.dot( x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape (dU,1) ) ) )
            r_t_r_t_1st_term =  multi_dot([ R_x_dyn[t,:,:], x_t_x_t , np.transpose (R_x_dyn[t,:,:]) ]) 
            r_t_r_t_2nd_term =  multi_dot([R_x_dyn[t,:,:] ,np.transpose(u_t_x_t_transpose) , np.transpose (R_u_dyn[t,:,:]) ]  )   
            r_t_r_t_3rd_term= np.transpose(r_t_r_t_2nd_term)
            r_t_r_t_4th_term = covrew[t,:]
            u_t_u_t_transpose= ( multi_dot( [k[:(dU*dX)].reshape(dU,dX) ,  np.dot( x_t_smooth, np.transpose(x_t_smooth) ) +cov_smooth[t,:,:] , np.transpose(k[:(dU*dX)].reshape(dU,dX))] ) \
            +multi_dot([k[:(dU*dX)].reshape(dU,dX),x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))]) \
                +  np.transpose (multi_dot([k[:(dU*dX)].reshape(dU,dX),x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))]) )  \
            + np.dot(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1) ,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))  ) \
                + np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)) )   )

            r_t_r_t_5th_term = multi_dot ([ R_u_dyn[t,:,:] , u_t_u_t_transpose , np.transpose(R_u_dyn[t,:,:]) ])

            r_t_r_t=r_t_r_t_1st_term+r_t_r_t_2nd_term+r_t_r_t_3rd_term+ r_t_r_t_4th_term +r_t_r_t_5th_term

            zeta_zeta=  np.vstack( ( np.hstack(  (    np.dot(x_t_plus_1_smooth, np.transpose(x_t_plus_1_smooth)) + cov_smooth[t+1,:,:]  \
            , x_t_plus_1_r_t_transpose   ) )  ,   np.hstack(  (   np.transpose(x_t_plus_1_r_t_transpose)  \
            ,  r_t_r_t  ) )   )  ) 

            #print  "shape of zeta zeta is",zeta_zeta.shape
            
            zeta_z_1 = np.hstack(( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:]  \
            , np.dot( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:], np.transpose(k[:(dU*dX)].reshape(dU,dX))  ) \
            + multi_dot([x_t_plus_1_smooth,np.transpose (k[(dU*dX): (dU*dX)+dU ].reshape(dU,1))])) )

            

            r_t_x_t_transpose=   np.dot(R_x_dyn[t,:,:],x_t_x_t ) + np.dot(R_u_dyn[t,:,:] , \
            u_t_x_t_transpose )  

            r_t_u_t_transpose = multi_dot([ R_x_dyn[t,:,:], x_t_x_t , np.transpose (k[:(dU*dX)].reshape(dU,dX)) ])  + \
                multi_dot([ R_u_dyn[t,:,:] ,  u_t_x_t_transpose ,  np.transpose(k[:(dU*dX)].reshape(dU,dX)) ]) \
                    + multi_dot([R_x_dyn[t,:,:] ,x_t_smooth, np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1)) ]) \
                        + multi_dot([ R_u_dyn[t,:,:], np.dot(k[:(dU*dX)].reshape(dU,dX),x_t_smooth)+k[(dU*dX): (dU*dX)+dU ].reshape(dU,1) \
                            , np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,1))])

            zeta_z_2= np.hstack((  r_t_x_t_transpose,r_t_u_t_transpose ))
            zeta_z = np.vstack (( zeta_z_1,zeta_z_2 ))
            
            #print "shape of zeta z is",zeta_z.shape
            z_z_1=  np.hstack  ((x_t_x_t, np.transpose(u_t_x_t_transpose)))
            z_z_2=  np.hstack(( u_t_x_t_transpose, u_t_u_t_transpose  )) 

            #print "z_z_1 shape and z_z2_shapes are",z_z_1.shape,z_z_2.shape

            z_z = np.vstack((z_z_1,z_z_2))
            #print "z_z shape is ",z_z.shape
            if np.linalg.eigvals(u_t_u_t_transpose).all() >0 : 

            #print u_t_u_t_transpose
                
                Expec_log_joint_2nd_term= -.5 * ( np.trace( np.dot( np.linalg.inv(sigma_total) , (zeta_zeta - np.dot( zeta_z, np.transpose(A_total)) - np.transpose(np.dot( zeta_z, np.transpose(A_total)) )  \

                + multi_dot ([A_total , z_z , np.transpose(A_total)])  )  )) )  -.5 * np.linalg.det(sigma_total)
                Expec_log_joint_sum  = Expec_log_joint_sum +Expec_log_joint_2nd_term#+(reward[t])
                """ con_term = con_term-    np.log(  np.linalg.det(  np.dot( np.eye(2)* 1.e-16 + data_sigma[t,:].reshape(2,2)\
                        , np.transpose(data_sigma[t,:].reshape(2,2)) )) ) +         \
                        np.log(  np.linalg.det(  np.dot( np.eye(2)* 1.e-6 + k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) )) )\
                            - epsilon """
                #control_constraint_1  = (np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))  - (np.zeros(2,) + 20)
                #control_constraint_2  = (np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))  + (np.zeros(2,) + 20)
                    #np.random.multivariate_normal((np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))
                                                                                                             # ,np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) ),1) \
                                                                                                                #  - (np.zeros(2,) + 20) 
                #(np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,) ).reshape(2,)
                sigma_phi_i = (np.dot(  data_sigma[t,:].reshape(dU,dU), np.transpose(data_sigma[t,:].reshape(dU,dU)) ))
                sigma_phi_i_det = np.linalg.det(sigma_phi_i)
                inv_sigma_phi_i = np.linalg.inv(sigma_phi_i)
                sigma_phi =np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)) )
                sigma_phi_det = np.linalg.det(sigma_phi)
                inv_sigma_phi = np.linalg.inv(sigma_phi)
                #mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_t_smooth.reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
                #mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_t_smooth.reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
                """ mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_sim[t,:].reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
                mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_sim[t,:].reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
                kl_term =  (0.5 * (  np.log( np.true_divide( sigma_phi_det, (sigma_phi_i_det) )  ) \
                    + np.trace( np.dot(sigma_phi,sigma_phi_i) ) +  multi_dot([np.transpose(mean_phi-mean_phi_i), inv_sigma_phi, \
                        (mean_phi-mean_phi_i)])  ) ).reshape(1,)  """

                #print kl_term.shape 
            else: break
        """ Expec_log_joint=-1* ( Expec_log_joint_1st_term +Expec_log_joint_sum  - k[18]* con_term -\
             k[19]* (np.dot(   np.ones((1,2)) ,  control_constraint_1.reshape((2,1)) )).reshape(1,)   )   -\
                  k[19]* (np.dot(   np.ones((1,2)) ,  control_constraint_2.reshape((2,1)) )).reshape(1,)     """ 
        Expec_log_joint=-1* ( Expec_log_joint_1st_term +Expec_log_joint_sum  )  
        return Expec_log_joint
    """ calculate_complete_loglikelihood(T,state_dim,action_dim,A_kal,B_kal,data_K,data_k,
                                            data_sig,data_X,s_hat,covrew,
                                            reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn
                                            ,R_u_dyn,x_est_smooth,cov_smooth,M,A_total) """


    #@njit 
    #@cuda.jit(nopython=True)
    @jit(nopython=True)
    #@jit(target ="cuda")   
    def f_for_GPU(k):  
        x_1_n=x_est_smooth[0,:].reshape(dX,1)
        
        mu_1=initial_state_mu.reshape(dX,1)


        x_sim_inside=np.zeros((T,state_dim))
        u_sim_inside=np.zeros((T,action_dim))
        reward_inside=np.zeros((T,))

        #############################################################
        ### Expectation expression for the log likelihood
        #############################################################
        temp = np.dot(x_1_n, np.transpose(x_1_n) ) + p_1_n - np.dot(x_1_n,np.transpose(mu_1)) -  np.dot( mu_1,np.transpose(x_1_n) ) \
            - np.dot( mu_1,np.transpose(mu_1) )
        Expec_log_joint_1st_term=-.5 *( np.trace(np.dot( np.linalg.inv( p_1_n ), temp )))  -.5 * np.linalg.det(p_1_n)
        con_term = 0
        
        #print "shape of the A_total is",A_total.shape
        Expec_log_joint_sum=0 
        for t in range (T-1):
            """ if t==0:
                x_sim_inside[0,:]=  initial_state_mu#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
            else:
                x_sim_inside[t,:]= np.dot(A_kal[t,:,:],x_sim[t-1,:]) + np.dot(B_kal[t,:,:],u_sim[t-1,:]) + np.dot(s_hat[t,:,:], np.dot(np.linalg.inv(covrew[t,:]),reward[t-1])).reshape(1,state_dim) +w_t_kal[t,:]
            u_sim_inside[t,:]=np.random.multivariate_normal((np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))\
                                                                                                              ,np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) ),1)    
            reward[t] =  (np.dot(R_x_dyn[t,:,:],x_sim[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn[t,:,:],u_sim[t,:].reshape(2,1)) +v_t[t,:] ).reshape(1,)
            """
            
            sigma_total=  np.vstack((np.hstack (( Sigmadyn[t,:,:] , s_hat[t,:,:] )) ,np.hstack( ( np.transpose(s_hat[t,:,:])  ,covrew[t,:]))))  
            #print "shape of the sigma total is ",sigma_total.shape

            A_total= np.vstack((np.hstack((A_x_dyn[t,:,:],B_u_dyn[t,:,:])) , np.hstack((R_x_dyn[t,:,:],R_u_dyn[t,:,:])) ))
            
            x_t_smooth = x_est_smooth[t,:].reshape(dX,1)
            x_t_plus_1_smooth = x_est_smooth[t+1,:].reshape(dX,1)
            x_t_x_t=np.dot(x_t_smooth,np.transpose(x_t_smooth))+cov_smooth[t,:,:]
            x_t_plus_1_r_t_transpose= np.dot(np.dot(x_t_plus_1_smooth, np.transpose( x_t_smooth ) ) + M[t+1,:,:] , np.transpose(R_x_dyn[t,:,:])  + \
                np.dot( np.transpose(k[:(dU*dX)].reshape(dU,dX)) , np.transpose(R_u_dyn[t,:,:]) ) \
            ) + np.dot( x_t_plus_1_smooth ,   np.dot( np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,1)) , np.transpose(R_u_dyn[t,:,:]) )  )
            u_t_x_t_transpose = np.transpose  (np.dot(x_t_x_t , np.transpose(k[:(dU*dX)].reshape(dU,dX)) )   +\
                           np.dot( x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape (dU,1) ) ) )
            r_t_r_t_1st_term =  np.dot    (np.dot(  R_x_dyn[t,:,:], x_t_x_t) , np.transpose (R_x_dyn[t,:,:]) ) 
            r_t_r_t_2nd_term =  np.dot  (  np.dot(R_x_dyn[t,:,:] ,np.transpose(u_t_x_t_transpose) ) , np.transpose (R_u_dyn[t,:,:])   )   
            r_t_r_t_3rd_term= np.transpose(r_t_r_t_2nd_term)
            r_t_r_t_4th_term = covrew[t,:]
            u_t_u_t_transpose= np.dot(  np.dot ( k[:(dU*dX)].reshape(dU,dX) ,  np.dot( x_t_smooth, np.transpose(x_t_smooth) ) +cov_smooth[t,:,:] ) , np.transpose(k[:(dU*dX)].reshape(dU,dX))   ) \
            +np.dot (  k[:(dU*dX)].reshape(dU,dX), np.dot(x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1)) )  ) \
                +  np.transpose ( np.dot (  k[:(dU*dX)].reshape(dU,dX), np.dot(x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1)) )  ) )  \
            + np.dot(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1) ,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))  ) \
                + np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)) )   

            r_t_r_t_5th_term = np.dot (np.dot (  R_u_dyn[t,:,:] , u_t_u_t_transpose ) , np.transpose(R_u_dyn[t,:,:])  )

            r_t_r_t=r_t_r_t_1st_term+r_t_r_t_2nd_term+r_t_r_t_3rd_term+ r_t_r_t_4th_term +r_t_r_t_5th_term

            zeta_zeta=  np.vstack( ( np.hstack(  (    np.dot(x_t_plus_1_smooth, np.transpose(x_t_plus_1_smooth)) + cov_smooth[t+1,:,:]  \
            , x_t_plus_1_r_t_transpose   ) )  ,   np.hstack(  (   np.transpose(x_t_plus_1_r_t_transpose)  \
            ,  r_t_r_t  ) )   )  ) 

            #print  "shape of zeta zeta is",zeta_zeta.shape
            
            zeta_z_1 = np.hstack(( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:]  \
            , np.dot( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:], np.transpose(k[:(dU*dX)].reshape(dU,dX))  ) \
            +  np.dot (  x_t_plus_1_smooth,np.transpose (k[(dU*dX): (dU*dX)+dU ].reshape(dU,1))    ) ) )

            

            r_t_x_t_transpose=   np.dot(R_x_dyn[t,:,:],x_t_x_t ) + np.dot(R_u_dyn[t,:,:] , \
            u_t_x_t_transpose )  

            r_t_u_t_transpose =  np.dot(np.dot (  R_x_dyn[t,:,:], x_t_x_t ) , np.transpose (k[:(dU*dX)].reshape(dU,dX))  )  + \
                np.dot  (  R_u_dyn[t,:,:] , np.dot( u_t_x_t_transpose ,  np.transpose(k[:(dU*dX)].reshape(dU,dX))  )) \
                    + np.dot(   R_x_dyn[t,:,:] ,  np.dot(x_t_smooth, np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))  ) ) \
                        + np.dot(   R_u_dyn[t,:,:],  np.dot( np.dot(k[:(dU*dX)].reshape(dU,dX),   x_t_smooth)+k[(dU*dX): (dU*dX)+dU ].reshape(dU,1)  \
                            , np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,1)) )       )

            zeta_z_2= np.hstack((  r_t_x_t_transpose,r_t_u_t_transpose ))
            zeta_z = np.vstack (( zeta_z_1,zeta_z_2 ))
            
            #print "shape of zeta z is",zeta_z.shape
            z_z_1=  np.hstack  ((x_t_x_t, np.transpose(u_t_x_t_transpose)))
            z_z_2=  np.hstack(( u_t_x_t_transpose, u_t_u_t_transpose  )) 

            #print "z_z_1 shape and z_z2_shapes are",z_z_1.shape,z_z_2.shape

            z_z = np.vstack((z_z_1,z_z_2))
            #print "z_z shape is ",z_z.shape
            if np.linalg.eigvals(u_t_u_t_transpose).all() >0 : 

            #print u_t_u_t_transpose
                
                Expec_log_joint_2nd_term= -.5 * ( np.trace( np.dot( np.linalg.inv(sigma_total) , (zeta_zeta - np.dot( zeta_z, np.transpose(A_total)) - np.transpose(np.dot( zeta_z, np.transpose(A_total)) )  \

                + np.dot (np.dot(  A_total , z_z) , np.transpose(A_total))    )  )  ) )  -.5 * np.linalg.det(sigma_total)
                Expec_log_joint_sum  = Expec_log_joint_sum +Expec_log_joint_2nd_term#+(reward[t])
                """ con_term = con_term-    np.log(  np.linalg.det(  np.dot( np.eye(2)* 1.e-16 + data_sigma[t,:].reshape(2,2)\
                        , np.transpose(data_sigma[t,:].reshape(2,2)) )) ) +         \
                        np.log(  np.linalg.det(  np.dot( np.eye(2)* 1.e-6 + k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) )) )\
                            - epsilon """
                #control_constraint_1  = (np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))  - (np.zeros(2,) + 20)
                #control_constraint_2  = (np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))  + (np.zeros(2,) + 20)
                    #np.random.multivariate_normal((np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))
                                                                                                             # ,np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) ),1) \
                                                                                                                #  - (np.zeros(2,) + 20) 
                #(np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,) ).reshape(2,)
                sigma_phi_i = (np.dot(  data_sigma[t,:].reshape(dU,dU), np.transpose(data_sigma[t,:].reshape(dU,dU)) ))
                sigma_phi_i_det = np.linalg.det(sigma_phi_i)
                inv_sigma_phi_i = np.linalg.inv(sigma_phi_i)
                sigma_phi =np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)) )
                sigma_phi_det = np.linalg.det(sigma_phi)
                inv_sigma_phi = np.linalg.inv(sigma_phi)
                #mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_t_smooth.reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
                #mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_t_smooth.reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
                """ mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_sim[t,:].reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
                mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_sim[t,:].reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
                kl_term =  (0.5 * (  np.log( np.true_divide( sigma_phi_det, (sigma_phi_i_det) )  ) \
                    + np.trace( np.dot(sigma_phi,sigma_phi_i) ) +  np.dot(np.dot ( np.transpose(mean_phi-mean_phi_i), inv_sigma_phi), \
                        (mean_phi-mean_phi_i)   )  ) ).reshape(1,)  """

                #print kl_term.shape 
            else: break
        """ Expec_log_joint=-1* ( Expec_log_joint_1st_term +Expec_log_joint_sum  - k[18]* con_term -\
             k[19]* (np.dot(   np.ones((1,2)) ,  control_constraint_1.reshape((2,1)) )).reshape(1,)   )   -\
                  k[19]* (np.dot(   np.ones((1,2)) ,  control_constraint_2.reshape((2,1)) )).reshape(1,)     """ 
        Expec_log_joint=-1* ( Expec_log_joint_1st_term +Expec_log_joint_sum  )  
        return Expec_log_joint




    def f_recent(k):  
        
        x_1_n=x_est_smooth[0,:].reshape(dX,1)
        
        mu_1=initial_state_mu.reshape(dX,1)


        x_sim_inside=np.zeros((T,state_dim))
        u_sim_inside=np.zeros((T,action_dim))
        reward_inside=np.zeros((T,))

        #############################################################
        ### Expectation expression for the log likelihood
        #############################################################
        temp = np.dot(x_1_n, np.transpose(x_1_n) ) + p_1_n - np.dot(x_1_n,np.transpose(mu_1)) -  np.dot( mu_1,np.transpose(x_1_n) ) \
            - np.dot( mu_1,np.transpose(mu_1) )
        Expec_log_joint_1st_term=-.5 *( np.trace(np.dot( np.linalg.inv( p_1_n ), temp )))  -.5 * np.linalg.det(p_1_n)
        con_term = 0
        
        #print "shape of the A_total is",A_total.shape
        Expec_log_joint_sum=0 
        t=counter
        #print "======",t
        #for t in range (T-1):
        """ if t==0:
            x_sim_inside[0,:]=  initial_state_mu#np.random.multivariate_normal( initial_state_mu , p_1_n ,1)    #np.random.randn(state_dim).reshape(1,state_dim)
        else:
            x_sim_inside[t,:]= np.dot(A_kal[t,:,:],x_sim[t-1,:]) + np.dot(B_kal[t,:,:],u_sim[t-1,:]) + np.dot(s_hat[t,:,:], np.dot(np.linalg.inv(covrew[t,:]),reward[t-1])).reshape(1,state_dim) +w_t_kal[t,:]
        u_sim_inside[t,:]=np.random.multivariate_normal((np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))\
                                                                                                            ,np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) ),1)    
        reward[t] =  (np.dot(R_x_dyn[t,:,:],x_sim[t,:].reshape(state_dim,1))+ np.dot(R_u_dyn[t,:,:],u_sim[t,:].reshape(2,1)) +v_t[t,:] ).reshape(1,)
        """
        
        sigma_total=  np.vstack((np.hstack (( Sigmadyn[t,:,:] , s_hat[t,:,:] )) ,np.hstack( ( np.transpose(s_hat[t,:,:])  ,covrew[t,:]))))  
        #print "shape of the sigma total is ",sigma_total.shape

        A_total= np.vstack((np.hstack((A_x_dyn[t,:,:],B_u_dyn[t,:,:])) , np.hstack((R_x_dyn[t,:,:],R_u_dyn[t,:,:])) ))
        
        x_t_smooth = x_est_smooth[t,:].reshape(dX,1)
        x_t_plus_1_smooth = x_est_smooth[t+1,:].reshape(dX,1)
        x_t_x_t=np.dot(x_t_smooth,np.transpose(x_t_smooth))+cov_smooth[t,:,:]
        x_t_plus_1_r_t_transpose= np.dot(np.dot(x_t_plus_1_smooth, np.transpose( x_t_smooth ) ) + M[t+1,:,:] , np.transpose(R_x_dyn[t,:,:])  + np.dot( np.transpose(k[:(dU*dX)].reshape(dU,dX)) , np.transpose(R_u_dyn[t,:,:]) ) \
        ) + np.dot( x_t_plus_1_smooth ,   np.dot( np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,1)) , np.transpose(R_u_dyn[t,:,:]) )  )
        u_t_x_t_transpose = np.transpose  (np.dot(x_t_x_t , np.transpose(k[:(dU*dX)].reshape(dU,dX)) )   +\
                        np.dot( x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape (dU,1) ) ) )
        r_t_r_t_1st_term =  multi_dot([ R_x_dyn[t,:,:], x_t_x_t , np.transpose (R_x_dyn[t,:,:]) ]) 
        r_t_r_t_2nd_term =  multi_dot([R_x_dyn[t,:,:] ,np.transpose(u_t_x_t_transpose) , np.transpose (R_u_dyn[t,:,:]) ]  )   
        r_t_r_t_3rd_term= np.transpose(r_t_r_t_2nd_term)
        r_t_r_t_4th_term = covrew[t,:]
        u_t_u_t_transpose= ( multi_dot( [k[:(dU*dX)].reshape(dU,dX) ,  np.dot( x_t_smooth, np.transpose(x_t_smooth) ) +cov_smooth[t,:,:] , np.transpose(k[:(dU*dX)].reshape(dU,dX))] ) \
        +multi_dot([k[:(dU*dX)].reshape(dU,dX),x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))])\
             +  np.transpose (multi_dot([k[:(dU*dX)].reshape(dU,dX),x_t_smooth,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1))]) )  \
        + np.dot(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1) ,np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1)) \
             ) + np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(dU,dU)) )   )

        r_t_r_t_5th_term = multi_dot ([ R_u_dyn[t,:,:] , u_t_u_t_transpose , np.transpose(R_u_dyn[t,:,:]) ])

        r_t_r_t=r_t_r_t_1st_term+r_t_r_t_2nd_term+r_t_r_t_3rd_term+ r_t_r_t_4th_term +r_t_r_t_5th_term

        zeta_zeta=  np.vstack( ( np.hstack(  (    np.dot(x_t_plus_1_smooth, np.transpose(x_t_plus_1_smooth)) + cov_smooth[t+1,:,:]  \
        , x_t_plus_1_r_t_transpose   ) )  ,   np.hstack(  (   np.transpose(x_t_plus_1_r_t_transpose)  \
        ,  r_t_r_t  ) )   )  ) 

        #print  "shape of zeta zeta is",zeta_zeta.shape
        
        zeta_z_1 = np.hstack(( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:]  \
        , np.dot( np.dot (x_t_plus_1_smooth,np.transpose(x_t_smooth)) + M[t+1,:,:], np.transpose(k[:(dU*dX)].reshape(dU,dX))  ) \
        + multi_dot([x_t_plus_1_smooth,np.transpose (k[(dU*dX): (dU*dX)+dU ].reshape(dU,1))])) )

        

        r_t_x_t_transpose=   np.dot(R_x_dyn[t,:,:],x_t_x_t ) + np.dot(R_u_dyn[t,:,:] , \
        u_t_x_t_transpose )  

        r_t_u_t_transpose = multi_dot([ R_x_dyn[t,:,:], x_t_x_t , np.transpose (k[:(dU*dX)].reshape(dU,dX)) ])  + \
            multi_dot([ R_u_dyn[t,:,:] ,  u_t_x_t_transpose ,  np.transpose(k[:(dU*dX)].reshape(dU,dX)) ]) \
                + multi_dot([R_x_dyn[t,:,:] ,x_t_smooth, np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,).reshape(dU,1)) ])+ \
                     multi_dot([ R_u_dyn[t,:,:], np.dot(k[:(dU*dX)].reshape(dU,dX),x_t_smooth)+k[(dU*dX): (dU*dX)+dU ].reshape(dU,1) , np.transpose(k[(dU*dX): (dU*dX)+dU ].reshape(dU,1))])

        zeta_z_2= np.hstack((  r_t_x_t_transpose,r_t_u_t_transpose ))
        zeta_z = np.vstack (( zeta_z_1,zeta_z_2 ))
        
        #print "shape of zeta z is",zeta_z.shape
        z_z_1=  np.hstack  ((x_t_x_t, np.transpose(u_t_x_t_transpose)))
        z_z_2=  np.hstack(( u_t_x_t_transpose, u_t_u_t_transpose  )) 

        #print "z_z_1 shape and z_z2_shapes are",z_z_1.shape,z_z_2.shape

        z_z = np.vstack((z_z_1,z_z_2))
        #print "z_z shape is ",z_z.shape
        if np.linalg.eigvals(u_t_u_t_transpose).any() >0 : 

        #print u_t_u_t_transpose
        
            Expec_log_joint_2nd_term= -.5 * ( np.trace( np.dot( np.linalg.inv(sigma_total) , (zeta_zeta - np.dot( zeta_z, np.transpose(A_total)) - np.transpose(np.dot( zeta_z, np.transpose(A_total)) )  \

            + multi_dot ([A_total , z_z , np.transpose(A_total)])  )  )) )  -.5 * np.linalg.det(sigma_total)
            Expec_log_joint_sum  = Expec_log_joint_sum +Expec_log_joint_2nd_term#+(reward[t])
            """ con_term = con_term-    np.log(  np.linalg.det(  np.dot( np.eye(2)* 1.e-16 + data_sigma[t,:].reshape(2,2)\
                    , np.transpose(data_sigma[t,:].reshape(2,2)) )) ) +         \
                    np.log(  np.linalg.det(  np.dot( np.eye(2)* 1.e-6 + k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) )) )\
                        - epsilon
            control_constraint_1  = (np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))  - (np.zeros(2,) + 20)
            control_constraint_2  = (np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))  + (np.zeros(2,) + 20) """
                #np.random.multivariate_normal((np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,))
                                                                                                            # ,np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2) , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) ),1) \
                                                                                                            #  - (np.zeros(2,) + 20) 
            #(np.dot(k[:(dU*dX)].reshape((2,state_dim)),x_sim[t,:])+k[(dU*dX): (dU*dX)+dU ].reshape(2,) ).reshape(2,) 
            """ sigma_phi_i = (np.dot(  data_sigma[t,:].reshape(2,2), np.transpose(data_sigma[t,:].reshape(2,2)) ))
            sigma_phi_i_det = np.linalg.det(sigma_phi_i)
            inv_sigma_phi_i = np.linalg.inv(sigma_phi_i)
            sigma_phi =np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) )
            sigma_phi_det = np.linalg.det(sigma_phi)
            inv_sigma_phi = np.linalg.inv(sigma_phi) """
            #mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_t_smooth.reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
            #mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_t_smooth.reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
            """ kl_term = (0.5 * (  np.log( np.true_divide( sigma_phi_det, (sigma_phi_i_det) )  ) \
                + np.trace( np.dot(sigma_phi,sigma_phi_i) ) +  multi_dot([np.transpose(mean_phi-mean_phi_i), inv_sigma_phi, \
                    (mean_phi-mean_phi_i)])  ) ).reshape(1,) """
        """ Expec_log_joint=-1* ( Expec_log_joint_1st_term +Expec_log_joint_sum  - k[18]* con_term -\
                k[19]* (np.dot(   np.ones((1,2)) ,  control_constraint_1.reshape((2,1)) )).reshape(1,)   )   -\
                    k[19]* (np.dot(   np.ones((1,2)) ,  control_constraint_2.reshape((2,1)) )).reshape(1,)     """ 
        Expec_log_joint=-1* ( Expec_log_joint_1st_term +Expec_log_joint_sum   )  
        return Expec_log_joint

    data_sig_chol=np.zeros((T,dU,dU))
    chol_covrew=np.zeros((T,1,1))
    chol_sigma_dyn=np.zeros((T,state_dim,state_dim))
    for t in range(T):
        if is_pd(data_sig[t,:,:].reshape(dU,dU)) :
            data_sig_chol[t,:,:] =  np.linalg.cholesky(data_sig[t,:,:])
            chol_sigma_dyn[t,:,:]=np.linalg.cholesky( Sigmadyn[t,:,:] )
            chol_covrew[t,:,:]=np.linalg.cholesky(covrew[t,:,:])
        else: 
            print "Breaking.....at line 401" 
            break

    maxiteration=1500
    #print "the dat sigma cholesky is", data_sigma[0,:,:]
    def constraint(k):
        t=counter
        sigma_phi_i = (np.dot(  data_sigma[t,:].reshape(2,2), np.transpose(data_sigma[t,:].reshape(2,2)) ))
        sigma_phi_i_det = np.linalg.det(sigma_phi_i)
        inv_sigma_phi_i = np.linalg.inv(sigma_phi_i)
        sigma_phi =np.dot( k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)  , np.transpose(k[(dU*dX)+dU : ( (dU*dX)+dU + dU*dU) ].reshape(2,2)) )
        sigma_phi_det = np.linalg.det(sigma_phi)
        inv_sigma_phi = np.linalg.inv(sigma_phi)
        #mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_t_smooth.reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
        #mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_t_smooth.reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
        mean_phi = (np.dot(k[:(dU*dX)].reshape(dU,dX),x_sim[t,:].reshape(6,1)) + k[(dU*dX): (dU*dX)+dU ].reshape(2,1) ).reshape(2,1)
        mean_phi_i = (np.dot(data_K[t,:,:].reshape(dU,dX),x_sim[t,:].reshape(6,1)) + data_k[t,:].reshape(2,1) ).reshape(2,1)
        kl_term =  (0.5 * ( np.log( np.true_divide( sigma_phi_det, (sigma_phi_i_det) )  ) \
            + np.trace( np.dot(sigma_phi,sigma_phi_i) ) +  multi_dot([np.transpose(mean_phi-mean_phi_i), inv_sigma_phi, \
                (mean_phi-mean_phi_i)])  ) ).reshape(1,) 
        #print kl_term
        return kl_term-0.002

    #params_during_iteration=np.zeros((maxiteration,18))
    #ll_values_during_opt=np.zeros(maxiteration)
    epsilon =0.4
    def callbackF(Xi):
        global Nfeval
        global function_opt_intermed_vals
        #print "value of Xi is ",Xi
        #f_temp=f(Xi)
        #print Nfeval
        print " fval:::: ",f(Xi)
        #ll_values_during_opt[Nfeval]=f(Xi)
        #print ":::::", ll_values_during_opt[Nfeval]
        #params_during_iteration[Nfeval]=Xi
        #print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[:12], Xi[12:14], Xi[14:16], f(Xi))
        Nfeval += 1

    #############################################################
    ### LBFGS Optimization of the Joint Complete Likelihood 
    ### of the rewards and latent variables 
    #############################################################
    #k_param=np.hstack ((k1.reshape(1,12),k2.reshape(1,2),policy_sigma.reshape(1,4)))
    #f(k_param.reshape((18,)))


    #print "k_param is ",k_param
    #print "the original value fo the function with the initial params are",f(k_param.reshape((18,)))
    total_dimension_param_space = (dU*dX)+ dU+ (dU*dU)+1
    updated_params_fvalue=np.zeros((T,1))
    updated_params=np.zeros((T,total_dimension_param_space))
    #updated_params=np.zeros((T,102))
    global counter
    output =  open(file_name+'updated_em_params%s_optiter_Without%s.pkl'%(argument[1],argument[2]), 'wb') 
    output_fval = open(file_name+'updated_em_params_fval%s_optiter_Without%s.pkl'%(argument[1],argument[2]), 'wb')#open('./run/Qasir/updated_em_params_fval%s.pkl'% iterator, 'wb')
    output_fval_intermediate_vals = open( file_name + 'intermediate_fvals.pkl', 'wb')
    #param_controller=np.zeros((1,1)) + 0.1
    param_kl = np.zeros((1,1))
    time1 = timeit.timeit()

    if (argument[0]==0):
        for t in range(T):
            counter =counter-1
            start=timer()
            [xopt, fopt, gopt, Bopt, func_calls] \
                    =  optimize.fmin_cg(f_for_GPU, \
                                            (np.hstack((data_K[t,:,:].reshape(1,dU*dX),data_k[t,:].reshape(1,dU),data_sigma[t,:,:].reshape(1,dU*dU) ,
                                            param_kl  ))).reshape((total_dimension_param_space,)  ) \
                                                    ,callback=callbackF,maxiter=argument[2], full_output=True)
            print("without GPU::::::::::::::::::::::", timer()-start) 
            """ data_K[t,:,:] = xopt[:dX*dU].reshape(dU,dX)
            data_k[t,:] = xopt[dX*dU:(dX*dU+dU)].reshape(dU,)
            data_sig [t,:,:] = xopt[(dX*dU+dU):(dX*dU+dU)+(dU*dU)].reshape(dU,dU)
            x_est_smooth,cov_smooth,M,x_est,p =  robust_kf_ks_time_vary_change(T,state_dim,action_dim,A_kal\
                                    ,B_kal,data_K,data_k,data_sig,data_X,s_hat,covrew\
                                            ,reward,initial_state_mu,p_1_n,covdyn_kal,R_x_dyn,R_u_dyn,u_sim,x_sim,dU,dX) """
            print "finished",t,"iterations in"
            print "xopt shape is", xopt.shape
            updated_params[t,:]=xopt
            updated_params_fvalue[t,:]=fopt
        #print updated_params
    """ if (argument[0]==1):
        con1 = {'type': 'ineq', 'fun': constraint}
        #con2 = {'type': 'ineq', 'fun': control_constraint}
        cons = ([con1])
        #nonlinear_constraint = ([NonlinearConstraint(cons_fun, -np.inf, 0)])

        #linear_constraint = LinearConstraint(-np.inf, 0)
        
        for t in range(T):
            solution  = optimize.minimize(f, (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2),data_sigma[t,:,:].reshape(1,4) ,param_kl  )  )  ).reshape((19,)) \
                , method="SLSQP",constraints=cons, callback=callbackF,options={'maxiter': 40})
                            #,callback=callbackF,maxiter=100, full_output=True, retall=False)
            #eta = eta - step_multiplier * (epsilon -   temporary_Term )
            #print fopt
            print "The time step iteration number is",t
            #print "value of eta is ",eta 
            for i in range(T):
            #print xopt
                updated_params[i,:]=solution.x
                updated_params_fvalue[i,:]=solution.fun """
    time2 = timeit.timeit()
    print "updated parameters shape is", updated_params.shape
    #print ' function optimization took %0.3f ms' % ( (time2-time1)*1000.0)
    pickle.dump(updated_params, output)
    pickle.dump(updated_params_fvalue, output_fval)
    
    print "The parameters are updated and dumped into the picke file"
    output.close()
    output_fval.close()

    print "~~~~~~.....The EM algorithm finished successfully......~~~~~~~" 


    """ll_values_during_opt_file= open(file_name+'ll_values_during_opt%s.pkl'%argument[1], 'wb') 
    pickle.dump(ll_values_during_opt, ll_values_during_opt_file)
    ll_values_during_opt_file.close()
 
    params_during_opt_file= open(file_name+'params_during_opt%s.pkl'%argument[1], 'wb') 
    pickle.dump(params_during_iteration, params_during_opt_file)
    params_during_opt_file.close() """
    
    #############################################################
    ### retrieving the updated EM k parameters from the .pkl file
    #############################################################
    updated_em = open(file_name+'updated_em_params%s.pkl'%argument[1], 'rb')
    params = pickle.load(updated_em)
    updated_em_params_fval = open(file_name+'updated_em_params_fval%s.pkl'%argument[1], 'rb')
    params_fval = (pickle.load(updated_em_params_fval)).reshape((T,))
    
    
    updated_em.close()
    updated_em_params_fval.close()
    for t in range(T):
        params[t,((dX*dU)+dU):((dX*dU)+dU+(dU*dU)) ]= ( np.dot (params[t,((dX*dU)+dU):((dX*dU)+dU+(dU*dU))].reshape(dU,dU) , \
            np.transpose(params[t,((dX*dU)+dU):((dX*dU)+dU+(dU*dU))].reshape(dU,dU)))).reshape(1,dU*dU)
        np.linalg.cholesky(params[t,((dX*dU)+dU):((dX*dU)+dU+(dU*dU))].reshape(dU,dU))
        #params[t,18+36+12:102]= ( np.dot (params[t,18+36+12:102].reshape(6,6) , np.transpose(params[t,18+36+12:102].reshape(6,6)))).reshape(1,36)
        #params[t,110:111]= np.dot( params[t,110:111].reshape(1,1), np.transpose(params[t,110:111].reshape(1,1))  ).reshape(1,1)

    #############################################################
    ### Testing the parameters and producing the updated EM trajectory coordinates
    #############################################################
    nu=np.array([.8])

    
    """ xi_orig = np.zeros((T,18))
    xi_new  = params
    for i in range(T):
        xi_orig [i,:] =  (np.hstack((data_K[t,:,:].reshape(1,12),data_k[t,:].reshape(1,2),data_sigma[t,:,:].reshape(1,4)))).reshape((18,))
        print "likelihood_values_before_em",f(xi_orig [i,:] )
        print " -- "
        print "likelihood_values_after_em", f(xi_new[i,:] ) """
        

    EM_REW, ORIG_REW, EM_LLS, ORIG_LLS = print_results_time_vary(f,T,state_dim,action_dim,params,\
                    A_x_dyn,B_u_dyn,w_t_kal,w_t,Adyn,Bdyn,Sigmadyn,data_K,data_k,data_sig,t_s,params_fval,initial_state_mu,p_1_n,A_kal,B_kal,s_hat,covrew,reward,\
                        R_x_dyn,R_u_dyn,nu,argument[1],multiply_factor_for_exponent,mean_reward,file_name)
    return 0, 0, 0, 0,0
    #return EM_REW, ORIG_REW, EM_LLS, ORIG_LLS, params