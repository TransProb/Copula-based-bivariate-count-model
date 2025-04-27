# -*- coding: utf-8 -*-
"""

Author: Jianbiao Wang  
Email: wang.jianbiao98@gmail.com


"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
from optimparallel import minimize_parallel
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from scipy.stats import norm
from numdifftools import Hessian
import time
import warnings
import math
from mpmath import mp, matrix
import os
mp.dps = 50

warnings.filterwarnings('ignore')    

def find_Zeroindex(lik):
    found = False
    
    for i in range(lik.rows):
        if lik[i, 0] == 0:
            
            found = True
            break
        
    return i



class GPsn (object):
    
    def __init__(self, ZI, model, ZI_2, model_2, typ):
        
        self.ZI = ZI #define zero-inflation or not
        self.ZI_2 = ZI_2 #define zero-inflation or not
        self.core_model = model # define ZIP or Poisson model
        self.core_model_2 = model_2 # define ZIP or Poisson model
        self.theta = 0 #default parameter to be estimated
        self.facList = {i:np.math.factorial(i) for i in range(33)} #dict of factorial number
        self.type = typ #define copula type
        # self.NBPara = 0 
        self.Niter = 1 #define iteration in callback function
         
        
    # define ZI or not, Poisson or negative bionomial Poisson expression for one observation
    # input:
        # lam : expected value lambda ~ 1x1
        # u: utility for being zero part ~ 1x1
        # y: dependent variable ~ 1x1
        # NBPara: overdispersion parameter for NB distribution ~ 1x1
    # output: 
        # p: probability density function (pdf) for each observation ~ 1x1
    def psn(self, lam, u, y, NBPara, ind): 
        
        u_mp = mp.mpf(u)
        lam_mp = mp.mpf(lam)   
        y_mp = mp.mpf(y)       
        # phi_mp = mp.mpf(phi)   
        fac_y_mp = mp.mpf(self.facList[y])  
        NBPara_mp = mp.mpf(NBPara)  
        
        if ind == 1:
            zi = self.ZI
            ml = self.core_model
        else:
            zi = self.ZI_2
            ml = self.core_model_2
        
        if zi == 'ZI':
            #phi = np.exp(u)/(1+np.exp(u))

            
            
            phi_mp = mp.exp(u_mp) / (1 + mp.exp(u_mp))

        else:
            phi_mp = mp.mpf(0)    
        
        if ml == 'Poisson':
        
            psn_ = mp.exp(-lam_mp) * (lam_mp ** y_mp) / fac_y_mp
            p = phi_mp * (y_mp == 0) + (1 - phi_mp) * psn_
        else: 

            tau_mp = mp.mpf(1) / NBPara_mp  # 计算 tau
            fac_y_mp = mp.mpf(self.facList[y])  # 阶乘列表转换为高精度
            nb_ = (mp.gamma(y_mp + tau_mp) / (mp.gamma(tau_mp) * fac_y_mp)) * \
                  (tau_mp / (tau_mp + lam_mp)) ** tau_mp * \
                  (lam_mp / (tau_mp + lam_mp)) ** y_mp
            p = phi_mp * (y_mp == 0) + (1 - phi_mp) * nb_
    
        if p <= 0:
            print('error')
        if p >=1:
            print('error', p)

        return p
     
    # cumulative function, cumfpsn (y) = [psn(1) + psn(2) + ....+ psn(n)]
    # input:  
        # lam : expected value lambda ~ 1xn   \\n is the number of observation 
        # u: utility for being zero part ~ 1xn 
        # y: dependent variable ~ 1xn 
        # NBPara: overdispersion parameter for NB distribution ~ 1xn; but all the same value 
    # output:  
        # p: cumulative density function (cdf) for all observation ~ 1xn
    def cmfpsn (self, lam ,u, y, NBPara, ind):
       
        p = mp.fsum([self.psn(lam, u, i, NBPara, ind) for i in range(y + 1)]) 
        # if p == 1:
        #     print(lam ,u, y, NBPara, ind)
        
        return p 
    
    # define different copula function
    # input:
        # the: correlation parameter (linear expression) for u1 and u2 probability ~ 1xn 
        # u1: cumulative probability function for unrealized demand ~ 1xn  
        # u2: cumulative probability function for realized demand ~ 1xn  
        # typ: the type of copula ~ str 
    # output:
        # joint probability of realized and unrealized demand considering copula ~ 1xn 
    def copula (self, the, u1, u2, typ):

        if typ == 'indep': # ✔
            
            return matrix([u1[i] * u2[i] for i in range(len(u1))])
            # return u1 * u2 
         
        if typ == 'FGM':  # ✔
        
            #t = 2/(1+np.exp(the)) - 1
            t_m = matrix([2/(1+mp.exp(t)) - 1 for t in the])
            res = matrix([u1_i * u2_i * (1 + t_i * (1-u1_i) * (1-u2_i)) for u1_i, u2_i, t_i in zip(u1, u2, t_m)])
            return res #u1 * u2 * (1 + t * (1-u1) * (1-u2))
         
        if typ == 'Frank': # ✔
            t_m = matrix([mp.mpf(t) for t in the])
            res = matrix([ -1/t_i * mp.log( 1 + ( mp.exp( -t_i * u1_i ) -1 ) * ( mp.exp( -t_i * u2_i ) -1 ) / ( mp.exp( -t_i ) -1 ) )
                          for u1_i, u2_i, t_i in zip(u1, u2, t_m)])
            
            return res #-1/the * np.log( 1 + ( np.exp( -the * u1 ) -1 ) * ( np.exp( -the * u2 ) -1 ) / ( np.exp( -the ) -1 ) )
        
        if typ == 'Joe': #  ✔
            # t = np.exp(the) + 1
            t_m = matrix([mp.exp(t) + 1 for t in the])
            res = []
            for u1_i, u2_i, t_i in zip(u1, u2, t_m):
                if u1_i==0 or u2_i == 0:
                    res.append(0)
                else:
                    res.append(1 - ( (1-u1_i)**t_i + (1-u2_i)**t_i - (1-u1_i)**t_i * (1-u2_i)**t_i )**(1/t_i))

            return matrix(res) # 1 - ( (1-u1)**t + (1-u2)**t - (1-u1)**t * (1-u2)**t )**(1/t)
        
        if typ == 'Gaussian': # 

            t_m = matrix([2/(1+mp.exp(t)) - 1 for t in the])
            def integrand(t, u1, u2):
                """外部定义的被积函数，已适配参数 u1, u2"""
                exp_term = mp.exp(-(u1**2 - 2*t*u1*u2 + u2**2) / (2*(1 - t**2)))
                denom = 2 * mp.pi * mp.sqrt(1 - t**2)
                return exp_term / denom
            res = []
            for u1_i, u2_i, t_i in zip(u1, u2, t_m):
                if u1_i == 0 or u2_i == 0:
                    res.append(0)
                else:
                    u1_i_ = mp.sqrt(2) * mp.erfinv(2 * u1_i - 1)
                    u2_i_ = mp.sqrt(2) * mp.erfinv(2 * u2_i - 1)
                    # print(u1_i_, u2_i_)
                    integral, _ = mp.quad(
                        lambda t: integrand(t, u1_i_, u2_i_),  # 固定 u1_i 和 u2_i
                        [0, t_i],  # 积分区间 [0, t_i]
                        error=True
                    )
                    phi_u1 = 0.5 * (1 + mp.erf(u1_i_ / mp.sqrt(2)))
                    phi_u2 = 0.5 * (1 + mp.erf(u2_i_ / mp.sqrt(2)))

                    res.append(phi_u1 * phi_u2 + integral)
                    
            return matrix(res) 
        
        
        
        if typ == 'Clayton': # ✔

            t_m = matrix([mp.exp(t) for t in the])
            res = []
            for u1_i, u2_i, t_i in zip(u1, u2, t_m):
                if u1_i==0 or u2_i == 0:
                    res.append(0)
                else:
                    res.append((u1_i ** (-t_i) + u2_i ** (-t_i) - 1) ** (-1/t_i) )
                    
            return matrix(res) #(u1 ** (-t) + u2 ** (-t) - 1) ** (-1/t)          

        if typ == 'Gumbel':  # ✔
          
            t_m = matrix([mp.exp(t) + 1 for t in the])
            res = []
            for u1_i, u2_i, t_i in zip(u1, u2, t_m):
                if u1_i==0 or u2_i == 0:
                    res.append(0)
                else:
                    res.append(mp.exp(-( ( -mp.log(u1_i) ) **t_i + ( -mp.log(u2_i) ) **t_i) ** (1/t_i)) )

            return matrix(res)
       
    # define the loglikehood function
    # input:
        # theta: parameter list to be estimated ~ 1xk // k is the number of parameters
        # Ex: explanatory variable for unrealized lam part ~ s1 x n // s1 is the number of variables in Ex
        # Ew: explanatory variable for unrealized zero-utility part ~ s2 x n 
        # Ey: dependent variable --- unrealized trip ~ 1 x n 
        # Sx: explanatory variable for realized lam part  ~ s3 x n 
        # Sw: explanatory variable for realized zero-utility part ~ s4 x n 
        # Sy: dependent variable --- realized trip ~ 1 x n 
        # Cx: explanatory variable for correlation part~  s5 x n 
    # output: 
        # sloglik: the loglikelihood function to be minimized ~ 1x1; func
        
    def ll(self, theta, Ex, Ew, Ey, Sx, Sw, Sy, Cx, stage='1'):
        '''
        theta[ paras of Ex | paras of Ew | paras of Sx | paras of Sw |para of theta correlation| para of tau NB]
        '''
      
        # calculate the lambda and utility for Unrealized trip  
        Elam = np.exp(theta [:len(Ex)] @ Ex)
        # print(theta [:len(Ex)], 'test')
        Eutil = theta[len(Ex) : len(Ex) + len(Ew)] @ Ew
        # calculate the lambda and utility for Realized trip 
        Slam = np.exp(theta [len(Ex) + len(Ew) :len(Ex) + len(Ew) + len(Sx)] @ Sx)
        Sutil = theta[len(Ex) + len(Ew) + len(Sx) : len(Ex) + len(Ew) + len(Sx) + len(Sw) ] @ Sw
        # calculate linear expression for correlation parameter    
        
        if self.ZI == ' ': 
            Eutil = 100 * np.ones(Ex.shape[1])
        if self.ZI_2 == ' ':
            Sutil = 100 * np.ones(Ex.shape[1])
        
        
        if self.type == 'indep':
            Cx = [ ]
      
        the = theta [len(Ex) + len(Ew) + len(Sx) + len(Sw): len(Ex) + len(Ew) + len(Sx) + len(Sw) + len(Cx)] @ Cx 
        # the = np.array([mp.mpf(x) for x in the], dtype=object)
        # define model  
        # If the model is Poisson, then overdispersion parameters ENBPara and SNBPara are NOT to be estimated, 
        # thus, they can be set as any constant value and not used in the model. The reason for setting but not
        # using is because we want to combine the Poisson and NB model in one routine.
        # If the model is NB, then overdispersion parameters are to be estimated. The overdispersion parameter 
        # are the last two parameter
        ENBPara = 1 * np.ones(Ex.shape[1])
        SNBPara = 1 * np.ones(Ex.shape[1])
        if self.core_model != 'Poisson' :
            ENBPara = theta [len(Ex) + len(Ew) + len(Sx) + len(Sw) + len(Cx):
                             len(Ex) + len(Ew) + len(Sx) + len(Sw) + len(Cx) + 1] * np.ones(Ex.shape[1])
        
                
        if self.core_model_2 != 'Poisson' :
            
            aux = int(self.core_model == 'Poisson')
            SNBPara = theta [len(Ex) + len(Ew) + len(Sx) + len(Sw) + len(Cx) + 1 - aux:
                             len(Ex) + len(Ew) + len(Sx) + len(Sw) + len(Cx) + 2 - aux] * np.ones(Ex.shape[1])

         
        Ind = np.ones(Ex.shape[1])
        # calculate the cumulative probability for unrealized trip, Eu: 1xn
        EF = list(map(self.cmfpsn, Elam, Eutil, Ey, ENBPara, Ind))
        Eu = matrix(EF)
        # print(Eu)
        # assert 1 > 3 ,' test'
        assert min(Eu) >= 0 and max(Eu) < 1, 'EU numerical issue, min={}, max={}'.format(min(Eu),max(Eu))
        # Eu = np.array(EF) #+ 1e-20 
       
        # calculate the cumulative probability for unrealized trip with y-1, Eu_: 1xn
        EF_ = list(map(self.cmfpsn, Elam, Eutil, Ey-1, ENBPara, Ind))
        Eu_ = matrix(EF_)
        assert min(Eu_) >= 0 and max(Eu_) < 1, 'EU_ numerical issue, min={}, max={}'.format(min(Eu_),max(Eu_))
        # Eu_ = np.array(EF_) #+ 1e-20

        # calculate the cumulative probability for realized trip, Su: 1xn
        SF = list(map(self.cmfpsn, Slam, Sutil, Sy, SNBPara, Ind+1))
        Su = matrix(SF)
        assert min(Su) >= 0 and max(Su) < 1, 'SU numerical issue, min={}, max={}'.format(min(Su),max(Su))
        # Su = np.array(SF) #+ 1e-20

        # calculate the cumulative probability for realized trip with y-1 , Su_: 1xn
        SF_ = list(map(self.cmfpsn, Slam, Sutil, Sy-1, SNBPara, Ind+1))
        Su_ = matrix(SF_)
        assert min(Su_) >= 0 and max(Su_) < 1, 'SU_ numerical issue, min={}, max={}'.format(min(Su_),max(Su_))
        # Su_ = np.array(SF_) #+ 1e-20
         
        # calculate the copula function
        typ = self.type
        
        # try:
        P1 = self.copula(the, Eu, Su, typ)
        # assert min(P1) > 0 , 'P1_ numerical issue, min={}, max={}'.format(min(P1),max(P1))
        P2 = self.copula(the, Eu_, Su, typ) 
        P3 = self.copula(the, Eu, Su_, typ) 
        P4 = self.copula(the, Eu_, Su_, typ)

        prob_mp = P1 - P2 - P3 + P4
        # print(prob_mp)
        # prob = np.array([float(x) for x in prob_mp])
        
        lik = prob_mp #+ 1e-6
        # if stage == '2':
        #     lik = lik + 1e-10
        # if min(lik) == 0:
        #     indx = find_Zeroindex(lik)
        #     # print(type(P1))
        #     print('------------------------------')
        #     print([P1[indx], P2[indx], P3[indx], P4[indx]])
            
        #     # print(theta)
        #     # print(the[indx])
        #     print(Ey[indx],Sy[indx])
        #     print(Eu[indx],Eu_[indx],Su[indx],Su_[indx])
        #     print(Elam[indx],Eutil[indx])
            # print(indx)
             
        assert min(lik) > 0 and max(lik) < 1, 'numerical issue, min={}, max={}'.format(min(lik),max(lik))
        # calculate the whole contribution ~ 1xn
        log_prob_mp = matrix([mp.log(x) for x in lik])
        log_prob_sum_mp = mp.fsum(log_prob_mp)  # 计算高精度的 sum
        #loglik = np.log(lik)
        #loglik[np.isnan(loglik)] = np.log(1e-20)
        sloglik = -float(log_prob_sum_mp)
            
        return sloglik #, prob1, prob2, prob


    # parameter estimation
    # input:
        # {theta, Ex, Ew, Ey, Sx, Sw, Sy, Cx}: see above function ll
        # bnds : the bounds for parameters to be estimated
        # hes : the way calculating the hessian maxtrix 
        # numdif is numerical difference (accurate but slow); hes is bfgs result (wrong but fast)  
        # detail: show the parameter values in each estimation iteration (True), otherwise (False); default is false
    # output :
        # res: estimation result of minimize_parallel routine
        # est_res: orginized result for result print
    def estimate (self, theta, Ex, Ew, Ey, Sx, Sw, Sy, Cx, bnds, hes = 'numdif', detail=False):
        self.num = hes
        self.samplesize = Ex.shape[1]
        # estimate the parameters with l-bfgs-b method
        timestamp1 = time.time()
        if detail == True:
            #res = opt.minimize(self.ll, theta , method='L-BFGS-B', tol = 1e-8, args=(Ex, Ew, Ey, Sx, Sw, Sy, Cx), callback=self.callbackF)
            res = minimize_parallel(self.ll,theta, bounds=bnds, args=(Ex, Ew, Ey, Sx, Sw, Sy, Cx),callback=self.callbackF)
        else:
            res = minimize_parallel(self.ll,theta, bounds=bnds, args=(Ex, Ew, Ey, Sx, Sw, Sy, Cx))
        timestamp2 = time.time() 
        print('opt. time is: ', timestamp2 - timestamp1)
        
        # calculate the numerical hessian matrix, 
        # the original hessian matrix from L-BFGS-B is not correct (at least for count model)
        # cov_matrix is the inverse matrix of hessian matrix
        f = lambda x: self.ll(x,Ex, Ew, Ey, Sx, Sw, Sy, Cx, stage='2')
        if hes == 'numdif':
            hess_fn = Hessian(f, method='forward', step=1e-2) 
            # hess_fn = Hessian(f, method='forward', step=1e-2)  #step=1e-2 正常情况下就可以得到正确的结，但是有时候，比如估计的参数数值特别大，会导致该t-值计算失败
            hess = hess_fn(res.x)
            self.hess = hess
            # self.hess = hess
            cov_matrix = np.linalg.inv(hess)
            # print(cov_matrix)
        else:
            cov_matrix = res.hess_inv if type(res.hess_inv) == np.ndarray else res.hess_inv.todense()
        print('hess. cal. time is: ', time.time() - timestamp2)
        
        #calculate the standard error and t-test
        stderr = np.sqrt(np.diag(cov_matrix)) 
        t_test = res.x/stderr
        
        # originize the output data, est_res is a dataframe contain "estimated value", "S.E." and 't-value'
        ind = list(U_x.keys())+list(U_w.keys())+list(R_x.keys())+list(R_w.keys())
        if self.type != 'indep':
            ind = ind + list(cor.keys()) 
            
        if self.core_model == 'Poisson' and self.core_model_2 == 'Poisson':
            ind = ind
        elif self.core_model == 'Poisson' or self.core_model_2 == 'Poisson':
            ind= ind + ['tau1']
        else:
            ind= ind + ['tau1','tau2']
            
        est_res = pd.DataFrame(np.array([res.x,stderr,t_test]).T, 
                                columns=['Paras','S.E.','t-value'],  
                                index=ind)
        
        return res, est_res
    
    # print result
    # input:
        # res, est_res : see the output in function estimate
    def print_(self,res, est_res):
                
        print(res.message) # convergence info
        print('ll(0):', -self.ll(self.theta,Ux,Uw,Uy,Rx,Rw,Ry,Cx))  # initial likelihood
        print('ll:', -res.fun)  # convergence likelihood
        print('BIC:', self.Npara * np.log(self.samplesize) + 2*res.fun)
        print(est_res) # result
        
    # define bounds for parameters
    # input :
        # ind : the index list indicating which parameter's bound should be positive(0, +00) ~ type: list
    # output:
        # df_bnds : the list containing the bounds for each parameter ~ type: list 1xk
    def bnds (self, ind):
        
        _default = (None, None) # default is no bounds
        df_bnds = [] # bound list for all parameters
        for i in range(len(self.theta)):
            # for the parameters whose index is NOT in ind, the bounds are (none, none)
            # for the parameters whose index is in ind, the bounds are (0, none)
            if i not in ind:
                df_bnds.append(_default)
            else:
                df_bnds.append((1e-1, None))
         
        return df_bnds
    
    
    # callback function to show the iteration result
    # input:
        # x : the estimated parameter at each iteration 
    def callbackF(self,x):
        Niter = self.Niter 
        txt = f'{Niter: d}'
        for i in range(len(x)): txt += f'\t{x[i]:.4f}'
        print(txt)
        self.Niter += 1
    
    # def callback(self, xk):
    #     print(f"Current x value: {xk}")
        
    
    # generate the initial value and corresponding boundsfor parameters
    def IniPara (self, Ux, Uw, Rx, Rw, Cx):
        
        lenth = len(Ux)+len(Uw)+len(Rx)+len(Rw)+len(Cx)
        if self.type == 'indep':
            lenth = lenth - len(Cx)
            
        theta = np.zeros(lenth) + 1e-1    
        if self.core_model == 'Poisson' and self.core_model_2 == 'Poisson':
            # theta = np.zeros(lenth) + 1e-1
            self.theta = theta
            ind = [-1]  
        elif self.core_model == 'Poisson' or self.core_model_2 == 'Poisson':
            # theta = np.zeros(lenth) + 1e-1
            theta = np.append(theta, 1)
            self.theta = theta
            ind = [len(theta)-1] 
            
        else:
            # theta = np.zeros(lenth+2) + 1
            # theta = np.zeros(lenth) + 1e-1 
            theta = np.append(theta, [1, 1])
            self.theta = theta
            ind = [len(theta)-2,len(theta)-1] 
        bd = self.bnds(ind)
        
        self.Npara = lenth
        return theta, bd
    



    
    
if __name__ == "__main__":
    
    '''
    read dataset and define independent and dependent variables
    '''
   
    path_  = ' '
    data = pd.read_csv(os.path.join(path_, 'data.csv'), sep=',')
    
    # dfs = [data.sample(n=data.shape[0], replace=True, random_state=i) for i in range(30)]
    dfs = [data]
    est_list = [ ]
    #Fixed
    
    for i, df in enumerate(dfs):
    # # x for lambda of trip frequency model (purpose 1)
        try:
            print(i)
            U_x = {} 
            U_x['U_cons'] = np.array(df.Cons)

            # # w for phi of zero-inflated model (purpose 1)
            U_w = {}
            U_w['U_cons_inflat'] = np.array(df.Cons)

            # orginze the data
            Ux = np.array(list(U_x.values()))
            Uw = np.array(list(U_w.values()))
            Uy = np.array(df.Unrealized)

            # # x for lambda of trip frequency model (purpose 2)
            R_x = {} 
            R_x['U_cons'] = np.array(df.Cons)
 
            # # x for lambda of zero-inflated model (purpose 2)
            R_w = {} 
            # R_w['R_cons_inflat'] = np.array(df.Cons) 

            # orginze the data
            Rx = np.array(list(R_x.values()))
            Rw = np.array(list(R_w.values())) 
            Ry = np.array(df.Hospital)
             
            # # independent variables for correlation (purpose 2)
            cor = {}
            cor['cor_cons'] = np.array(df.Cons)
            Cx = np.array(list(cor.values()))

            # define initial value for theta
            # theta = np.zeros(len(Ux)+len(Uw)+len(Rx)+len(Rw)+len(Cx) + 2) + 1e-4
        
            #indep, Joe, Frank, FGM, Gaussian, Clayton, Gumbel
            gpsn = GPsn(ZI = 'ZI', model='Poisson', ZI_2 = ' ', model_2='NB', typ='Gumbel')    #Gumbel
            # initialize the theta and corresponding bound 
            theta, bd = gpsn.IniPara(Ux, Uw, Rx, Rw, Cx)      
            #define the bounds for parameters    
            #ind contains the list of parameter index of which the range should be larger than 0
            #in this case, the overdispersion parameters for NB model is larger than 0
            # ind = [len(theta)-2,len(theta)-1] 
            # bd = gpsn.bnds(ind)
    
            res, est_res = gpsn.estimate (theta, Ux, Uw, Uy, Rx, Rw, Ry, Cx, bnds = bd, hes = 'numdif', detail = False) 
            
            est_list.append(res.x)
        except:
            print(f'numerical problem, unsucessful for {i}')
        
    # est_array = np.array(est_list)
    # est_res['Paras'] = np.mean(est_array, axis=0)  # 按列（每列的均值）
    # est_res['S.E.'] = np.std(est_array, ddof=1, axis=0)    # 按列（每列的标准差）
    # est_res['t-value'] = est_res['Paras'] / est_res['S.E.']
    # print(est_res) 
    gpsn.print_(res, est_res)  
    

