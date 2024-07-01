import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random
import itertools
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import IPython

from dataclasses import dataclass
#from torchvision import datasets, transforms
from typing import Any

from math import log, exp


from model_selection_algorithms.bandit_algs import UCBalgorithm, EXP3


def binary_search(func,xmin,xmax,tol=1e-5):
    ''' func: function
    [xmin,xmax] is the interval where func is increasing
    returns x in [xmin, xmax] such that func(x) =~ 1 and xmin otherwise'''

    assert isinstance(xmin, float)
    assert isinstance(func(0.5*(xmax+xmin)), float)

    l = xmin
    r = xmax
    while abs(r-l) > tol:
        x = 0.5*(r + l)
        if func(x) > 1.0:
            r = x
        else:
            l = x

    x = 0.5*(r + l)
    return x




class UCBHyperparam:

    def __init__(self, m, burn_in = 1, confidence_radius = 2, 
        min_range = 0, max_range = 1, epsilon = 0):
        #self.hyperparam_list = hyperparam_list
        self.ucb_algorithm = UCBalgorithm(m, burn_in = 1, min_range = 0, max_range = 1, epsilon = epsilon)
        #self.discount_factor = discount_factor
        #self.forced_exploration_factor = forced_exploration_factor
        self.m = m
        self.confidence_radius = confidence_radius
        self.burn_in = burn_in
        self.T = 1

        #self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/(1.0*self.m)
        #self.importance_weighted_cum_rewards = np.zeros(self.m)
        #self.T = T
        #self.counter = 0
        #self.anytime = False
        #self.forced_exploration_factor = forced_exploration_factor
        #self.discount_factor = discount_factor
        # if self.anytime:
        #     self.T = 1


    def sample_base_index(self):
        index = self.ucb_algorithm.get_ucb_arm(self.confidence_radius)
        if self.T <= self.burn_in:
            self.base_probas = np.ones(self.m)/(1.0*self.m)
        else:
            self.base_probas = np.zeros(self.m)
            self.base_probas[index] = 1
        self.T += 1
        return index


    def get_distribution(self):
        return self.base_probas

        
    def update_distribution(self, arm_idx, reward, more_info = dict([])):        
        self.ucb_algorithm.update_arm_statistics(arm_idx, reward)




class EXP3Hyperparam:
    def __init__(self,m,T=1000, anytime = False, discount_factor = .9, 
        eta_multiplier = 1, forced_exploration_factor = 0):

        self.exp3_algorithm = EXP3(m, T, anytime = anytime, discount_factor = discount_factor, eta_multiplier = eta_multiplier, forced_exploration_factor = forced_exploration_factor)    

    def sample_base_index(self):
        return self.exp3_algorithm.get_arm()


    def get_distribution(self):
        return self.exp3_algorithm.get_distribution()


    def update_distribution(self, arm_idx, reward, more_info = dict([])):
        self.exp3_algorithm.update_arm_statistics(arm_idx, reward)






class CorralHyperparam:

    def __init__(self,m,T=1000,eta=0.1, anytime = False):
        #self.hyperparam_list = hyperparam_list
        self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/self.m
        self.gamma = 1.0/T
        self.beta = exp(1/log(T))
        self.rho = np.asarray([2*self.m]*self.m)
        self.etas = np.ones(self.m)*eta
        self.T = T
        self.counter = 0
        self.anytime = False
        if self.anytime:
            self.T = 1


    def sample_base_index(self):
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def get_distribution(self):
        return self.base_probas

    
    
    def update_distribution(self, arm_idx, reward, more_info = dict([])):
      loss = 1-reward

      l = np.zeros(self.m)
      p = self.base_probas[arm_idx]
      assert(p>1e-8)
      l[arm_idx] = loss/p  #importance weighted loss vector
      probas_new = self.log_barrier_OMD(self.base_probas,l,self.etas)
      self.base_probas = (1-self.gamma)*probas_new + self.gamma*1.0/self.m
      assert(min(self.base_probas) > 1e-8)

      self.update_etas()

      self.counter += 1
      if self.anytime:
          self.T += 1


    def update_etas(self):
        '''Updates the eta vector'''
        for i in range(self.m):
            if 1.0/self.base_probas[i] > self.rho[i]:
                self.rho[i] = 2.0/self.base_probas[i]
                self.etas[i] = self.beta*self.etas[i]

    def log_barrier_OMD(self,p,loss,etas, tol=1e-5):
        '''Implements Algorithm 2 in the paper
        Updates the probabilities using log barrier function'''
        assert(len(p)==len(loss) and len(loss) == len(etas))
        assert(abs(np.sum(p)-1) < 1e-3)

        xmin = min(loss)
        xmax = max(loss)
        pinv = np.divide(1.0,p)
        thresh = min(np.divide(pinv,etas) + loss) # the max value such that all denominators are positive
        xmax = min(xmax,thresh)

        def log_barrier(x):
            assert isinstance(x,float)
            inv_val_vec = ( np.divide(1.0,p) + etas*(loss-x) )
            if (np.min(np.abs(inv_val_vec))<1e-5):
                print(thresh,xmin,x,loss)

            # print("p ", p)
            # print("etas ", etas)
            # print("loss ", loss)
            # print("x ", x)
            # print("inv val vec ", inv_val_vec)
            assert( np.min(np.abs(inv_val_vec))>1e-5)
            val = np.sum( np.divide(1.0,inv_val_vec) )
            return val

        x = binary_search(log_barrier,xmin,xmax,tol)

        assert(abs(log_barrier(x)-1) < 1e-2)

        inv_probas_new = np.divide(1.0,self.base_probas) + etas*(loss-x)
        assert(np.min(inv_probas_new) > 1e-6)
        probas_new = np.divide(1.0,inv_probas_new)
        assert(abs(sum(probas_new)-1) < 1e-1)
        probas_new = probas_new/np.sum(probas_new)

        return probas_new





### when the descending keyword is active the balancing algorithm starts with 
### high putative bounds and reduces them
class BalancingClassic:
    def __init__(self, m, putative_bounds_multipliers, delta =0.01, 
        c = 1, classic = True):
        
        ### balancing_test_multiplier = c
        ### initial_putative_bound = dmin


        self.minimum_putative = .0001
        self.maximum_putative = 10000

        self.classic = classic ### This corresponds to classic sampling among the algorithms

        self.m = m
        
        self.putative_bounds_multipliers = putative_bounds_multipliers        
        self.balancing_potentials = [p*np.sqrt(1) for p in putative_bounds_multipliers]
        ### We will set a balancing potential to infinity if the algorithm is misspecified.


        self.c = c ### This is the Hoeffding constant
        self.T = 1
        self.delta = delta
        

        self.all_rewards = 0

        ### these store the optimistic and pessimistic estimators of Vstar for all 
        ### base algorithms.


        self.cumulative_rewards = [0 for _ in range(self.m)]
        self.mean_rewards = [0 for _ in range(self.m)]

        self.num_plays = [0 for _ in range(self.m)]

        #self.vstar_lowerbounds = [-float("inf") for _ in range(self.m)]
        #self.vstar_upperbounds = [float("inf") for _ in range(self.m)]


        self.normalize_distribution()
        


    def sample_base_index(self):
        if self.classic:
            return np.argmin(self.balancing_potentials)
        else:
            if sum([np.isnan(x) for x in self.base_probas]) > 0:
                print("Found Nan Values in the sampling procedure for base index")
                IPython.embed()
            sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
            return sample_array[0]


    def normalize_distribution(self):
        if self.classic:
            self.base_probas = [0 for _ in range(self.m)]
            self.base_probas[self.sample_base_index()] = 1 

        else:
            raise ValueError("Not implemented randomized selection rule for the algorithm index. Implement.")
            self.distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]

            normalization_factor = np.sum(self.distribution_base_parameters)
            self.base_probas = [x/normalization_factor for x in self.distribution_base_parameters]
    


    def get_distribution(self):
        return self.base_probas



    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        self.all_rewards += reward

        self.cumulative_rewards[algo_idx] += reward
        self.num_plays[algo_idx] += 1

        #### Update average reward per algorithm so far. 
        self.mean_rewards[algo_idx] = self.cumulative_rewards[algo_idx]*1.0/self.num_plays[algo_idx]


        U_t_lower_bounds = [0 for _ in range(self.m)]
        hoeffding_bonuses = [ self.c*np.sqrt(self.num_plays[i]*np.log((self.num_plays[i]+1)*1.0/self.delta)) for i in range(self.m)]
        #hoeffding_bonuses = [ self.c*np.sqrt(self.num_plays[i]) for i in range(self.m)]


        for i in range(self.m):
            U_t_lower_bounds[i] = (self.cumulative_rewards[i] - hoeffding_bonuses[i])*1.0/np.sqrt(max(self.num_plays[i], 1))


        #U_i_t_upper_bound = (self.cumulative_rewards[algo_idx] - hoeffding_bonuses[algo_idx])*1.0/np.sqrt(self.num_plays[algo_idx])
        U_i_t_upper_bound = (self.cumulative_rewards[algo_idx] + hoeffding_bonuses[algo_idx])*1.0/np.sqrt(self.num_plays[algo_idx])

        curr_algo_reg_upper_bound = self.putative_bounds_multipliers[algo_idx]*np.sqrt(self.num_plays[algo_idx]*np.log(self.num_plays[algo_idx]/self.delta))

        ### Misspecification Test
        if U_i_t_upper_bound + curr_algo_reg_upper_bound < max(U_t_lower_bounds):
            ### algorithm algo_idx is misspecified
            self.balancing_potentials[algo_idx] = float("inf")
        else:
            ### Update balancing potentials
            self.balancing_potentials[algo_idx] = curr_algo_reg_upper_bound

        print("Curr reward ", reward)
        print("All rewards ", self.all_rewards)
        print("Cumulative rewards ", self.cumulative_rewards)
        print("Num plays ", self.num_plays)
        print("Mean rewards ", self.mean_rewards)
        #print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        self.T += 1



        self.normalize_distribution()





### when the descending keyword is active the balancing algorithm starts with 
### high putative bounds and reduces them
class BalancingHyperparamDoublingDataDriven:
    def __init__(self, m, dmin, delta =0.01, 
        c = 1, classic = True, empirical= False):
        
        ### balancing_test_multiplier = c
        ### initial_putative_bound = dmin

        self.empirical = empirical

        self.minimum_putative = .0001
        self.maximum_putative = 10000

        self.classic = classic ### This corresponds to classic sampling among the algorithms

        self.m = m
        
        self.dmin = max(dmin, self.minimum_putative) ### this is dmin
        self.putative_bounds_multipliers = [dmin for _ in range(m)]
        

        self.balancing_potentials = [dmin*np.sqrt(1) for _ in range(m)]


        ### check these putative bounds are going up


        self.c = c ### This is the Hoeffding constant
        self.T = 1
        self.delta = delta
        

        self.all_rewards = 0

        ### these store the optimistic and pessimistic estimators of Vstar for all 
        ### base algorithms.


        self.cumulative_rewards = [0 for _ in range(self.m)]
        self.mean_rewards = [0 for _ in range(self.m)]

        self.num_plays = [0 for _ in range(self.m)]

        #self.vstar_lowerbounds = [-float("inf") for _ in range(self.m)]
        #self.vstar_upperbounds = [float("inf") for _ in range(self.m)]


        self.normalize_distribution()
        


    def sample_base_index(self):
        if self.classic:
            return np.argmin(self.balancing_potentials)
        else:
            if sum([np.isnan(x) for x in self.base_probas]) > 0:
                raise ValueError("Found Nan Values in the sampling procedure for base index")
                
                #IPython.embed()
            sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
            return sample_array[0]


    def normalize_distribution(self):
        if self.classic:
            self.base_probas = [0 for _ in range(self.m)]
            self.base_probas[self.sample_base_index()] = 1 

        else:
            #raise ValueError("Not implemented randomized selection rule for the algorithm index. Implement.")
            distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]

            normalization_factor = np.sum(distribution_base_parameters)
            self.base_probas = [x/normalization_factor for x in distribution_base_parameters]
    


    def get_distribution(self):
        return self.base_probas



    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        self.all_rewards += reward

        self.cumulative_rewards[algo_idx] += reward
        self.num_plays[algo_idx] += 1

        #### Update average reward per algorithm so far. 
        self.mean_rewards[algo_idx] = self.cumulative_rewards[algo_idx]*1.0/self.num_plays[algo_idx]


        U_t_lower_bounds = [0 for _ in range(self.m)]
        hoeffding_bonuses = [ self.c*np.sqrt(self.num_plays[i]*np.log((self.num_plays[i]+1)*1.0/self.delta)) for i in range(self.m)]
        #hoeffding_bonuses = [ self.c*np.sqrt(self.num_plays[i]) for i in range(self.m)]


        for i in range(self.m):
            U_t_lower_bounds[i] = (self.cumulative_rewards[i] - hoeffding_bonuses[i])*1.0/np.sqrt(max(self.num_plays[i], 1))


        #U_i_t_upper_bound = (self.cumulative_rewards[algo_idx] - hoeffding_bonuses[algo_idx])*1.0/np.sqrt(self.num_plays[algo_idx])
        U_i_t_upper_bound = (self.cumulative_rewards[algo_idx] + hoeffding_bonuses[algo_idx])*1.0/np.sqrt(self.num_plays[algo_idx])


        empirical_regret_estimator = self.num_plays[algo_idx]*( max(U_t_lower_bounds) - U_i_t_upper_bound )


        
        if self.empirical:
            clipped_regret = min( empirical_regret_estimator,  2*self.balancing_potentials[algo_idx])
            self.balancing_potentials[algo_idx] = max(clipped_regret, self.balancing_potentials[algo_idx], self.dmin*np.sqrt(self.num_plays[algo_idx]) )
            ### Compute implied putative bound multipliers.
            self.putative_bounds_multipliers[algo_idx] = max(self.balancing_potentials[algo_idx]/np.sqrt(self.num_plays[algo_idx]), self.dmin)



        else:
            ### test for misspecification
            if empirical_regret_estimator > self.putative_bounds_multipliers[algo_idx]*np.sqrt(self.num_plays[algo_idx]):
                self.putative_bounds_multipliers[algo_idx]= min(2*self.putative_bounds_multipliers[algo_idx], self.maximum_putative)


            self.balancing_potentials[algo_idx] = self.putative_bounds_multipliers[algo_idx]*np.sqrt(self.num_plays[algo_idx])





        print("Curr reward ", reward)
        print("All rewards ", self.all_rewards)
        print("Cumulative rewards ", self.cumulative_rewards)
        print("Num plays ", self.num_plays)
        print("Mean rewards ", self.mean_rewards)
        #print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        self.T += 1



        self.normalize_distribution()



class BalancingHyperparamMemoryless():

    def __init__(self, m, H, eta) -> None:
        self.m = m
        self.H = H
        self.eta = eta
        self.base_probs = (1/m)*np.ones(m)
        
    
    def sample_base_index(self):
        base_index = np.argmax(self.base_probs)
        # base_index = np.random.choice(range(self.m), 1, p=self.base_probas)
        return base_index
    

    def update_distribution(self, current_reward, agent, more_info = dict([])):
        base_J_progress = [0]*(self.m)
        for i in range(self.m):
            future_reward = agent.estimate_return_progress(i, self.eta)
            delta_J = future_reward - current_reward
            base_J_progress[i] = delta_J if delta_J>0 else 0

        self.base_probs = base_J_progress/np.sum(base_J_progress)


    def get_distribution(self):
        return self.base_probs



def get_modsel_manager(modselalgo, num_parameters, num_timesteps , parameters = []):

    ### We have anytime options for CORRAL and EXP3. They are not reflected here.

    if modselalgo == "CorralLow":
        modsel_manager = CorralHyperparam(num_parameters,  eta = .1/np.sqrt(num_timesteps), T = num_timesteps) ### hack
    elif modselalgo == "Corral":
        modsel_manager = CorralHyperparam(num_parameters, eta = 1/np.sqrt(num_timesteps), T = num_timesteps) ### hack
    elif modselalgo == "CorralHigh":
        modsel_manager = CorralHyperparam(num_parameters,  eta = 10/np.sqrt(num_timesteps), T = num_timesteps) ### hack    

    elif modselalgo == "CorralSuperHigh":
        modsel_manager = CorralHyperparam(num_parameters,  eta = 50/np.sqrt(num_timesteps), T = num_timesteps) ### hack    
    
    elif modselalgo == "EXP3Low":
            modsel_manager = EXP3Hyperparam(num_parameters, T = num_timesteps, eta_multiplier = 1, discount_factor = 1, 
                forced_exploration_factor = 0)
    elif modselalgo == "EXP3":
        modsel_manager = EXP3Hyperparam(num_parameters, T = num_timesteps, eta_multiplier = 1, discount_factor = 1, 
            forced_exploration_factor = .1)
    elif modselalgo == "EXP3High":
            modsel_manager = EXP3Hyperparam(num_parameters, T = num_timesteps, eta_multiplier = 1, discount_factor = 1, 
                forced_exploration_factor = 1)

    elif modselalgo == "EXP3LowLR":
            modsel_manager = EXP3Hyperparam(num_parameters, T = num_timesteps, eta_multiplier = .1, discount_factor = 1, 
                forced_exploration_factor = .1)
    elif modselalgo == "EXP3HighLR":
            modsel_manager = EXP3Hyperparam(num_parameters, T = num_timesteps, eta_multiplier = 10, discount_factor = 1, 
                forced_exploration_factor = .1)



    elif modselalgo == "UCB":
        modsel_manager = UCBHyperparam(num_parameters)
    
    elif modselalgo == "Greedy":
        modsel_manager = UCBHyperparam(num_parameters, confidence_radius = 0)
    # elif modselalgo == "EpsilonGreedy":
    #     modsel_manager = UCBHyperparam(num_parameters, confidence_radius = 0, epsilon = 0.05)
    

    elif modselalgo == "BalancingClassic":
        modsel_manager = BalancingClassic(num_parameters, putative_bounds_multipliers=parameters, classic = True)

    elif modselalgo in ["DoublingDataDriven", "DoublingDataDrivenBig"]:
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 1, classic = True)

    elif modselalgo in ["EstimatingDataDriven", "EstimatingDataDrivenBig"]:

        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 1, classic = True, empirical = True)


    elif modselalgo == "DoublingDataDrivenStoch":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 1, classic = False)

    elif modselalgo == "EstimatingDataDrivenStoch":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 1, classic = False, empirical = True)


    elif modselalgo == "DoublingDataDrivenMedium":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 3, classic = True)

    elif modselalgo == "EstimatingDataDrivenMedium":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 3, classic = True, empirical = True)

    elif modselalgo == "DoublingDataDrivenHigh":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 10, classic = True)

    elif modselalgo == "EstimatingDataDrivenHigh":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 10, classic = True, empirical = True)

    elif modselalgo == "DoublingDataDrivenSuperHigh":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 50, classic = True)

    elif modselalgo == "EstimatingDataDrivenSuperHigh":
        modsel_manager = BalancingHyperparamDoublingDataDriven(num_parameters, c = 1, dmin = 50, classic = True, empirical = True)


    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))

    return modsel_manager
