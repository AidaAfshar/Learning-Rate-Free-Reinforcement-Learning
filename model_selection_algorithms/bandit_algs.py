import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random
import itertools
import sys

import IPython

from dataclasses import dataclass
#from torchvision import datasets, transforms
from typing import Any

from math import log, exp

class UCBalgorithm:
    def __init__(self, num_arms, burn_in = 1, min_range = -float("inf"), max_range = float("inf"), epsilon = 0, delta = .1):
        self.num_arms = num_arms
        self.mean_estimators = [0 for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]
        self.reward_sums = [0 for _ in range(num_arms)]
        self.burn_in = burn_in
        self.min_range = min_range
        self.max_range = max_range
        self.epsilon = epsilon
        self.delta = delta
        self.global_time_step = 0

    def update_arm_statistics(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.reward_sums[arm_index] += reward
        self.mean_estimators[arm_index] = self.reward_sums[arm_index]/self.counts[arm_index] 
        self.global_time_step += 1

    def get_ucb_arm(self, confidence_radius, arm_info = None ):


        if sum(self.counts) <=  self.burn_in:
            #print("HERE")
            ucb_arm_index = random.choice(range(self.num_arms))
            ucb_arm_value = self.max_range
            lcb_arm_value = self.min_range
        else:
            ucb_bonuses = [confidence_radius*np.sqrt(np.log((self.global_time_step+1.0)/self.delta)/(count + .0000000001)) for count in self.counts ]
            ucb_arm_values = [min(self.mean_estimators[i] + ucb_bonuses[i], self.max_range) for i in range(self.num_arms)]
            #ucb_arm_index = np.argmax(ucb_arm_values)
            ucb_arm_values = np.array(ucb_arm_values)
            lcb_arm_values = [max(self.mean_estimators[i] - ucb_bonuses[i], self.min_range) for i in range(self.num_arms)]

            if np.random.random() <= self.epsilon:
                ucb_arm_index = np.random.choice(range(self.num_arms))
            else:
                ucb_arm_index = np.random.choice(np.flatnonzero(ucb_arm_values == ucb_arm_values.max()))
            
            ucb_arm_value = ucb_arm_values[ucb_arm_index]
            lcb_arm_value = lcb_arm_values[ucb_arm_index]
        return ucb_arm_index


    def get_arm(self, parameter, arm_info = None):
        return self.get_ucb_arm(parameter, arm_info = arm_info)


class LUCBalgorithm:
    def __init__(self, dimension, max_dimension, burn_in = 1, min_range = -float("inf"), 
        max_range = float("inf"), lambda_reg = 1, delta = .1, using_subset_dimensions = True):
        self.dimension = dimension
        self.max_dimension = max_dimension
        self.covariance  = np.eye(dimension)*lambda_reg
        self.delta = delta

        self.burn_in = burn_in
        self.min_range = min_range
        self.max_range = max_range
        self.lambda_reg = lambda_reg
        self.X_y = np.zeros(dimension) ## This should be 
        self.theta_hat = np.random.multivariate_normal(np.zeros(self.dimension), np.eye(self.dimension))
        self.theta_hat = self.theta_hat/np.linalg.norm(self.theta_hat)

        self.using_subset_dimensions = using_subset_dimensions

    def update_arm_statistics(self, arm_vector, reward):
        chopped_arm_vector = arm_vector[:self.dimension]
        self.X_y += chopped_arm_vector*reward

        self.covariance += np.outer(chopped_arm_vector, chopped_arm_vector)
        #print("covariance ", self.covariance)

        #IPython.embed()
        evalues, evectors = np.linalg.eigh(self.covariance)

        inverse_covariance = (evectors * 1.0/evalues) @ evectors.T
        #print("inverse_covariance ", inverse_covariance)
        self.theta_hat = np.dot(inverse_covariance, self.X_y)




    def get_ucb_arm(self, confidence_radius, arm_info = ("sphere", None) ):
        evalues, evectors = np.linalg.eigh(self.covariance)
        #inverse_covariance = np.linalg.inv(self.covariance)
        #IPython.embed()
        #theta_tilde = np.random.multivariate_normal(self.theta_hat, confidence_radius*self.dimension*inverse_covariance)
        #evalues, evectors = np.linalg.eigh(inverse_covariance)
        sqrt_inv_cov = (evectors * np.sqrt(1.0/evalues)) @ evectors.T
        perturbation = np.random.multivariate_normal(np.zeros(self.dimension), np.eye(self.dimension))


        theta_tilde = self.theta_hat + confidence_radius*np.sqrt(self.dimension)*np.dot(sqrt_inv_cov,perturbation)
        #IPython.embed()

        if arm_info[0] == "sphere":
            chopped_arm = theta_tilde/np.linalg.norm(theta_tilde)
            arm = np.zeros(self.max_dimension)
            arm[:self.dimension] = chopped_arm
        elif arm_info[0] == "hypercube":
            chopped_arm =  (2*(theta_tilde > 0) - 1)*1.0
            arm = (2*(np.random.normal(0,1,self.max_dimension) >= 0) - 1)*1.0
            arm[:self.dimension] = chopped_arm
            arm *= 1.0/np.sqrt(self.max_dimension)
            #IPython.embed()
        elif arm_info[0] == "contextual":
            arm_index = np.argmax([np.dot(theta_tilde, context[:self.dimension]) for context in arm_info[1] ])
            arm = arm_info[1][arm_index] ### This will output a max_dim arm

        else:
            raise ValueError("Arm info type not recognized {}".format(arm_info[0]))

        # ucb_arm_value = np.dot(arm[:self.dimension], theta_tilde)
        # lcb_arm_value = np.dot(arm[:self.dimension], theta_tilde)
        return arm


    def get_arm(self, parameter, arm_info = None):
        return self.get_ucb_arm(parameter, arm_info = arm_info)



class EXP3:
    def __init__(self,num_arms,T=1000, anytime = False, discount_factor = .9, 
        eta_multiplier = 1, forced_exploration_factor = 0, max_imp_weighted = 100):
        #self.hyperparam_list = hyperparam_list
        self.num_arms = num_arms# len(self.hyperparam_list)
        self.base_probas = np.ones(self.num_arms)/self.num_arms
        self.importance_weighted_cum_rewards = np.zeros(self.num_arms)
        self.T = T
        self.counter = 0
        self.anytime = False
        self.forced_exploration_factor = forced_exploration_factor
        self.eta_multiplier = eta_multiplier
        self.discount_factor = discount_factor

        self.max_imp_weighted = max_imp_weighted

        if self.anytime:
            self.T = 1


    def get_arm(self, parameter = None, arm_info = None):
        sample_array = np.random.choice(range(self.num_arms), 1, p=self.base_probas)
        return sample_array[0]


    def get_distribution(self):
        return self.base_probas

    
    def update_arm_statistics(self, arm_idx, reward):
        self.importance_weighted_cum_rewards[arm_idx] *= self.discount_factor
        self.importance_weighted_cum_rewards[arm_idx] += reward/self.base_probas[arm_idx]
        

        eta = self.eta_multiplier*np.sqrt( np.log(self.num_arms)/(self.num_arms*self.T))
        
        #IPython.embed()

        exponentials = np.exp( np.clip(self.importance_weighted_cum_rewards*eta, a_max = self.max_imp_weighted, a_min = None) )
        
        normalization_factor = np.sum(exponentials)
        #IPython.embed()

        self.counter += 1
        exp_dist = exponentials/normalization_factor
        forced_exploration_prob = min( self.forced_exploration_factor/np.sqrt(self.T), 1)
        uniform_distribution = np.ones(self.num_arms)*1.0/self.num_arms
        self.base_probas = (1-forced_exploration_prob)*exp_dist + forced_exploration_prob*uniform_distribution
        if self.anytime:
            self.T += 1



