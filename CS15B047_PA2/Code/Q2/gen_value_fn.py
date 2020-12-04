#!/usr/bin/env python

import click
import numpy as np
import gym
import time

import pickle
from itertools import product

def include_bias(ob):
    return np.concatenate((np.array([1]),ob))

def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    action = rng.normal(loc=mean, scale=1.)

    #TODO : Actions have norm of maximum 0.025 , not always
    if np.linalg.norm(action) > 0.025:        
        action = 0.025 * (action / np.linalg.norm(action))
    
    return action

def vishamC_get_action(theta, ob, rng = np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)

    action = rng.normal(loc=mean, scale=1.)

    action = np.clip(action,-0.025,0.025)

    # #TODO : Actions have norm of maximum 0.025 , not always
    # if np.linalg.norm(action) > 0.025:        
    #     action = 0.025 * (action / np.linalg.norm(action))

    return action

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(int(time.time()))

    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id == 'vishamC':
        from rlpa2 import vishamC
        env = gym.make('vishamC-v0')
        get_action = vishamC_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported environment: must be 'chakra' ")

    env.seed(int(time.time()))
    
    #Good policy
    # theta[0][0],theta[1][0],theta[0][2],theta[1][1] = 0,0,0,0
    # theta[0][1], theta[1][2] = -10,-10

    episode_length = 40
    
    max_itr = 1000
    itr = 0

    Gamma_list = [1.0]
    Batch_size_list = [250]
    Learning_rate_list = [0.01]

    for (gamma,batch_size,learning_rate) in product(Gamma_list,Batch_size_list,Learning_rate_list):
        print gamma,batch_size,learning_rate        
    
        # Generate trajectories with learned policy
        itr = 0
        num_trajectories = 10000
        
        theta = np.loadtxt("chakra/Theta_1.0_50_0.1")

        Trajectories = []
        trajectory_states = []
        trajectory_actions = []
        disc_returns = []
        Returns = []
        rewards = []
        while itr < num_trajectories:
            ob = env.reset()
            done = False
            #Get trajectory
            trajectory_states = []
            trajectory_actions = []
            while not done:
                ob_1 = include_bias(ob)   
                trajectory_states += [ob_1]
                
                action = get_action(theta, ob, rng=rng)
                # print action
                
                trajectory_actions += [action]
                next_ob, rew, done, _ = env.step(action)
                ob = next_ob
                rewards += [rew]
            Trajectories += [trajectory_states]
            disc_returns = [0]
            for x in rewards[::-1]:
                disc_returns = [x + gamma* disc_returns[0]] + disc_returns
            Returns += [disc_returns[:-1]]
            disc_returns = []
            rewards = []

            itr += 1

        np.savetxt(env_id+"/Values_1.0_50_0.1",Returns)
        pickle.dump(Trajectories, open(env_id+"/Trajectory_1.0_50_0.1" , "wb"))
        


if __name__ == "__main__":
    main()
