#!/usr/bin/env python

import click
import numpy as np
import gym
import time
import copy

import pickle
from itertools import product

#Gives gradient for 1 trajectory
def policy_gradient(theta, state_traj, action_traj, adv_func):
    gradient = np.zeros(theta.shape)    

    for (s,a,ret) in zip(state_traj, action_traj,adv_func):
        mu = np.matmul(theta, np.reshape(s,(s.size,1)))
        
        a = np.reshape(a,(a.size,1))
        s = np.reshape(s,(s.size,1))
        
        arg1 = np.subtract(a,mu)
        
        arg2 = np.transpose(s)        
        
        grad = np.matmul(arg1, arg2)        
        
        update = np.multiply(grad, ret)
        
        gradient += update

    # print gradient

    return gradient

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

    # action = np.clip(action,-0.025,0.025)

    #TODO : Actions have norm of maximum 0.025 , not always
    if np.linalg.norm(action) > 0.025:        
        action = 0.025 * (action / np.linalg.norm(action))

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

    env.seed(int(time.time()) + 1)
    
    #Good policy
    # theta[0][0],theta[1][0],theta[0][2],theta[1][1] = 0,0,0,0
    # theta[0][1], theta[1][2] = -10,-10

    episode_length = 40
    
    max_itr = 1000
    itr = 0

    Gamma_list = [1.0]
    Batch_size_list = [500]
    Learning_rate_list = [0.1]

    for (gamma,batch_size,learning_rate) in product(Gamma_list,Batch_size_list,Learning_rate_list):
        print gamma,batch_size,learning_rate
        
        # Initialize parameters
        # theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
        
        # Working with 0 initialization
        theta = np.zeros((action_dim, obs_dim + 1))    

        Returns = np.zeros(max_itr)
        Distance = np.zeros(max_itr)

        itr = 0
        
        while itr < max_itr:
            # if itr == max_itr/5:
            #     batch_size = 50

            samples = 0
            batch_grad = np.zeros(theta.shape)
            batch_avg_return = 0
            avg_final_dist = 0
            batch_avg = np.zeros(episode_length)

            #Random init for baseline assuming each episode surely lasts for 40 steps
            # Baseline is calulated here as avg of return at a particular timestep        
            
            baseline = np.arange(-episode_length*1.0,0,1)    
            # baseline = np.zeros(episode_length)
            while samples < batch_size:

                ob = env.reset()
                done = False
                # Only render the first trajectory
                # Collect a new trajectory
                rewards = []
                trajectory_states = []
                trajectory_actions = []
                disc_returns = []

                #Get trajectory
                while not done:
                    ob_1 = include_bias(ob)   
                    trajectory_states += [ob_1]
                    
                    action = get_action(theta, ob, rng=rng)
                    # print action
                    
                    trajectory_actions += [action]
                    next_ob, rew, done, _ = env.step(action)
                    ob = next_ob
                    if samples == 0:                    
                        env.render()
                    rewards += [rew]

                # print len(trajectory_states)
                final_dist = np.linalg.norm(trajectory_states[-1][1:]) # final_dist = dist of last state from origin

                # Calc returns
                disc_returns = [0]
                for rw in rewards[::-1] :
                    disc_returns =  [rw + gamma*disc_returns[0]] + disc_returns

                disc_returns = disc_returns[:-1]

                # Calc grad for 1 trajectory
                trajectory_grad = policy_gradient(theta, trajectory_states, trajectory_actions, disc_returns - baseline)

                # Baseline is avg of return at a particular timestep
                baseline += (disc_returns - baseline)/(samples + 1) # now changing always                

                # Accumulate grad for 1 batch
                batch_grad += trajectory_grad

                # print("Episode reward: %.2f" % np.sum(rewards))
                batch_avg_return += disc_returns[0]
                samples += 1
                avg_final_dist += (final_dist - avg_final_dist)/samples

                batch_avg += (disc_returns - batch_avg) / (samples*1.0)
            

            baseline = copy.deepcopy(batch_avg_return)

            # Average Batch gradient
            batch_grad = batch_grad / batch_size

            # Normalize grad
            batch_grad = batch_grad / (np.linalg.norm(batch_grad) + 1e-8)

            #Update theta
            theta += learning_rate * batch_grad

            print "Itr : "+str(itr)+" Avg return: "+str(batch_avg_return/batch_size) + "Avg Final Dist: "+ str(avg_final_dist)+ " Theta: "+str(theta)+" Grad : "+str(batch_grad)
            
            Returns[itr] = batch_avg_return/batch_size
            Distance[itr] = avg_final_dist

            itr += 1

        print "Training Done"


        # Generate trajectories with learned policy
        itr = 0
        num_trajectories = 100
        


        Trajectories = []
        trajectory_states = []
        trajectory_actions = []
        disc_returns = []
        Returns = []
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

            itr += 1

        np.savetxt(env_id+"/Theta_"+str(gamma)+"_"+str(batch_size)+"_"+str(learning_rate),theta)
        np.savetxt(env_id+"/Returns_"+str(gamma)+"_"+str(batch_size)+"_"+str(learning_rate),Returns)
        np.savetxt(env_id+"/Distance_"+str(gamma)+"_"+str(batch_size)+"_"+str(learning_rate),Distance)
        pickle.dump(Trajectories, open(env_id+"/Trajectories_"+str(gamma)+"_"+str(batch_size)+"_"+str(learning_rate) , "wb"))
        


if __name__ == "__main__":
    main()
