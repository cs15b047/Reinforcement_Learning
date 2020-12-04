import gym
from room_world import FourRooms
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from option import give_coord, is_option_allowed, is_stop_state, option_policy

#0-UP 1-RIGHT 2-DOWN 3-LEFT 	Primitive options
#4,5-->0 6,7-->1 8,9-->2 10,11-->3 			Multistep options in corresponding rooms


def set_options(scene):
	if scene == 1:
		num_options = 4
		options = np.arange(4)
	elif scene == 2:
		num_options = 8
		options = [4,5,6,7,8,9,10,11]		
	elif scene == 3 :
		num_options = 12
		options = np.arange(12)

	return num_options,options 

def visualize(Q, scene):
	V_show = np.zeros((12,12))
	for i in range((Q.shape)[0]-2):	
		[r,coord] = env.decode(i)
		[x,y] = give_coord(r,coord)
		a = [j for j in options if is_option_allowed(coord,r,j)]
		q = Q[[i]*len(a),a]
		V_show[ 11-x, y ] = np.max(q)

	cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','black','red'],100)
	img = plt.imshow(V_show,interpolation='nearest',cmap = cmap2,origin='lower')
	plt.colorbar(img,cmap=cmap2)
	plt.savefig("Values_"+str(algo))
	pickle.dump(V_show, open( "values_"+str(algo)+"_"+str(goal), "wb" ))
	plt.show()

def policy_print(Q, scene):
	policy = (-1)*np.ones((12,12))
	V_show = np.zeros((12,12))	


	for s in range(env.n_states - 1):
		r,p = env.decode(s)
		x,y = give_coord(r,p)


		a = [i for i in options if is_option_allowed(p,r,i)]
		q = Q[ [s]*len(a), a]
		policy[x,y] = a[np.argmax(q)]
		V_show[x,y] = np.max(q)

	pickle.dump(policy, open( "policy_"+str(algo)+"_"+str(goal), "wb" ))
	print policy


np.random.seed(int(time.time()))

env = gym.make('room_world-v0')
env.seed()
env.reset()

# G1 or G2
goal = int(input())

# if goal == 1:
# 	env.goal = [1,[6,2]]
# elif goal == 2:
# 	env.goal = [2,[1,2]]


#1 -->SMDP Q learning
#2 --> Intra option Q-Learning
algo = int(input())


## Changing start state to middle of bottom left 
start_room = 3
sz = env.room_sizes[start_room]
env.start_state = env.encode([3,[2,2]])

alpha = 0.1
gamma = 0.9
epsilon = 0.1

max_episodes = 10000
runs = 30
scene_list = [3]

#Option policy type
option_policy_type = 0

#1 --> A, 2--> O , 3 --> A + O
for scene in scene_list:
	avg_episode_length_list = np.zeros(max_episodes)
	for run_count in range(runs):
		#============
		start_time = time.time()
		print "scene"+str(scene)+" Run"+str(run_count)
		#============

		#===============Init==============
		num_options, options = set_options(scene)
		Q = np.zeros((env.n_states + 1, 12))
		Q[env.terminal_state,] = env.terminal_reward
		done = False
		num_ep = 0 
		episode_length_list = []
		#=================================

		while num_ep < max_episodes:
			episode_length = 0
			while not done:
				# env._render()
	
				#======= S ==============
				curr_state = env.state
				start_of_option = env.state
				[room, pos_in_room] = env.decode(env.state)				
				#=============================================

				# print [room, pos_in_room]

				#================== O ===========================
				# Allowed options
				allowed = [op for op in options if is_option_allowed(pos_in_room, room, op)]
				
				opt = allowed[np.argmax(Q[[curr_state]*len(allowed) ,allowed])]
				if np.random.uniform() <= epsilon:
					# print "rendom action"
					opt = allowed[np.random.randint(len(allowed))]
				#==============================================

				# print allowed, Q[[curr_state]*len(allowed) ,allowed],np.argmax(Q[[curr_state]*len(allowed) ,allowed])
				# print opt

				#=======================SMDP Q Learning ===========================
				
				# =========Execute option===============
				cum_rew = 0
				tau = 0
				# print "Currnt option : "+str(opt)
				# enter even if start state is in termination states. exit in terminal state of whole MDP
				while not done and ((not is_stop_state(pos_in_room, room, opt)) or tau == 0):

					#State before action
					prev_state = env.state
					#Take step according to option policy
					action = option_policy(pos_in_room, room, opt, option_policy_type)
					curr_state, rew, done, ignore = env.step(action)

					# print "prev state and action :"+str(env.decode(prev_state))+" "+str(action)


					#================Intra-Option============================
					if algo == 2:
						# Use state before action and currently executing option

						possible_options = [op for op in options if (is_option_allowed(pos_in_room,room,op) and (option_policy(pos_in_room, room, op, option_policy_type) == action))]

						# possible_options.remove(opt) #Modiification

						# print action, possible_options

						for op in possible_options:
							target = 0 

							# Episode terminated --> so all q values at that state -> 0
							if done :
								# print "ep Done"
								target = 0

							# Option (for which we are updating) terminates at that state
							elif is_stop_state(env.decode(env.state)[1],env.decode(env.state)[0], op ):
								# print "Stop state for :"+str(env.decode(env.state))+" "+str(op)
								[r,p] = env.decode(env.state)
								valid = [x for x in options if (is_option_allowed(p,r,x))]
								# print valid							
								target = np.amax(Q[[env.state]*len(valid), valid])
								# print Q[[env.state]*len(valid), valid], target

							# Option (for which we are updating) is still on
							else:
								# print "Option "+str(op)+" valid"
								target = Q[env.state, op]

							Q[prev_state, op] += alpha*(rew + gamma * target - Q[prev_state, op])
					#===========================================================

					# reqd, as, if done, decode gives error
					if not done:
						room, pos_in_room = env.decode(env.state)
					
					cum_rew += pow(gamma, tau)*rew # Cum Reward is addition of discounted reward
					tau += 1
					episode_length += 1
				#=======================================================
				
				#s' --> curr_state,room,pos_in_room

				if algo == 1  :
					
					#================= Target =================
					# If at hallway, some options of both rooms available 
					allowed = [op for op in options if is_option_allowed(pos_in_room, room, op)]							
					max_val = np.max(Q[[curr_state]*len(allowed) ,allowed])
					#==================================
					
					#=========Make Q-Learning update=================
					Q[start_of_option, opt] += alpha*( cum_rew + pow(gamma, tau)*max_val - Q[start_of_option, opt] )
					# print "QLearning Update: "+str(env.decode(start_of_option))+" "+str(opt)+" "+str(Q[start_of_option, opt])
					#================================================
				#==================================================


				# print Q[start_of_option, opt]

			#=======After-episode resets==============
			episode_length_list += [episode_length]
			env.reset()
			done = False
			# print num_ep
			num_ep += 1
			#=========================================

		#============After-run averaging======================
		avg_episode_length_list += (episode_length_list - avg_episode_length_list)/(run_count + 1)
		#=====================================================

		print "Time: = "+str(time.time() - start_time)


	#============Postprocessing and Plotting=============
	policy_print(Q,scene)	
	visualize(Q,scene)
	#Every value is average of 10 values
	avg_step = 10
	x_axis = np.log10(np.arange(1,len(avg_episode_length_list)+1,1))
	avg_episode_length_list = np.mean(avg_episode_length_list.reshape(-1, 10), axis=1)
	x_axis = np.mean(x_axis.reshape(-1, avg_step), axis=1)

	pickle.dump( [x_axis,avg_episode_length_list], open( "data_"+str(algo)+"_"+str(goal), "wb" ) )

	plt.plot(x_axis, np.log10(avg_episode_length_list))
	#=====================================================

plt.show()