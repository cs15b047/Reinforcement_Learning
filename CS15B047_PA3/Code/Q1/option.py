import gym
from room_world import FourRooms
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product

env = gym.make('room_world-v0')

def give_coord(room, coord):
	x,y = 0,0
	if room == 0:
		x=0;y=0
	elif room == 1:
		x=0;y=6
	elif room == 2:
		x=7;y=6
	elif room == 3:
		x=6;y=0

	x,y = x+ coord[0], y+ coord[1]

	return [x,y]

# Beta
def is_stop_state(room_pos, room, option):
	#Primitive action
	if option < 4:
		return True

	# Stop position at every hallway position 
	for i in range(4):
		if room_pos == env.hallway_coords[i]:
			return True

	#Out of room
	if ((option == 4 or option == 5) and (room != 0)) or ((option == 6 or option == 7) and (room != 1)) or ((option == 8 or option == 9) and (room != 2)) or ((option == 10 or option == 11) and (room != 3)) :
		return True

	return False

# I
def is_option_allowed(room_pos, room, option):
	#Primitive action
	if option < 4:
		return True
	
	# If agent is in hallway
	if room == 0 and room_pos == env.hallway_coords[0]:
		if option == 4 or option == 7:
			return True
		return False
	elif room == 1 and room_pos == env.hallway_coords[1]:
		if option == 6 or option == 9:
			return True
		return False
	elif room == 2 and room_pos == env.hallway_coords[2]:
		if option == 8 or option == 11:
			return True
		return False
	elif room == 3 and room_pos == env.hallway_coords[3]:
		if option == 5 or option == 10:
			return True
		return False

	# Agent is inside a room
	if ((option == 4 or option == 5) and (room == 0)) or ((option == 6 or option == 7) and (room == 1)) or ((option == 8 or option == 9) and (room == 2)) or ((option == 10 or option == 11) and (room == 3)) :
		return True
	return False

#Assume deterministic
# ****Policy types******* Diffeent types of optimal policies (2 simplest types considered)
def option_policy(room_pos, room, option,policy_type):
	#Primitive action
	if option < 4:
		return option

	agent_pos = give_coord(room, room_pos)
	# print agent_pos

	# Going into upper/lower room
	if option == 4 or option == 11 or option == 7 or option == 8:		
		if option == 4 or option == 11:
			hall_pos = give_coord(3,env.hallway_coords[3])		
		if option == 7 or option == 8:
			hall_pos = give_coord(1,env.hallway_coords[1])

		if policy_type == 0:
			if agent_pos[1] > hall_pos[1]:
				return 3
			elif agent_pos[1] < hall_pos[1]:
				return 1
			
			if agent_pos[0] < hall_pos[0]:
				return 2
			elif agent_pos[0] > hall_pos[0]:
				return 0

		if policy_type == 1:
			if agent_pos[0] < hall_pos[0] - 1:
				return 2
			elif agent_pos[0] > hall_pos[0] + 1:
				return 0

			if agent_pos[1] > hall_pos[1]:
				return 3
			elif agent_pos[1] < hall_pos[1]:
				return 1
			elif agent_pos[1] == hall_pos[1] :
				if agent_pos[0] == hall_pos[0] - 1:
					return 2
				elif agent_pos[0] == hall_pos[0] + 1:
					return 0


	# Going into sideways room
	if option == 5 or option == 6 or option == 9 or option == 10:
		if option == 5 or option == 6:
			hall_pos = give_coord(0,env.hallway_coords[0])
		if option == 9 or option == 10:
			hall_pos = give_coord(2,env.hallway_coords[2])
	
		if policy_type == 0:		
			if agent_pos[0] > hall_pos[0]:
				return 0
			elif agent_pos[0] < hall_pos[0]:
				return 2
			elif agent_pos[1] < hall_pos[1]:
				return 1
			elif agent_pos[1] > hall_pos[1]:
				return 3

		if policy_type == 1:
			if agent_pos[1] < hall_pos[1] - 1:
				return 1
			elif agent_pos[1] > hall_pos[1] + 1:
				return 3

			if agent_pos[0] > hall_pos[0]:
				return 0
			elif agent_pos[0] < hall_pos[0]:
				return 2
			elif agent_pos[0] == hall_pos[0]:
				if agent_pos[1] == hall_pos[1] - 1:
					return 1
				elif agent_pos[1] == hall_pos[1] + 1:
					return 3	


# print give_coord(0,env.hallway_coords[0]),give_coord(1,env.hallway_coords[1]),give_coord(2,env.hallway_coords[2]),give_coord(3,env.hallway_coords[3])
