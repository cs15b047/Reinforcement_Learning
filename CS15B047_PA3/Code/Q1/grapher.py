import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from option import give_coord, is_option_allowed, is_stop_state, option_policy
import pickle

def visualize(Q):
	V_show = np.zeros((12,12))
	for i in range((Q.shape)[0]-2):	
		[r,coord] = env.decode(i)
		[x,y] = give_coord(r,coord)
		a = [j for j in options if is_option_allowed(coord,r,j)]
		q = Q[[i]*len(a),a]
		V_show[ 11-x, y ] = np.max(q)


	V_show[8,0] = 10
	V_show[0,8] = 20

	cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','black','red'],100)
	img = plt.imshow(V_show,interpolation='nearest',cmap = cmap2,origin='lower')
	plt.colorbar(img,cmap=cmap2)
	plt.savefig("Values_"+method+"_G"+str(goal))
	# pickle.dump(V_show, open( "values_"+str(algo)+"_"+str(goal), "wb" ))
	plt.show()

#Comparison
for goal in [1,2]:
	for algo in [1,2]:
		data = pickle.load( open( "data_"+str(algo)+"_"+str(goal), "rb") )
		
		method = "Intra-option"
		if algo == 1:
			method = "SMDP"		

		plt.plot(data[0],np.log10(data[1]),label="algo:"+method+" goal:G"+str(goal))

	plt.legend(loc='best')
	plt.ylabel("log10 (Steps per episode)")
	plt.xlabel("log10 (Number per episode)")
	plt.savefig("Comaprison_G"+str(goal))
	plt.clf()

#Val Fn Visualzn
for goal in [1,2]:
	for algo in [1,2]:
		data = pickle.load( open( "values_"+str(algo)+"_"+str(goal), "rb") )
		method = "Intra-option"
		if algo == 1:
			method = "SMDP"

		cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','black','red'],100)
		img = plt.imshow(data,interpolation='nearest',cmap = cmap2,origin='lower')
		plt.colorbar(img,cmap=cmap2)
		plt.savefig("Values_"+method+"_G"+str(goal))
		plt.clf()