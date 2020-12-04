import pickle
import numpy as np

k=10
bandit_problem = [] 
num_problems = 2000

for i in range(num_problems):
	q = np.random.normal(size=k)
	bandit_problem += [q]

with open('testbed', 'wb') as fp:
    pickle.dump(bandit_problem, fp)