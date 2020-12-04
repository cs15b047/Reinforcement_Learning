import matplotlib.pyplot as plt
import pickle
import numpy as np


with open ('eps_greedy', 'rb') as fp:
    eps_greedy = pickle.load(fp)

with open ('softmax', 'rb') as fp:
    softmax = pickle.load(fp)

with open ('ucb1', 'rb') as fp:
    ucb1 = pickle.load(fp)

with open ('eps_greedy_10arms', 'rb') as fp:
    eps_greedy_1 = pickle.load(fp)

with open ('softmax_10arms', 'rb') as fp:
    softmax_1 = pickle.load(fp)

with open ('ucb1_10arms', 'rb') as fp:
    ucb1_1 = pickle.load(fp)

with open ('mea_q5', 'rb') as fp:
    mea_1000 = pickle.load(fp)

with open ('mea_q5_10', 'rb') as fp:
    mea_10 = pickle.load(fp)


figure = plt.figure()
figure2 = plt.figure()
figure3 = plt.figure()
figure4 = plt.figure()

plt1 = figure.add_subplot(211)
plt2 = figure.add_subplot(212)
plt3 = figure2.add_subplot(211)
plt4 = figure2.add_subplot(212)
plt5 = figure3.add_subplot(211)
plt6 = figure3.add_subplot(212)
plt7 = figure4.add_subplot(211)
plt8 = figure4.add_subplot(212)


#Assumes only 1 value of parameters for plotting graphs for each arm setting.

plt1.plot(eps_greedy[0][0],label="1000 arms")
plt1.plot(eps_greedy_1[0][0],label="10 arms")
plt1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt1.set_xlabel('Steps')
plt1.set_ylabel('Average Rewards')

plt2.plot(eps_greedy[1][0],label="1000 arms")
plt2.plot(eps_greedy_1[1][0],label="10 arms")
plt2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt2.set_xlabel('Steps')
plt2.set_ylabel('% Optimal Actions')

figure.suptitle('Comparing 10 vs 1000 arms Epsilon-greedy (epsilon = 0.1)')


plt3.plot(softmax[0][0],label="1000 arms")
plt3.plot(softmax_1[0][0],label="10 arms")
plt3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt3.set_xlabel('Steps')
plt3.set_ylabel('Average Rewards')

plt4.plot(softmax[1][0],label="1000 arms")
plt4.plot(softmax_1[1][0],label="10 arms")
plt4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt4.set_xlabel('Steps')
plt4.set_ylabel('% Optimal Actions')

figure2.suptitle('Comparing 10 vs 1000 arms Softmax (temp = 0.3)')


plt5.plot(ucb1[0][0],label="1000 arms")
plt5.plot(ucb1_1[0][0],label="10 arms")
plt5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt5.set_xlabel('Steps')
plt5.set_ylabel('Average Rewards')

plt6.plot(ucb1[1][0],label="1000 arms")
plt6.plot(ucb1_1[1][0],label="10 arms")
plt6.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt6.set_xlabel('Steps')
plt6.set_ylabel('% Optimal Actions')
figure3.suptitle('Comparing 10 vs 1000 arms UCB1 (c = 1)')


plt7.plot(mea_1000[4],label="1000 arms")
plt7.plot(mea_10[4],label="10 arms")
plt7.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt7.set_xlabel('Steps')
plt7.set_ylabel('Average Rewards')

plt8.plot(mea_1000[5],label="1000 arms")
plt8.plot(mea_10[5],label="10 arms")
plt8.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt8.set_xlabel('Steps')
plt8.set_ylabel('% Optimal Actions')
figure4.suptitle('Comparing 10 vs 1000 arms MEA ')

plt.show()