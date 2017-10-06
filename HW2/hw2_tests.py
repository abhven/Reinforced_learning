import numpy as np 
import gym
import frozen_lakes
from hw2 import dyna_q_learning, prioritized_sweeping
import matplotlib.pyplot as plt
import sys

def main(prob):

	env1 = gym.make('FrozenLakeLarge-v0').unwrapped
	env1.seed(0)
	env2 = gym.make('FrozenLakeLargeShiftedIce-v0').unwrapped
	env2.seed(0)

	if prob == '1':

		# Dyna-Q 
		
		Q,reward_dyna= dyna_q_learning(env1)
		step_dyna = range( len( reward_dyna ))
		plt.plot(step_dyna, reward_dyna, 'r',label="Dyna-Q")

		# Dyna-Q+

		Q_plus,reward_dyna_plus= dyna_q_learning(env1, kappa = 0.0001)
		step_dyna_plus = range( len( reward_dyna_plus ))
		plt.plot(step_dyna_plus, reward_dyna_plus, 'b',label="Dyna-Q+")

		# common plot functions

		plt.legend(bbox_to_anchor=(0.80, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Time steps')
		plt.ylabel('Cummulative Rewards')
		plt.show()

	elif prob == '2':
		pass

	elif prob == '3':
		pass


if __name__ == "__main__":
    if len(sys.argv)==1:
        prob=1
    else:
    	prob=sys.argv[1]

    main(prob)