import numpy as np 
import gym
import frozen_lakes
from hw2 import dyna_q_learning, dyna_q_learning_modified, prioritized_sweeping
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=3)

def main(prob):

	env1 = gym.make('FrozenLakeLarge-v0').unwrapped
	env1.seed(0)
	env2 = gym.make('FrozenLakeLargeShiftedIce-v0').unwrapped
	env2.seed(0)

	if prob == '1':

		# Dyna-Q 
		
		Q,reward_dyna= dyna_q_learning(env1, num_episodes = 400, num_planning = 50, epsilon = 0.02,\
				alpha = 0.1, gamma = 0.99)
		step_dyna = range( len( reward_dyna ))
		plt.plot(step_dyna, reward_dyna, 'r',label="Dyna-Q")

		# Dyna-Q+
		print("Dyan-Q policy:", Q)

		Q_plus,reward_dyna_plus= dyna_q_learning(env1, num_episodes = 400, num_planning = 50, \
				epsilon = 0.02,	alpha = 0.1, gamma = 0.99, kappa = 0.0001)
		step_dyna_plus = range( len( reward_dyna_plus ))
		plt.plot(step_dyna_plus, reward_dyna_plus, 'b',label="Dyna-Q+")

		# common plot functions
		print("Dyan-Q+ policy:", Q_plus)
		plt.legend(bbox_to_anchor=(0.80, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Time steps')
		plt.ylabel('Cummulative Rewards')
		plt.show()

	elif prob == '2':
		
		# Dyna-Q 
		multi_rewards_dyna = []
		for _ in range(10): # averaging over 10 agents
			np_seed = np.random.randint(0,100)
			env_seed = np.random.randint(0,100)
			np.random.seed(np_seed)
			env1.seed(env_seed)
			env2.seed(env_seed)

			_,reward_dyna= dyna_q_learning_modified(env1, env2, num_episodes = 400, num_planning = 20, epsilon = 0.05,\
				alpha = 0.1, gamma = 0.95)
			multi_rewards_dyna.append( reward_dyna )

		avg_rewards_dyna = np.mean(multi_rewards_dyna, axis=0)	
		step_dyna = range( len( avg_rewards_dyna ))
		plt.plot(step_dyna, avg_rewards_dyna, 'r',label="Dyna-Q")

		# Dyna-Q+
		# print ("staring Dyna_Q +")
		multi_rewards_dyna_plus = []
		for _ in range(10): # averaging over 10 agents
			np_seed = np.random.randint(0,100)
			env_seed = np.random.randint(0,100)
			np.random.seed(np_seed)
			env1.seed(env_seed)
			env2.seed(env_seed)

			_,reward_dyna_plus= dyna_q_learning_modified(env1, env2, num_episodes = 400, num_planning = 20, \
					epsilon = 0.05,	alpha = 0.1, gamma = 0.95, kappa = 0.0001)
			multi_rewards_dyna_plus.append( reward_dyna_plus )
		
		avg_rewards_dyna_plus = np.mean(multi_rewards_dyna_plus, axis=0)	

		step_dyna_plus = range( len( avg_rewards_dyna_plus ))
		plt.plot(step_dyna_plus, avg_rewards_dyna_plus, 'b',label="Dyna-Q+")

		# common plot functions

		plt.legend(bbox_to_anchor=(0.80, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Time steps')
		plt.ylabel('Cummulative Rewards')
		plt.show()

	elif prob == '3':
	
		policy, reward  = prioritized_sweeping( env1, num_episodes = 1000, num_planning = 20, epsilon = 0.02,\
				alpha = 0.1, theta=0.5, gamma = 0.99)
		steps = range(len(reward))
	
		plt.plot(steps, reward, 'r',label="Prioritized Sweeping")
		plt.legend(bbox_to_anchor=(0.70, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Time steps')
		plt.ylabel('Cummulative Rewards')
		plt.show()

		print policy	



if __name__ == "__main__":
    if len(sys.argv)==1:
        prob=1
    else:
    	prob=sys.argv[1]

    main(prob)