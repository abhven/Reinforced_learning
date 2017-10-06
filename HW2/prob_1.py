import numpy as np 
import gym
import frozen_lakes
from hw2 import dyna_q_learning, prioritized_sweeping
import matplotlib.pyplot as plt

from gym.envs.registration import register

# register(id='FrozenLakeLargeShiftedIce-v0',entry_point='frozen_lakes_new.lake_envs:FrozenLakeLargeShiftedIceEnv',)
np.set_printoptions(precision=3)

prob_1 = True
prob_2 = True
prob_3 = False

if prob_1:
	# env = gym.make('FrozenLakeLargeShiftedIce-v0')
	# env = gym.make('FrozenLakeLarge-v0').unwrapped
	env=gym.make("FrozenLake-v0").unwrapped
	env.seed(0)
	# env.render()
	if False: #Dyna-Q
		Q,reward_dyna= dyna_q_learning(env)
		step_dyna = range( len( reward_dyna ))
		plt.plot(step_dyna, reward_dyna, 'r',label="Dyna-Q")

	if True: #Dyna-Q+
		Q_plus,reward_dyna_plus= dyna_q_learning(env, kappa = 0.01)
		step_dyna_plus = range( len( reward_dyna_plus ))
		plt.plot(step_dyna_plus, reward_dyna_plus, 'b',label="Dyna-Q+")

	plt.legend(bbox_to_anchor=(0.80, 1), loc=2, borderaxespad=0.)
	plt.xlabel('Time steps')
	plt.ylabel('Cummulative Rewards')
	plt.show()

if prob_2:
	pass

if prob_3:
	env = gym.make('FrozenLakeLargeShiftedIce-v0')
	# env = gym.make('FrozenLakeLarge-v0')
	env.seed(0)
	policy, reward  = prioritized_sweeping( env)
	steps = range(len(reward))
	plt.plot(steps, reward, 'r',label="Prioritized Sweeping")
	plt.xlabel('Time steps')
	plt.ylabel('Cummulative Rewards')
	plt.show()

	print policy	

