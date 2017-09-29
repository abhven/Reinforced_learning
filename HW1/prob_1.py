import numpy as np
import matplotlib.pyplot as plt
from hw1 import learn_bandit_rewards


def generate_4_bandits(num_of_bandits):
	def bandit_1():
		return 8

	def bandit_2():
		p=np.random.uniform()
		if p < 0.12:
			return 100
		else:
			return 0
	def bandit_3():
		p=np.random.uniform()
		return (p*45 - 10)

	def bandit_4():
		p=np.random.uniform()
		if p > (2.0/3.0):
			return 0
		elif p > (1.0/3.0):
			return 20
		else:
			reward_list=range(8,19)
			return reward_list[np.random.randint(0,len(reward_list))]			

	return [bandit_1, bandit_2, bandit_3, bandit_4] 

np.random.seed([42])
bandits=generate_4_bandits(4)

episodes=100

Q=np.empty((3,episodes))

_,Q[0]=learn_bandit_rewards(bandits,0.3,episodes)
_,Q[1]=learn_bandit_rewards(bandits,0.1,episodes)
_,Q[2]=learn_bandit_rewards(bandits,0.03,episodes)
steps=range(episodes)

plt.plot(steps, Q[0], 'r',label="0.3")
plt.plot(steps, Q[1], 'b',label="0.1")
plt.plot(steps, Q[2], 'g',label="0.03")
plt.legend(bbox_to_anchor=(0.80, 1), loc=2, borderaxespad=0.)
plt.xlabel('Number of episode')
plt.ylabel('Rewards')
print Q[0][99],Q[1][99],Q[2][99]
plt.show()
