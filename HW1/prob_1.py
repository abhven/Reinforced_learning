import numpy as np
import matplotlib.pyplot as plt
from hw1 import learn_bandit_rewards

def generate_bandits(num_of_bandits):
	np.random.seed(42)
	# Q_val=np.random.randn(num_of_bandits)
	Q_val=np.random.normal(size=num_of_bandits)
	bandits={}
	for i in range(num_of_bandits):
		def bandit():
			# return np.random.randn() + Q_val[i]
			return np.random.normal(loc = Q_val[i])
		bandits[i]=bandit
	print Q_val
	return bandits


bandits=generate_bandits(4)


episodes=400

Q=np.empty((3,episodes))

for i in range(episodes):
	np.random.seed(42)
	Q[0][i]=np.max(learn_bandit_rewards(bandits,0.3,i))
	np.random.seed(42)
	Q[1][i]=np.max(learn_bandit_rewards(bandits,0.1,i))
	np.random.seed(42)
	Q[2][i]=np.max(learn_bandit_rewards(bandits,0.03,i))

steps=range(episodes)
plt.plot(steps, Q[0], 'r', steps, Q[1], 'b', steps, Q[2], 'g')
print Q[0][99],Q[1][99],Q[2][99]
plt.show()
