import gym
import numpy as np
import matplotlib.pyplot as plt
from hw1 import monte_carlo

env=gym.make("FrozenLake-v0").unwrapped
np.random.seed(42)

policy, Q = monte_carlo(env, epsilon=0.18, num_episodes = 10000)

policy_compact=[ np.where(r > 0.5)[0][0] for r in policy ]

print np.reshape(policy_compact,(4,4))
print Q
print policy
