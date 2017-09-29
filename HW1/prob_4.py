import gym
import numpy as np
import matplotlib.pyplot as plt
from hw1 import policy_iteration

env=gym.make("FrozenLake-v0").unwrapped
np.random.seed(42)

policy = policy_iteration(env)
print policy

# policy_compact=[ np.where(r==1)[0][0] for r in policy ]
# print np.reshape(policy_compact, (4,4))
