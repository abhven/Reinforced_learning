import gym
import numpy as np
import matplotlib.pyplot as plt
from hw1 import value_iteration

env=gym.make("FrozenLake-v0").unwrapped
env.reset()
policy=np.ones([env.nS,env.nA])/env.nA

policy = value_iteration(env)
print policy

# policy_compact=[ np.where(r==1)[0][0] for r in policy ]
# print np.reshape(policy_compact, (4,4))
