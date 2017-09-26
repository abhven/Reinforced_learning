import gym
import numpy as np
import matplotlib.pyplot as plt
from hw1 import policy_evaluation

env=gym.make("FrozenLake-v0").unwrapped
env.reset()
policy=np.ones([env.nS,env.nA])/env.nA
np.random.seed(42)
V=policy_evaluation(policy,env)
print np.reshape(V,(4,4))