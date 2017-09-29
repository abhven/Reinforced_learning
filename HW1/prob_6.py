import gym
import numpy as np
import matplotlib.pyplot as plt
from hw1 import q_learning

env=gym.make("FrozenLake-v0").unwrapped
np.random.seed(42)

Q = q_learning(env)

print Q
