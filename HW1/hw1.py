import numpy as np

def learn_bandit_rewards(bandits, epsilon, num_episodes):

	num_of_bandits=len(bandits)
	Q_max_all=np.zeros(num_episodes)
	Q=np.zeros(num_of_bandits)
	N=np.zeros(num_of_bandits)
	np.random.seed([42])
	for i in range(num_episodes):
		action_prob=np.random.uniform()
		if action_prob >=epsilon:
			A_list=np.argwhere(Q == np.max(Q)).flatten() #taking all the Q values that are of the max value
			A=A_list[np.random.randint(0,len(A_list))] #random tie breaker
		else:
			A=np.random.randint(0,num_of_bandits)

		R=bandits[A]()
		N[A]+=1
		Q[A]+=(1.0/N[A])*(R-Q[A])
		Q_max_all[i]=np.max(Q)

	return Q,Q_max_all

def policy_evaluation(policy, env, gamma=0.95, theta= 0.0001):
	V=np.zeros(env.nS)
	while 1:
		delta=0
		for state in range(env.nS):
			v=V[state]
			V[state]=0
			for action in range(env.nA):
				transitions = env.P[state][action]
				action_reward=0
				for transition in transitions:
					prob,next_state,reward,_=transition
					action_reward+=prob*(reward + gamma* V[next_state])	
				V[state]+=policy[state,action]*action_reward
			delta = max(delta,abs(v-V[state]))
		if delta < theta:
			break

	return V

def value_iteration(env, gamma=0.95, theta=0.0001):
	V=np.zeros(env.nS)
	while 1:
		delta=0
		for state in range(env.nS):
			v=V[state]
			for action in range(env.nA):
				action_reward=0
				transitions = env.P[state][action]
				for transition in transitions:
					prob,next_state,reward,_=transition
					action_reward+=prob*(reward + gamma* V[next_state])
				if action == 0:
					V[state]=action_reward

				if action_reward > V[state]:
					V[state]=action_reward
			delta = max(delta,abs(v-V[state]))
		if delta < theta:
			break
	
	pi=np.zeros(env.nS,dtype='int')
	
	for state in range(env.nS):

		for action in range (env.nA):
			action_reward=0
			transitions = env.P[state][action]
			for transition in transitions:
				prob,next_state,reward,_=transition
				action_reward+=prob*(reward + gamma* V[next_state])
			if action == 0:
				max_reward=action_reward

			if action_reward > max_reward:
				max_reward=action_reward
				pi[state]=action
	policy=np.zeros((env.nS,env.nA))
	policy[np.arange(env.nS),pi]=1
	return policy # need to add a method to return the probablity of returning state x action probability policy
	# i think the rationale is that there will be multiple argmax and you split the probability to account for this 

def policy_iteration(env, gamma = 0.95):
	V = np.zeros( env.nS )
	policy = np.ones([env.nS, env.nA ]) / env.nA
	policy_stable = False
	while not policy_stable:
		V = policy_iteration(policy,env)
		policy, policy_stable = policy_improvement(policy, env, gamma)

	return policy, V



def policy_improvement(policy, env, gamma = 0.95):
	policy_stable=True
	for state in range(env.nS):
		old_action=policy[state]
		for action in range(env.nA):
			action_reward=0
			transitions=env.P[state][action]
			for transition in transitions:
				prob,next_state,reward,_=transition
				action_reward+=prob*(reward + gamma* V[next_state])
			if action==0:
				max_reward=action_reward
				policy[state][0]=1
				policy[state][1:env.nA]=0

			if action_reward > max_reward:
				max_reward=action_reward
				for i in range(env.nA):
					if i == action:
						policy[state][i]=1
					else:
						policy[state][i]=0

		if not min(old_action==policy[state]):
			policy_stable=False

	return policy






