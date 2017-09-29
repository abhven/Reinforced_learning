import numpy as np

def learn_bandit_rewards(bandits, epsilon, num_episodes):

	num_of_bandits=len(bandits)
	Q_max_all=np.zeros(num_episodes)
	Q=np.zeros(num_of_bandits)
	N=np.zeros(num_of_bandits)
	# np.random.seed	(42)
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

def policy_evaluation2(policy, env, gamma=0.95, theta= 0.0001):
	V=np.zeros(env.nS)
	while 1:
		delta=0
		for state in range(env.nS):
			v=V[state]
			V[state]=0
			action=np.argmax(policy[state])
			transitions = env.P[state][action]
			for transition in transitions:
				prob,next_state,reward,_=transition
				V[state]+=prob*(reward + gamma* V[next_state])	

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
	return policy 

def policy_iteration(env, gamma = 0.95):

	V = np.zeros( env.nS )
	policy = np.zeros([env.nS, env.nA ])
	for i in range(env.nS):
		policy[i][np.random.randint(0, env.nA)]=1
	# policy=np.ones([env.nS,env.nA])/env.nA

	policy_stable = False

	while not policy_stable:

		V = policy_evaluation(policy , env, theta=0.0000001)

		policy, policy_stable = policy_improvement(policy, env, V, gamma)

	return policy



def policy_improvement(policy, env, V, gamma = 0.95):
	policy_stable=True
	for state in range(env.nS):
		old_action=np.copy(policy[state])
		# print old_action
		for action in range(env.nA):
			action_reward=0
			transitions=env.P[state][action]
			for transition in transitions:
				prob,next_state,reward,_=transition
				action_reward+=prob*(reward + gamma* V[next_state])
			if action==0:
				max_reward=action_reward
				policy[state][0]=1
				for i in range(1,env.nA):
					policy[state][i]=0	

			if action_reward > max_reward:
				max_reward=action_reward
				for i in range(env.nA):
					if i == action:
						policy[state][i]=1
					else:
						policy[state][i]=0
		# print policy[state], '\n'

		if not min(old_action==policy[state]):
			policy_stable=False

	return policy,policy_stable

def monte_carlo(env, gamma=0.95, epsilon = 0.1, num_episodes = 5000):
	'''
	implementing first visit monte carlo method
	'''
	np.random.seed(42)
	Q=np.zeros((env.nS,env.nA))
	state_action_returns=np.zeros((env.nS,env.nA))
	state_action_count=np.zeros((env.nS,env.nA))

	policy=np.ones([env.nS,env.nA])*(epsilon/env.nA)
	for i in range(env.nS):
		policy[i][np.random.randint(0, env.nA)]=1 - epsilon + epsilon / env.nA

	for episode in range(num_episodes):
		state=env.reset()
		done = False
		episode_info=[]
		episode_states=np.zeros((env.nS))
		# episode_state_action = np.zeros((env.nS , env.nA))

		while not done:
		#'''PART A: Generating a episode'''	
			action=np.random.choice(env.nA, p=policy[state])
			new_state, reward, done, _ = env.step(action=action)
			episode_states[state]=1
			episode_info.append([state,action,reward])
			state=new_state
		#'''PART B'''
		for count, info in enumerate(episode_info):
			for i in range(count+1):
				if (info[0]== episode_info[i][0]) and (info[1] == episode_info[i][1]):
					first_occurance = i
					G=0
					for time_step, j in enumerate(range(i+1,len(episode_info))):
						G+=episode_info[j][2]*(gamma**time_step)
					state_action_count[info[0],info[1]]+=1
					state_action_returns[info[0],info[1]]+=G
					Q[info[0],info[1]]=state_action_returns[info[0],info[1]]/state_action_count[info[0],info[1]]
					break		

			# if not episode_state_action[info[0],info[0]]:
			# 	episode_state_action[info[0],info[0]]=1

		#'''PART C'''
		for i in range(env.nS):		
			
			if episode_states[i]==1:
				best_action=np.argmax(Q[i])
				# best_action=np.random.choice(np.argwhere(Q[i] == np.max(Q[i])).flatten())
			
				for a in range(env.nA):
					if a == best_action:
						policy[i][a]= 1.0- epsilon + epsilon/env.nA
					else:
						policy[i][a]= epsilon/env.nA

		# for info in episode_info:		
			
		# 	best_action=np.random.choice(np.argwhere(Q[info[0]] == np.max(Q[info[0]])).flatten())
			
		# 	for a in range(env.nA):
		# 		if a == best_action:
		# 			policy[info[0]][a]= 1.0- epsilon + epsilon/env.nA
		# 		else:
		# 			policy[info[0]][a]= epsilon/env.nA


		# for s in range(env.nS):		
		# 	if episode_states[s]:
		# 		best_action=np.argmax(Q[s])
		# 	for a in range(env.nA):
		# 		if a == best_action:
		# 			policy[s][a]= 1.0- epsilon + epsilon/env.nA
		# 		else:
		# 			policy[s][a]= epsilon/env.nA

	optimal_policy=np.zeros((env.nS,env.nA))
	for i in range(env.nS):
		optimal_policy[i][np.argmax(Q[i])]=1
		
	return optimal_policy, Q



def q_learning(env, alpha=0.5, gamma= 0.95, epsilon= 0.1, num_episodes=500):

	Q=np.zeros((env.nS,env.nA))
	for episode in range(num_episodes):
		state=env.reset()
		while True:
			p=np.random.uniform()
			if p >= epsilon:
				action=np.random.choice(np.argwhere(Q[state] == np.max(Q[state])).flatten())

			else:
				action=np.random.randint(0,env.nA)

			new_state, reward, done, _ =env.step(action=action)
			Q[state][action] += alpha * (reward + gamma* np.max(Q[new_state]) - Q[state][action])
			state=new_state

			if done:
				break
	return Q




