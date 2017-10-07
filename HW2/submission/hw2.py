import numpy as np 	
from Queue import PriorityQueue

def epsilon_soft(entries, epsilon):
	best=np.argmax(entries)
	num_entries=len(entries)
	prob=np.empty(num_entries)
	for a in range(num_entries):
		if a==best:
			prob[a] = 1 - epsilon + epsilon/num_entries
		else:
			prob[a] = epsilon/num_entries
	return prob

def epsilon_greedy(entries, epsilon):

	prob=np.random.uniform()
	if prob >= epsilon:
		# return np.random.choice(np.argwhere(entries == np.max(entries)).flatten())
		return np.argmax(entries)
	else:
		return np.random.choice(len(entries))


def dyna_q_learning(env, num_episodes = 30, num_planning = 50, epsilon = 0.1,\
		alpha = 0.1, gamma = 0.95, kappa= None):

	np.random.seed(42)
	Q = np.zeros((env.nS , env.nA))
	model = [ [ [] for i in range(env.nA) ] for j in range(env.nS) ]

	cumm_reward=[]
	time_state_action=np.zeros((env.nS,env.nA))
	time_step = -1.0 
	visited_state=np.zeros(env.nS)
	visited_state_action=np.zeros((env.nS,env.nA))
	if kappa == None:
		kappa = 0

	for _ in range(num_episodes):
		S = env.reset()
		done = False
		while not done:
			A = epsilon_greedy(Q[S], epsilon)
			time_state_action += 1 # adding one time step to all state action pairs
			time_step += 1
			# A=np.random.choice(env.nA, p=Q[S])
			visited_state[S] = 1
			visited_state_action[S][A] = 1
			time_state_action[S][A] = 0
			new_state, reward, done, _ = env.step(action=A)
			# if reward > 0:
			# 	print reward

			Q[S][A] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[S][A])
			model[S][A].append([reward, new_state]) # adding all the observed set of rewards and possible next state from observation
			if time_step==0:
				cumm_reward.append(reward)
			else:
				cumm_reward.append(cumm_reward[int(time_step-1)]+reward)
				# cumm_reward.append(reward)


			for _ in range(num_planning):
				p_S = np.random.choice(env.nS, p = (visited_state / np.sum(visited_state)))
				p_A = np.random.choice(env.nA, p = (visited_state_action[p_S] / np.sum(visited_state_action[p_S])))
				index = np.random.choice(len(model[p_S][p_A])) #choosing a random reaward and next state from all previous encounters 
				# print p_A
				p_R, p_Sp = model[p_S][p_A][index] 
				Q[p_S][p_A] += alpha * (p_R + kappa * np.sqrt( time_state_action[p_S][p_A]) + gamma * np.max( Q[p_Sp]) - Q[p_S][p_A])
			
			S=new_state
	policy = np.zeros((env.nS , env.nA))

	for i in range(env.nS):
		policy[i][np.argmax(Q[i])]=1
	# print time_state_action

	return policy, cumm_reward

def dyna_q_learning_modified(env1, env2, num_episodes = 30, num_planning = 50, epsilon = 0.1,\
		alpha = 0.1, gamma = 0.95, kappa= None, change_time=3000, end_time=6000):

	env=env1
	Q = np.zeros((env.nS , env.nA))
	model = [ [ [] for i in range(env.nA) ] for j in range(env.nS) ]

	cumm_reward=[]
	time_state_action=np.zeros((env.nS,env.nA))
	time_step = -1.0 
	visited_state=np.zeros(env.nS)
	visited_state_action=np.zeros((env.nS,env.nA))
	if kappa == None:
		kappa = 0

	for i in range(num_episodes):

		if time_step == end_time:
			break
		S = env.reset()
		done = False
		while not done:
			A = epsilon_greedy(Q[S], epsilon)
			time_state_action += 1 # adding one time step to all state action pairs
			time_step += 1
			# A=np.random.choice(env.nA, p=Q[S])
			visited_state[S] = 1
			visited_state_action[S][A] = 1
			time_state_action[S][A] = 0
			new_state, reward, done, _ = env.step(action=A)
			# if reward > 0:
			# 	print reward

			Q[S][A] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[S][A])
			model[S][A].append([reward, new_state]) # adding all the observed set of rewards and possible next state from observation
			if time_step==0:
				cumm_reward.append(reward)
			else:
				cumm_reward.append(cumm_reward[int(time_step-1)]+reward)
				# cumm_reward.append(reward)


			for _ in range(num_planning):
				p_S = np.random.choice(env.nS, p = (visited_state / np.sum(visited_state)))
				p_A = np.random.choice(env.nA, p = (visited_state_action[p_S] / np.sum(visited_state_action[p_S])))
				index = np.random.choice(len(model[p_S][p_A])) #choosing a random reaward and next state from all previous encounters 
				# print p_A
				p_R, p_Sp = model[p_S][p_A][index] 
				Q[p_S][p_A] += alpha * (p_R + kappa * np.sqrt( time_state_action[p_S][p_A]) + gamma * np.max( Q[p_Sp]) - Q[p_S][p_A])
			
			S=new_state
			if time_step == change_time:
				env=env2
				S=env.reset()
			if time_step == end_time:
				done = True
	policy = np.zeros((env.nS , env.nA))

	for i in range(env.nS):
		policy[i][np.argmax(Q[i])]=1
	# print time_state_action

	return policy, cumm_reward

def prioritized_sweeping( env, num_episodes = 30, num_planning = 50,\
	epsilon = 0.1, alpha = 0.1, theta = 0.1, gamma = 0.95):

	PQueue= PriorityQueue()
	np.random.seed(42)
	Q = np.zeros((env.nS , env.nA))
	model = [ [ [] for i in range(env.nA) ] for j in range(env.nS) ]
	path_to_state=[ [] for i in range(env.nS) ]
	queue_time_step = -1
	cumm_reward=[]
	tot_reward=0
	for _ in range(num_episodes):
		S = env.reset()
		done = False
		while not done:
			A = epsilon_greedy(Q[S], epsilon)
			
			new_state, reward, done, _ = env.step(action=A)
			model[S][A].append([reward, new_state]) # adding all the observed set of rewards and possible next state from observation
			priority = np.abs( reward + gamma * np.max(Q[new_state]) - Q[S][A] )
			path_to_state[new_state].append((S,A, reward))
			tot_reward += reward
			cumm_reward.append(tot_reward)
			# path_to_state[new_state]=
			if priority > theta:
				queue_time_step += 1
				PQueue.put([-priority, queue_time_step, S, A]) # appending -ve sign to Pqueue to prefer larger value of priority
			for _ in range(num_planning):
				
				if PQueue.empty():
					break
				_ , _ , p_S , p_A = PQueue.get()
				
				index = np.random.choice(len(model[p_S][p_A])) #choosing a random reaward and next state from all previous encounters 
				p_R, p_Sp = model[p_S][p_A][index] 
				Q[p_S][p_A] += alpha * (reward + gamma * np.max( Q[p_Sp]) - Q[p_S][p_A])
				leading_states_actions=np.unique(path_to_state[new_state], axis=0)
				# print leading_states_actions
				for S_dash, A_dash, transit_reward in leading_states_actions:
					# print S_dash, A_dash, transit_reward
					S_dash= int(S_dash)
					A_dash= int(A_dash)
					
					R_dash = transit_reward
					priority = np.abs(transit_reward + gamma * np.max(Q[p_S]) - Q[S_dash][A_dash])
					if priority > theta:
						queue_time_step += 1
						PQueue.put([-priority, queue_time_step, S_dash, A_dash]) # appending -ve sign to Pqueue to prefer larger value of priority
				
			S=new_state

	policy = np.zeros((env.nS , env.nA))
	for i in range(env.nS):
		policy[i][np.argmax(Q[i])]=1
	# print time_state_action
	print Q
	return policy, cumm_reward