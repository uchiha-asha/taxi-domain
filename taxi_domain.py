import random
'''<<<<<<< Updated upstream
import matplotlib.pyplot as plt

======='''
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
'''>>>>>>> Stashed changes'''

NORTH = 'North'
SOUTH = 'South'
EAST = 'East'
WEST = 'West'
PICKUP = 'Pick Up'
PUTDOWN = 'Put Down'
action_list = [NORTH, SOUTH, EAST, WEST, PICKUP, PUTDOWN]

ALLACTIONS = [NORTH, SOUTH, EAST, WEST, PICKUP, PUTDOWN]
ALLNAVIGATIONS = [NORTH, SOUTH, EAST, WEST]

class Utility:
	def sample_navigation(self):
		x = random.randint(1, 100)
		return True


	def sample_navigation_action():
		x = random.randint(0, 3)
		return action_list[x]

	def sample_action():
		x = random.randint(0, 5)
		return action_list[x]

	def sample_exploration(epsilon):
		x = random.randint(1, 10000)
		# print(x)
		return x <= int(epsilon*10000)



class Grid:
	def __init__(self, gridPath, carPos, passengerPos, destination):
		self.carPos = carPos
		self.passengerPos = passengerPos
		self.destination = destination
		self.actionSpace = {}

		self.passengerInCar = False
		self.rows = 0
		self.cols = 0

		self.construct_grid(gridPath)

	# Only vertical wall allowed apart from box wall
	def construct_grid(self, gridPath):
		file = open(gridPath, "r")
		lines = file.readlines()
		curRow = None
		for i in range(len(lines)):
			row = list(lines[i].split())
			curRow, curCol = i+1, 1
			for j in range(len(row)):
				if row[j] == '0':
					actions = []
					# No horizontal wall
					if i != 0:
						actions.append(NORTH)
					if i != len(lines)-1:
						actions.append(SOUTH)
					if curCol != 1 and row[j-1] != '|':
						actions.append(WEST)
					if j < len(row)-1 and (j == len(row)-1 or row[j+1] != '|'):
						actions.append(EAST)
					actions.extend([PICKUP, PUTDOWN])
					self.actionSpace[(curRow, curCol)] = actions
					curCol += 1
			self.cols = curCol-1
		self.rows = curRow


	def perform_action(self, action, verbose=False):
		newState = self.get_next_state((self.carPos, self.passengerInCar, self.passengerPos), action)
		
		(self.carPos, self.passengerInCar, self.passengerPos) = newState

		if verbose:
			print('Action:', action, '--- New car position:', self.carPos, '--- Passenger Picked:', self.passengerInCar, '--- Passenger position:', self.passengerPos)
		return newState

	def passenger_dropped(self):
		return self.destination == self.passengerPos and self.passengerInCar == False

	def get_next_state(self, curState, action):
		if action not in self.actionSpace[curState[0]]:
			return curState

		x, y = curState[0][0], curState[0][1]
		passengerInCar, passengerPos = curState[1], curState[2]
		if action == NORTH and x != 1:
			x -= 1
		if action == SOUTH and x != self.rows:
			x += 1
		if action == EAST and y != self.cols:
			y += 1
		if action == WEST and y != 1:
			y -= 1
		if action == PICKUP and curState[0] == curState[2]:
			passengerInCar = True
		if action == PUTDOWN and curState[1]:
			passengerInCar = False

		if passengerInCar:
			passengerPos = (x,y)

		return ((x, y), passengerInCar, passengerPos)



class TaxiDomain:
	def __init__(self, grid, depos, online=False):
		self.grid = grid
		self.depos = depos
		self.policy = {}
		self.Q = None
		for cell1 in self.grid.actionSpace.keys():
			for cell2 in self.grid.actionSpace.keys():
				self.policy[(cell1, False, cell2)] = WEST
				self.policy[(cell1, True, cell1)] = WEST
			

	def get_reward(self, curState, action):
		if action == PUTDOWN:
			if curState[0] == self.grid.destination and curState[1]:
				return 20
			elif curState[0] != curState[2]:
				return -10
			else:
				return -1

		if action == PICKUP and curState[2] != curState[0]:
			return -10
		
		return -1

	def simulate(self):
		while True:
			action = self.policy[(self.grid.carPos, self.grid.passengerInCar, self.grid.passengerPos)]
			
			if self.grid.passengerPos in self.depos and self.grid.passengerInCar != False:
				break
				
			self.grid.perform_action(action, verbose=True)


	def decay(self, epsilon=0.1, iterations=1):
		return epsilon/iterations



	def value_iteration(self, eps, iterations=1000, gamma = 0.9):
		U = {}
		U1 = {state : (0, None) for state in self.policy.keys()}

		max_norm_index = []

		delta = eps*(1-gamma)/gamma
		threshold = eps*(1-gamma)/gamma

		iteration = 0

		while delta >= eps*(1-gamma)/gamma and iteration < iterations:
			iteration += 1
			U = U1.copy()
			delta = 0
			for s in self.policy.keys():
				mx = 0
				mx_act = None

				if s[0] == self.grid.destination and s[1] == True:
					U1[s] = (20, None)
					continue
				for action in self.grid.actionSpace[s[0]]:
					if action not in ALLNAVIGATIONS:
						reward = self.get_reward(s, action)
						sample = reward + gamma * U[self.grid.get_next_state(s, action)][0]
						if mx_act == None or sample > mx:
							mx = sample
							mx_act = action
					else:
						sample = 0
						for a in ALLNAVIGATIONS:
							if a == action:
								sample += 0.85*(self.get_reward(s, a) + gamma * U[self.grid.get_next_state(s, a)][0])
							elif a not in self.grid.actionSpace[s[0]]:
								sample += 0.05*(-1 + gamma * U[s][0])
							else:
								sample += 0.05*(self.get_reward(s,a) + gamma* U[self.grid.get_next_state(s, a)][0])

						if mx_act == None or sample > mx:
							mx = sample
							mx_act = action



				U1[s] = (mx, mx_act)

				if abs(U1[s][0] - U[s][0]) > delta:
					delta = abs(U1[s][0] - U[s][0])

			max_norm_index.append(delta)

		for key in U.keys():
			self.policy[key] = U[key][1]

		print("convergence obtained in", iteration, "iterations.")

		return max_norm_index

	'''<<<<<<< Updated upstream

		def policy_iteration(self, linalg = False):
			return NotImplemented
	======='''
	def P_matrix(self):
		idx_mapping = {}
		j = 0

		for key in self.policy.keys():
			idx_mapping[key] = j
			j += 1

		P = {}

		for s in self.policy.keys():
			A = {}

			for action in ALLACTIONS:
				if action not in ALLNAVIGATIONS:
					k = {state : 0 for state in self.policy.keys()}

					sNext = self.grid.get_next_state(s, action) 
					k[sNext] = 1

					A[action] = k
				else:
					k = {state : 0 for state in self.policy.keys()}

					for a in ALLNAVIGATIONS:
						if a == action:
							next = self.grid.get_next_state(s, a)
							k[next] += 0.85
						else:
							next = self.grid.get_next_state(s, a)
							k[next] += 0.05

					A[action] = k



			P[s] = A

		return P

	def R_matrix(self):
		idx_mapping = {}
		j = 0

		for key in self.policy.keys():
			idx_mapping[key] = j
			j += 1

		R = {}

		for s in self.policy.keys():
			A = {}

			for action in ALLACTIONS:
				if action not in ALLNAVIGATIONS:
					k = {state : 0 for state in self.policy.keys()}

					re = self.get_reward(s, action)

					k[s] = re

					A[action] = k
				else:
					k = {state : 0 for state in self.policy.keys()}

					re = self.get_reward(s, action)

					k[s] = re

					A[action] = k

			R[s] = A

		return R

	def linalg_policy_evaluation(self, P, R, gamma):
		X_mat = []
		Y_mat = []

		for s in self.policy.keys():
			x = []
			y = 0

			for s1 in self.policy.keys():
				if s == s1:
					x.append(1 - P[s][self.policy[s]][s1] * gamma)
					y += P[s][self.policy[s]][s1] * R[s][self.policy[s]][s1]
				elif P[s][self.policy[s]][s1] != 0:
					x.append(-gamma * P[s][self.policy[s]][s1])
					y += P[s][self.policy[s]][s1] * R[s][self.policy[s]][s1]
				else:
					x.append(0)

			X_mat.append(x)
			Y_mat.append(y)

		# print(X_mat)
		#print(Y_mat)
		V_mat = np.linalg.solve(X_mat, Y_mat)
		V = {}
		i = 0
		for s in self.policy.keys():
			V[s] = V_mat[i]
			i += 1

		# print(V)
		# print(X_mat)
		return V

	def iterative_policy_evaluation(self, gamma, eps, iterations = 10):
		iteration = 1
		V0 = {s: 0 for s in self.policy.keys()}

		while True:
			pass

	def get_max_action(self, s, P, R, V):
		mx = 0
		mx_act = None

		for action in ALLACTIONS:
			sum = 0
			for s1 in self.policy.keys():
				sum += P[s][action][s1]*(R[s][action][s1] + V[s1])

			sum += self.get_reward(s, action)
			if sum > mx or mx_act == None:
				mx = sum
				mx_act = action

		return mx_act

	def policy_iteration(self, epsilon = 0.01, iterations = 100, gamma = 0.9, linalg = False):
		idx_mapping = {}
		j = 0
		P = self.P_matrix()
		R = self.R_matrix()

		for key in self.policy.keys():
			idx_mapping[key] = j
			j += 1

		method = [self.iterative_policy_evaluation, self.linalg_policy_evaluation]

		iteration = 1
		unchanged = False

		while not unchanged and iteration < iterations:
			print("iteration:", iteration)
			iteration += 1
			unchanged = True
			changes = []
			V = method[linalg](P, R, gamma)
			#print(V)
			for s in self.policy.keys():
				act1 = self.policy[s]
				act2 = self.get_max_action(s, P, R, V)

				if act1 != act2:
					self.policy[s] = act2
					changes.append([s, act1, act2])
					unchanged = False

			print(len(changes))
			

	'''
	>>>>>>> Stashed changes
	'''
	def best_action(self, curState):
		best_action_val, best_q_value = NORTH, -(2**63)
		for action in action_list:
			if self.Q[(curState, action)] > best_q_value and (action in self.grid.actionSpace[curState[0]]):
				best_q_value = self.Q[(curState, action)]
				best_action_val = action
		return [best_action_val, best_q_value]

	def sample_episode(self):
		cell = list(self.grid.actionSpace.keys())[random.randint(0, len(self.grid.actionSpace)-1)]
		passengerPos = list(self.grid.actionSpace.keys())[random.randint(0, len(self.grid.actionSpace)-1)]
		return cell, passengerPos

	def sample_action(self, curState, epsilon):
		if Utility.sample_exploration(epsilon):
			action = None
			while action is None:
				action = Utility.sample_action()
				if action not in self.grid.actionSpace[curState[0]]:
					action = None	
		else:
			action = self.best_action(curState)[0]
		return action

	def q_learning_episode(self, alpha=0.25, epsilon=0.1, gamma=0.99, 
		maxEpisodeIter=500, decaying=False, iterations=1, sarsa=False, evaluate=False):
		self.grid.carPos, self.grid.passengerPos = self.sample_episode()
		curState = (self.grid.carPos, False, self.grid.passengerPos)
		if self.Q is None:
			self.Q = {}
			for key in self.policy.keys():
				for action in action_list:
					self.Q[(key, action)] = 0
					
		discounted_reward, gamma1 = 0, 1
		for i in range(maxEpisodeIter):
			if curState[2] == self.grid.destination and (not curState[1]):
				return discounted_reward, iterations

			if decaying:
				epsilon1 = self.decay(epsilon, iterations)
			else:
				epsilon1 = epsilon
			action = self.sample_action(curState, epsilon1)
			reward = self.get_reward(curState, action)
			newState = self.grid.get_next_state(curState, action)

			cur_q_value = self.Q[(curState, action)]
			if not evaluate:
				if sarsa:
					new_action = self.sample_action(newState, epsilon1)
					new_q_value = self.Q[(newState, new_action)]
				else:
					new_q_value = self.best_action(newState)[1]
				self.Q[(curState, action)] = (1-alpha)*cur_q_value + alpha*(reward + gamma*new_q_value)

			discounted_reward += gamma1*reward
			gamma1 *= gamma

			curState = newState
			iterations += 1
		
		return discounted_reward, iterations

	def sarsa(self, epsilon = 0.1, decaying = False):
		return NotImplemented
'''
<<<<<<< Updated upstream
'''

# grid = Grid('grid_5x5.txt', (1,2), (1,1), (1,5))
# td1 = TaxiDomain(grid, [(1,1), (1,5), (5,1), (5,4)])

# grid2 = Grid('grid_2x2.txt', (2,1), (1,1), (1,2))
# td1 = TaxiDomain(grid2, [(1,2)])


# rewards = []
# iterations = 1
# for i in range(2000):
# 	r, iterations = td1.q_learning_episode(sarsa=False, decaying=True, iterations=iterations)
# 	rewards.append(r)

# for key in td1.policy.keys():
# 	for action in action_list:
# 		print(key, action, td1.Q[(key, action)])
# 		pass


# plt.plot(rewards)
# plt.show()

# print(grid.actionSpace)
# grid.perform_action(EAST, verbose = True)
'''
======='''
if __name__ == '__main__':
	grid = Grid('grid_5x5.txt', (3,3), (1,1), (5,5))
	taxi = TaxiDomain(grid, [(5,5)])
	taxi.policy_iteration(linalg = True)
	#print(grid.actionSpace[(2,2)])
	#print(taxi.R_matrix()[((2,2), False, (2,1))][PICKUP])
	print(taxi.policy)

	taxi.simulate()
'''>>>>>>> Stashed changes
'''