import random
import matplotlib.pyplot as plt


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

		# if verbose:
		# 	print('Action:', action, '--- New car position:', self.carPos, '--- Passenger Picked:', self.passengerPicked, '--- Passenger Dropped:', self.passengerDropped)
		return newState

	def passenger_dropped(self):
		return self.destination == self.passengerPos

	def get_next_state(self, curState, action):
		if action not in self.actionSpace[curState[0]]:
			return (self.carPos, self.passengerInCar, self.passengerPos)

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
				self.policy[(cell1, True, cell2)] = WEST
			

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
		while not self.grid.passenger_dropped():
			action = self.policy[(self.grid.carPos, self.grid.passengerInCar, self.grid.passengerPos)]
			if action in [PICKUP, PUTDOWN]:
				self.grid.perform_action(action)
			else:
				if Utility.sample_navigation():
					self.grid.perform_action(action)
				else:
					self.grid.perform_action(Utility.sample_navigation_action())


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


	def policy_iteration(self, linalg = False):
		return NotImplemented

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



# grid = Grid('grid_5x5.txt', (1,2), (1,1), (1,5))
# td1 = TaxiDomain(grid, [(1,1), (1,5), (5,1), (5,4)])

grid2 = Grid('grid_2x2.txt', (2,1), (1,1), (1,2))
td1 = TaxiDomain(grid2, [(1,2)])


rewards = []
iterations = 1
for i in range(200):
	r, iterations = td1.q_learning_episode(sarsa=False, decaying=True, iterations=iterations)
	rewards.append(r)

for key in td1.policy.keys():
	for action in action_list:
		print(key, action, td1.Q[(key, action)])
		pass


plt.plot(rewards)
plt.show()

# print(grid.actionSpace)
# grid.perform_action(EAST, verbose = True)

