import random


NORTH = 'North'
SOUTH = 'South'
EAST = 'East'
WEST = 'West'
PICKUP = 'Pick Up'
PUTDOWN = 'Put Down'

ALLACTIONS = [NORTH, SOUTH, EAST, WEST, PICKUP, PUTDOWN]
ALLNAVIGATIONS = [NORTH, SOUTH, EAST, WEST]

class Utility:
	def sample_navigation(self):
		x = random.randint(1, 100)
		return True

	def sample_navigation_action(self):
		x = random.randint(1, 100)
		if x <= 25:
			return NORTH
		if x <= 50:
			return SOUTH
		if x <= 75:
			return EAST
		return WEST


class Grid:
	def __init__(self, gridPath, carPos, passengerPos, destination):
		self.carPos = carPos
		self.passengerPos = passengerPos
		self.destination = destination
		self.actionSpace = {}
		self.cells = []
		self.passengerPicked = False
		self.passengerDropped = False
		self.construct_grid(gridPath)

	# Only vertical wall allowed apart from box wall
	def construct_grid(self, gridPath):
		file = open(gridPath, "r")
		lines = file.readlines()
		for i in range(len(lines)):
			row = list(lines[i].split())
			curRow, curCol = i+1, 1
			for j in range(len(row)):
				if row[j] == '0':
					self.cells.append((curRow, curCol))
					actions = []
					# No horizontal wall
					if i != 0:
						actions.append(NORTH)
					if i != len(lines)-1:
						actions.append(SOUTH)
					if j != 0 and row[j-1] != '|':
						actions.append(WEST)
					if j != len(row)-1 and row[j+1] != '|':
						actions.append(EAST)
					actions.extend([PICKUP, PUTDOWN])
					self.actionSpace[(curRow, curCol)] = actions
					curCol += 1


	def perform_action(self, action, verbose=False):
		newState = self.get_next_state((self.carPos, self.passengerPicked, self.passengerDropped), action)
		
		(self.carPos, self.passengerPicked, self.passengerDropped) = newState
		if self.passengerPicked:
			self.passengerPos = self.carPos

		if verbose:
			print('Action:', action, '--- New car position:', self.carPos, '--- New passenger position:', self.passengerPos)
		return 

	def passenger_dropped(self):
		return self.passengerDropped

	def get_next_state(self, curState, action):
		print(curState)
		if action not in self.actionSpace[curState[0]]:
			return (self.carPos, self.passengerPicked, self.passengerDropped)

		x, y = curState[0][0], curState[0][1]
		isPassengerPicked, isPassengerDropped = curState[1], curState[2]
		if action == NORTH:
			y -= 1
		if action == SOUTH:
			y += 1
		if action == EAST:
			x += 1
		if action == WEST:
			x -= 1
		if action == PICKUP and self.passengerPos == self.carPos:
			isPassengerPicked = True
		if action == PUTDOWN and self.passengerPicked and self.passengerPos == self.destination:
			isPassengerDropped = True

		return ((x, y), isPassengerPicked, isPassengerDropped)


class TaxiDomain:
	def __init__(self, grid, online=False):
		self.grid = grid
		self.policy = {}
		for cell in self.grid.cells:
			self.policy[(cell, False, False)] = WEST
			self.policy[(cell, True, False)] = WEST
			self.policy[(cell, True, True)] = WEST

	def get_reward(self, curState, action):
		if action == PUTDOWN:
			if curState[0] == self.grid.destination:
				return 20
			else:
				return -10

		if action == PICKUP and self.grid.passengerPos != curState[0]:
			return -10
		
		return -1

	def simulate(self):
		while not self.grid.passenger_dropped():
			action = self.policy[(self.grid.carPos, self.grid.passengerPicked, self.grid.passengerDropped)]
			if action in [PICKUP, PUTDOWN]:
				self.grid.perform_action(action)
			else:
				if Utility.sample_navigation():
					self.grid.perform_action(action)
				else:
					self.grid.perform_action(Utility.sample_navigation_action())


	def value_iteration(self, eps, gamma = 0.9):
		U = {}
		U1 = {state : 0 for state in self.policy.keys()}

		delta = eps*(1-gamma)/gamma
		threshold = eps*(1-gamma)/gamma

		while delta >= eps*(1-gamma)/gamma:
			U = U1.copy()
			delta = 0

			for s in self.policy.keys():
				mx = 0
				mx_act = None

				if s[1] == True and s[2] == True:
					continue

				for action in self.grid.actionSpace[s[0]]:
					if action not in ALLNAVIGATIONS:
						reward = get_reward(s, action)
						sample = reward + gamma * U[s]

						if mx_act == None or sample > mx:
							mx = sample
							mx_act = action
					else:
						sample = 0
						for a in ALLNAVIGATIONS:
							if a == action:
								sample += 0.85*(self.get_reward(s, a) + gamma * U[self.grid.get_next_state(s, a)])
							elif a not in self.grid.actionSpace[s]:
								sample += 0.05*(-1 + gamma * U[s])
							else:
								sample += 0.05*(self.get_reward(s,a) + gamma* U[self.grid.get_next_state(s, a)])

						if mx_act == None or sample > mx:
							mx = sample
							mx_act = action


				U1[s] = mx

				if abs(U1[s] - U[s]) > delta:
					delta = abs(U1[s] - U[s])

		return U

	def policy_iteration(self, linalg = False):
		return NotImplemented

	def q_learning(self, epsilon = 0.1, decaying = False):
		return NotImplemented

	def sarsa(self, epsilon = 0.1, decaying = False):
		return NotImplemented


grid = Grid('grid_5x5.txt', (1,1), (1,1), (1,1))
taxi = TaxiDomain(grid)

print(grid.cells)
print(grid.actionSpace)

print(taxi.value_iteration(10))