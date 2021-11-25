import random


NORTH = 'North'
SOUTH = 'South'
EAST = 'East'
WEST = 'West'
PICKUP = 'Pick Up'
PUTDOWN = 'Put Down'


class Utility:
	def sample_navigation():
		x = random.randint(1, 100)
		return True

	def sample_navigation_action():
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
					if row[j-1] != '|':
						actions.append(WEST)
					if j == len(row)-1 or row[j+1] != '|':
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
			self.policy[(cell, False, True)] = WEST
			self.policy[(cell, True, False)] = WEST
			self.policy[(cell, True, True)] = WEST

	def get_reward(self, action):
		if action == PUTDOWN:
			if grid.carPos == grid.destination:
				return 20
			else:
				return -10

		if action == PICKUP and grid.passengerPos != grid.carPos:
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


	def value_iteration(self, gamma = 0.9):
		return NotImplemented

	def policy_iteration(self, linalg = False):
		return NotImplemented

	def q_learning(self, epsilon = 0.1, decaying = False):
		return NotImplemented

	def sarsa(self, epsilon = 0.1, decaying = False):
		return NotImplemented


grid = Grid('grid_5x5.txt', (1,1), (1,1), (1,1))
print(grid.actionSpace)
grid.perform_action(EAST, verbose = True)