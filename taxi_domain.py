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
	def __init__():
		self.carPos = None
		self.passengerPos = None
		self.destination = None
		self.cells = None

	def perform_action(self, action):
		return NotImplemented 


class TaxiDomain:
	def __init__(self, grid, online=False):
		self.grid = grid
		self.policy = {}
		for cell in self.grid.cells:
			self.policy[cell] = WEST

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
			curCell = self.grid.carPos
			action = self.policy[curCell]
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


