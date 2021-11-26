import sys
import matplotlib.pyplot as plt
from taxi_domain import Grid, TaxiDomain
import json

if sys.argv[1] == '2':
	grid = Grid('grid_5x5.txt', (3,3), (1,1), (5,5))
	taxi = TaxiDomain(grid)
	if sys.argv[2] == '1':
		taxi.value_iteration(sys.argv[3])
		taxi.simulate()
	elif sys.argv[2] == '2':
		gammas = [0.01, 0.1, 0.5, 0.8, 0.99]
		colors = ['r', 'b', 'g', 'y', 'tab:pink']

		for i in range(len(gammas)):
			norm_index = taxi.value_iteration(1e-10, gamma=gammas[i])
			print(taxi.policy)
			print(norm_index)
			plt.plot([i+1 for i in range(len(norm_index))], norm_index, color=colors[i], label=gammas[i])

		plt.legend()
		plt.savefig("A3-Q2.2.png")
	elif sys.argv[2] == '3':
		gammas = [0.1, 0.99]

		passengerPos = [(1,1), (3,3), (2,1)]
		destination = [(1,1), (5,5), (4,3)]
		carPos = [(4,4), (4,3), (2,3)]

		policies = {}

		for i in range(len(carPos)):
			grid = Grid('grid_5x5.txt', passengerPos[i], destination[i], carPos[i])
			taxi = TaxiDomain(grid)
			for gamma in gammas:
				taxi.value_iteration(0.1, gamma=gamma, iterations=20)
				policy = {str(key): taxi.policy[key] for key in taxi.policy.keys()}
				policies[f"{passengerPos[i]}, {destination[i]}, {carPos[i]}, {gamma}"] = policy

		with open("A3-Q2.3.json","w") as f:
			json.dump(policies, f, indent=6)
	else:
		print("Not a Valid Execution...")






