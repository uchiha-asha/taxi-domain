import sys
import matplotlib.pyplot as plt
from taxi_domain import Grid, TaxiDomain
import json, random

def get_plot(data, xlabel, ylabel, title, folder):
	plt.figure(figsize=(16, 6))
	plt.plot(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	
	plt.savefig(folder+'/'+title+'.png')


if sys.argv[1] == '1':
	if sys.argv[2] == '2':
		grid = Grid('grid_5x5.txt', (3,3), (1,1), (5,5))
		taxi = TaxiDomain(grid, [(5,5)])
		if sys.argv[3] == '1':
			taxi.value_iteration(sys.argv[4])
			taxi.simulate()
		elif sys.argv[3] == '2':
			gammas = [0.01, 0.1, 0.5, 0.8, 0.99]
			colors = ['r', 'b', 'g', 'y', 'tab:pink']

			for i in range(len(gammas)):
				norm_index = taxi.value_iteration(1e-10, gamma=gammas[i])
				print(taxi.policy)
				print(norm_index)
				plt.plot([i+1 for i in range(len(norm_index))], norm_index, color=colors[i], label=gammas[i])

			plt.legend()
			plt.savefig("A3-Q2.2.png")
		elif sys.argv[3] == '3':
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

elif sys.argv[1] == '2':
	if sys.argv[2] == '2':
		ind = 0
		names = ['Q-Learning', 'Q-Learning with decay', 'SARSA', 'SARSA with decay']
		for sarsa in [False, True]:
			for decay in [False, True]:
				random.seed(69)
				grid = Grid('grid_5x5.txt', (1,2), (1,1), (1,5))
				td = TaxiDomain(grid, [(1,1), (1,5), (5,1), (5,4)])
				discounted_reward = []
				iterations = 1
				for i in range(2000):
					_, iterations = td.q_learning_episode(sarsa=sarsa, decaying=decay, iterations=iterations)
					rewards = 0
					if i%10==0:
						for j in range(300):
							reward, _ = td.q_learning_episode(sarsa=sarsa, evaluate=True, epsilon=0)
							rewards += reward
						discounted_reward.append(rewards/300)

				print('Maximum reward for', names[ind], 'is:', max(discounted_reward), 'at', discounted_reward.index(max(discounted_reward))*10)
				get_plot(discounted_reward, 'Episodes*10', 'Reward', names[ind], 'B')
				ind += 1
	elif sys.argv[2] == '3':
		random.seed(69)
		grid = Grid('grid_5x5.txt', (1,2), (1,1), (1,5))
		td = TaxiDomain(grid, [(1,1), (1,5), (5,1), (5,4)])
		iterations=1
		for i in range(2000):
			_, iterations = td.q_learning_episode(sarsa=True, decaying=True, iterations=iterations)
		for i in range(5):
			iterations=1
			reward, iterations = td.q_learning_episode(sarsa=True, evaluate=True, epsilon=0, maxEpisodeIter=2000)
			print('Reward: ', reward, end = ', ')
			if iterations <= 1999:
				print('Goal achieved in ', iterations, 'steps')
			else:
				print('Not able to reach goal')

	elif sys.argv[2] == '4':
		random.seed(69)
		grid = Grid('grid_5x5.txt', (1,2), (1,1), (1,5))
		
		for exploration in [0, 0.05, 0.1, 0.5, 0.9]:
			rewards = []
			td = TaxiDomain(grid, [(1,1), (1,5), (5,1), (5,4)])
			for i in range(2000):
				reward, _ = td.q_learning_episode(epsilon=exploration)
				rewards.append(reward)
			get_plot(rewards, 'Episodes', 'Reward', 'epsilon='+str(exploration), 'B')

		for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
			rewards = []
			td = TaxiDomain(grid, [(1,1), (1,5), (5,1), (5,4)])
			for i in range(2000):
				reward, _ = td.q_learning_episode(alpha=alpha)
				rewards.append(reward)
			get_plot(rewards, 'Episodes', 'Reward', 'alpha='+str(alpha), 'B')

	elif sys.argv[2] == '5':
		depos = [(1,1),(1,6),(1,9),(4,4),(5,7),(9,1),(10,5),(10,10)]
		total_reward = 0
		random.seed(6969)
		for k in range(5):
			destination = depos[random.randint(0, 7)]
			print('Training policy for destination', destination)
			grid = Grid('grid_10x10.txt', (1,1), (1,1), destination)
			td  = TaxiDomain(grid, depos)
			for i in range(10000):
				reward, _ = td.q_learning_episode(sarsa=True)
			local_reward = 0
			for j in range(10):
				reward, _ = td.q_learning_episode(sarsa=True, evaluate=True, epsilon=0)
				local_reward += reward
			total_reward += local_reward/10
		print('Average accumulated discounted reward: ', total_reward/5)







