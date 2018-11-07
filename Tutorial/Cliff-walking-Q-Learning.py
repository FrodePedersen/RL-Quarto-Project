import numpy as np
import random

def worldEnum(enum):
	if enum == "ROAD":
		return 0
	elif enum == "CLIFF":
		return 1
	elif enum == "GOAL":
		return 2
	else:
		raise ValueError(f"Unknown world type: {enum}")

def possibleActions(enum):
	if enum == "UP":
		return 0
	elif enum == "DOWN":
		return 1
	elif enum == "LEFT":
		return 2
	elif enum == "RIGHT":
		return 3
	else:
		raise ValueError(f"Unknown action: {enum}")

def translateActionNumbers(i):
	if i == 0:
		return "UP"
	elif i == 1:
		return "DOWN"
	elif i == 2:
		return "LEFT"
	elif i == 3:
		return "RIGHT"

def createWorld(gridWorldSize = (4,12)):
	grid = np.zeros(gridWorldSize)
	rows, cols = gridWorldSize
	for i in range(1,cols-1):
		grid[rows-1,i] = worldEnum("CLIFF")
	grid[rows-1, cols-1] = worldEnum("GOAL")
	return grid



def perform_action(state, grid, action):

	def new_agent_pos(state, grid, action):
		p = [state[0], state[1]]
		if action == possibleActions("UP"):
			p[0] = max(0, p[0]-1)
		elif action == possibleActions("DOWN"):
			p[0] = min(grid.shape[0] - 1, p[0] + 1)
		elif action == possibleActions("LEFT"):
			p[1] = max(0, p[1] - 1)
		elif action == possibleActions("RIGHT"):
			p[1] = min(grid.shape[1] - 1, p[1] + 1)
		else:
			raise ValueError(f"Unknown action: {action}")
		return p

	p = new_agent_pos(state, grid, action)
	reward = 0
	is_done = False
	grid_item = grid[p[0], p[1]]

	if grid_item == worldEnum("ROAD"):
		reward = -1
	#Fall off cliff incurs a penalty of 100 and resets position to the start position
	elif grid_item == worldEnum("CLIFF"):
		reward = -100
		p = [grid.shape[0]-1, 0]
	elif grid_item == worldEnum("GOAL"):
		reward = -1
		is_done = True
	else:
		raise ValueError(f"Unknown grid item {grid_item}")

	next_state = p

	#print(f"action performed: {action}, from pos {state}, taking the agent to pos: {p}, is_done: {is_done}, reward: {reward}")

	return next_state, reward, is_done

# def q(state, q_table,action=None):

# 	if state not in q_table:
# 		q_table[state] = np.zeroes(4)

# 	if action is None:
# 		return q_table[state]

# 	return q_table[state][action]

#returns either a random action with prob epsilon in the range 0-3 or the one with the highest q_value
def choose_action(state, q_table, epsilon):
	row = state[0]
	col = state[1]
	if random.uniform(0,1) < epsilon:
		return random.choice([0,1,2,3])
	else:
		#print(f"Choosing action: {np.argmax(q_table[row, col])}, from actions: {q_table[row,col]}")
		return np.argmax(q_table[row, col])


#NEEDS FIXING FOR THE NEXT_STATE PERHAPS? ALSO the GLOBAL VAR NAMES ETC
def training(start_state, environment, q_table, N_EPISODES=20, MAX_EPISODE_STEPS=200, alpha=0.9, gamma=1.0, epsilon=0.1):

	for e in range(N_EPISODES):

		state = start_state
		total_reward = 0
		alpha = alpha
		steps = []

		for _ in range(MAX_EPISODE_STEPS):
			action = choose_action(state, q_table, epsilon)
			steps.append(translateActionNumbers(action)) 
			next_state, reward, is_done = perform_action(state, environment, action)
			total_reward += reward

			q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1],action])
			state = next_state
			if is_done:
				break

		print(f"Episode {e + 1}: total reward -> {total_reward}")
		#print(f"steps: {steps}")

def main():
	random.seed(42)
	worldSize = (4,12)
	N_STATES = worldSize[0] * worldSize[1]
	N_EPISODES = 500
	MAX_EPISODE_STEPS = 50000
	alpha = 0.5
	gamma = 0.7
	epsilon = 0.1
	grid = createWorld(worldSize)
	#print(grid)
	#state = State(grid, [worldSize[0]-1,0])
	start_state = [3,0]

	#1-coord is the rows, 2-coord is the column, 3-coord is the action.
	q_table = np.zeros((worldSize[0],worldSize[1],4))

	training(start_state, grid, q_table, N_EPISODES, MAX_EPISODE_STEPS, alpha, gamma, epsilon)

	print("done training!")
	print(q_table)
	return 0

class State:

	def __init__(self, grid, agent_pos):
		self.grid = grid
		self.agent_pos = agent_pos



#########################
## TESTING ##
#########################

def test_walk_up():
	worldSize = (4,12)
	grid = createWorld(worldSize)
	state = State(grid, [worldSize[0]-1,0])
	print("Walking UP from {}".format(state.agent_pos))
	state, reward, is_done = action(state, possibleActions("UP"))
	print("new pos: {}".format(state.agent_pos))
	print(reward)
	print(is_done)

def test_walk_down():
	worldSize = (4,12)
	grid = createWorld(worldSize)
	state = State(grid, [worldSize[0]-1,0])
	print("Walking DOWN from {}".format(state.agent_pos))
	state, reward, is_done = action(state, possibleActions("DOWN"))
	print("new pos: {}".format(state.agent_pos))
	print(reward)
	print(is_done)

def test_walk_left():
	worldSize = (4,12)
	grid = createWorld(worldSize)
	state = State(grid, [worldSize[0]-1,0])
	print("Walking LEFT from {}".format(state.agent_pos))
	state, reward, is_done = action(state, possibleActions("LEFT"))
	print("new pos: {}".format(state.agent_pos))
	print(reward)
	print(is_done)

def test_walk_right():
	worldSize = (4,12)
	grid = createWorld(worldSize)
	state = State(grid, [worldSize[0]-1,0])
	print("Walking RIGHT from {}".format(state.agent_pos))
	state, reward, is_done = action(state, possibleActions("RIGHT"))
	print("new pos: {}".format(state.agent_pos))
	print(reward)
	print(is_done)

def test_walking():
	test_walk_up()
	test_walk_down()
	test_walk_left()
	test_walk_right()

#########################
## MAIN ##
#########################

if __name__ == '__main__':
	#test_walking()
	main()