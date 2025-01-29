import numpy as np

# Define the environment (GridWorld)
grid_size = 4
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)

# Initialize the Q-table
q_table = np.zeros((grid_size, grid_size, num_actions))

# Define the rewards and terminal state
rewards = np.zeros((grid_size, grid_size))
rewards[3, 3] = 1  # Goal state
terminal_state = (3, 3)

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration probability
num_episodes = 1000

# Helper function to get the next state
def get_next_state(state, action):
    x, y = state
    if action == 0 and x > 0:  # Up
        x -= 1
    elif action == 1 and x < grid_size - 1:  # Down
        x += 1
    elif action == 2 and y > 0:  # Left
        y -= 1
    elif action == 3 and y < grid_size - 1:  # Right
        y += 1
    return (x, y)

# Training loop
for episode in range(num_episodes):
    state = (0, 0)  # Start at top-left corner
    while state != terminal_state:
        x, y = state
        
        # Choose an action (epsilon-greedy policy)
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)  # Explore
        else:
            action = np.argmax(q_table[x, y])  # Exploit
        
        # Get next state and reward
        next_state = get_next_state(state, action)
        reward = rewards[next_state]
        nx, ny = next_state

        # Update Q-value
        q_table[x, y, action] = q_table[x, y, action] + alpha * (
            reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
        )

        # Move to the next state
        state = next_state

# Display the learned Q-values
print("Learned Q-table:")
print(q_table)