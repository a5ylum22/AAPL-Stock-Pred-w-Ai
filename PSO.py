import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pso(n_particles, n_iterations, c1, c2, w, X, y):
    # Initialize particles and velocities
    particles = np.random.rand(n_particles, X.shape[1])
    velocities = np.zeros((n_particles, X.shape[1]))  # Initialize velocities with zeros
    
    # Initialize best positions and scores
    best_positions = particles
    best_scores = np.zeros(n_particles)
    
    # Initialize global best position and score
    global_best_position = np.zeros(X.shape[1])
    global_best_score = np.inf
    global_best_score_return = []

    # Iterate through each particle
    for i in range(n_iterations):
        # Update Velocities
        r1 = np.random.rand(n_particles, X.shape[1])
        r2 = np.random.rand(n_particles, X.shape[1])
        velocities = w * velocities + r1 * c1 * (best_positions - particles) + r2 * c2 * (global_best_position - particles)

        # Update Positions
        particles = particles + velocities

        # Evaluate Fitness for each particle
        scores = np.zeros(n_particles)
        for j in range(n_particles):
            y_pred = np.dot(X, particles[j])
            scores[j] = np.sqrt(np.mean((y - y_pred) ** 2))

            # Update best position and score for each particle
            if scores[j] < best_scores[j]:
                best_positions[j] = particles[j]
                best_scores[j] = scores[j]

            # Update global best position and score:
            if scores[j] < global_best_score:
                global_best_position = particles[j]
                global_best_score = scores[j]

        global_best_score_return.append(global_best_score)

        # Print current best score during each iteration (for debugging)
        print(f"Iteration {i+1}, Best Score: {global_best_score}")

    return global_best_position, global_best_score_return

# Data loading
df = pd.read_csv('AAPL.csv')
X = df.drop(['Date', 'Adj Close', 'Volume', 'Close'], axis=1).values
y = df['Close'].values

# PSO parameters
n_particles = 50
n_iterations = 500
c1 = 1
c2 = 1
w = 0.7

# Plot error convergence during iterations
score = []
for i in range(n_iterations):
    _, current_score = pso(n_particles, 1, c1, c2, w, X, y)
    score.append(current_score[0])  # Append the current score to the list

i = range(n_iterations)
plt.plot(i, score)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()

n_particles_list = [25, 50, 100, 200]
colors = ['b', 'r', 'g', 'c']

# Plot error convergence for different numbers of particles
#plt.figure()
for n_particle, color in zip(n_particles_list, colors):
    _, current_score = pso(n_particle, n_iterations, c1, c2, w, X, y)
    plt.plot(range(len(current_score)), current_score, color, label='n_particles = {}'.format(n_particle))

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title("Selecting Number of Particles")
plt.legend()
plt.axis([0, n_iterations, 0.25, 4])  # Set the axis limits
plt.show()
