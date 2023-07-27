import pandas as pd
import numpy as np

data = pd.read_csv("AAPL.csv")
data = data.drop(['Date', 'Adj Close', 'Volume'], axis = 1)

X = data.drop('Close', axis = 1).values
y = data['Close'].values

def fitness_function(params, X, y):
    w = params[:-1]
    b = params[-1]
    y_pred = X.dot(w) + b
    mse = np.mean((y_pred - y) ** 2)
    return mse

def abc(X, y, population_size = 10, limit = 100, lb = -1, ub = 1):
    fitness_history = []
    best_solution = None
    best_fitness = np.inf
    solutions = np.zeros((population_size, X.shape[1] + 1))
    for i in range(population_size):
        solutions[i, :-1] = lb + (ub - lb) * np.random.rand(X.shape[1])
        solutions[i, -1] = lb + (ub - lb) * np.random.rand()
        fitness = fitness_function(solutions[i], X, y)
        fitness_history.append(fitness)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solutions[i].copy()
    
    for it in range(limit):
        for i in range(population_size):
            #emplyed bee
            s = solutions[i].copy()
            j = np.random.randint(X.shape[1] + 1)
            k = np.random.randint(population_size)
            while k == 1:
                k = np.random.randint(population_size)
            s[j] = solutions[i, j] + np.random.uniform(-1, 1) * (solutions[i, j] - solutions[k, j])
            if s[-1] < lb:
                s[-1] = lb
            elif s[-1] > ub:
                s[-1] = ub
            fitness = fitness_function(s, X, y)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = s.copy()
            if fitness < fitness_function(solutions[i], X, y):
                solutions[i] = s.copy()
            else:
                #scout bee phase
                solutions[i, :-1] = lb + (ub - lb) * np.random.rand(X.shape[-1])
                solutions[i, -1] = lb + (ub - lb) * np.random.rand()
    return best_solution, best_fitness, fitness_history

best_solution, best_fitness, _ = abc(X, y)

print(best_fitness)
print(best_solution)

w = best_solution[:-1]
b = best_solution[-1]
y_pred = X.dot(w) + b

mse_train = np.mean((y_pred - y) ** 2)
print(f'mse on data set :  {mse_train:.2f}')

import matplotlib.pyplot as plt

plt.plot(y, label = 'actual prices')
plt.plot(y_pred, label = 'predicted prices')
plt.xlabel('time')
plt.ylabel('price')
plt.legend
plt.show()

print(y_pred)

population_size = [5, 10, 50, 100]
colors = ['b', 'r', 'g', 'c']
limit = 200

fig, axs = plt.subplots(2, 2)
for i in range(0, len(population_size)):
    _, _, fitness_history = abc(X, y, population_size[i], limit)

    axs[int(i/2), i%2].plot(range(len(fitness_history)), fitness_history, colors[i], label = 'population size {}'.format(population_size[i]))
    axs[int(i/2), i%2].set_title('population size {}'.format(population_size[i]))

    fig.suptitle('error vs iterations')

plt.show()

