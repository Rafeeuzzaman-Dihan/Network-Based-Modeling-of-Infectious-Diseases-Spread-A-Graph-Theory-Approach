import numpy as np
import matplotlib.pyplot as plt

# Define the initial conditions
total_population = 167420951 # Total population of Bangladesh in 2020
initial_infected = 3          # Initial number of infected individuals
initial_recovered = 0         # Initial number of recovered individuals
initial_deaths = 0          # Initial number of deaths individuals
initial_susceptible = total_population - initial_infected - initial_recovered - initial_deaths

# Define the model parameters
beta = 0.1585672707903705 # Transmission rate
gamma = 0.013494967200694463 # Recovery rate
t_max = 365 # Maximum time to simulate (days)

# Define the time grid
t = np.arange(t_max)

# Define the SIR model
def sir_model(S, I, R, beta, gamma, N):
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Simulate the model
S = np.zeros(t_max)
I = np.zeros(t_max)
R = np.zeros(t_max)
S[0] = initial_susceptible
I[0] = initial_infected
R[0] = initial_recovered

for i in range(1, t_max):
    dSdt, dIdt, dRdt = sir_model(S[i-1], I[i-1], R[i-1], beta, gamma, total_population)
    S[i] = S[i-1] + dSdt
    I[i] = I[i-1] + dIdt
    R[i] = R[i-1] + dRdt

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model of COVID 19 (Bangladesh)')
plt.legend()
plt.show()