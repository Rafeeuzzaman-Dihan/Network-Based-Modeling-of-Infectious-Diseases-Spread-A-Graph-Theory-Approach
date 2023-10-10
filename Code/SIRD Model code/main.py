import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Set the initial conditions
N = 166303498  # Population of Bangladesh
I0 = 1000  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
D0 = 0  # Initial number of deceased individuals
S0 = N - I0 - R0 - D0  # Initial number of susceptible individuals

# Set the model parameters
beta = 0.15856731724371695  # Transmission rate
gamma = 0.00134949822620583189  # Recovery rate
mu = 0.0167  # Mortality rate

# Set the time points to simulate
t = np.linspace(0, 365, 365)  # Simulate for 365 days

# Define the SIRD model equations
def sird_model(y, t, N, beta, gamma, mu):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return dSdt, dIdt, dRdt, dDdt

# Solve the SIRD model equations
y0 = S0, I0, R0, D0
sol = odeint(sird_model, y0, t, args=(N, beta, gamma, mu))
S, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SIRD Model of COVID-19 (Bangladesh)')
plt.legend()
plt.grid(True)
plt.show()