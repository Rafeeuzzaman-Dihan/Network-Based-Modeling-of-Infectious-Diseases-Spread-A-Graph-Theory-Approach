import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SEIRD Model Parameters
beta = 0.158  # Transmission rate
gamma = 0.01  # Recovery rate
mu = 0.0167  # Mortality rate

# Initial conditions
S0 = 167420948  # Initial susceptible population
I0 = 3  # Initial infected cases
R0 = 0  # Initial recovered cases
D0 = 0  # Initial deaths

# Time vector
t = np.arange(0, 500, 1)  # Assuming 365 days of simulation

# SEIRD Model
def seird_model(y, t):
    S, E, I, R, D = y

    dSdt = -beta * S * I
    dEdt = beta * S * I - gamma * E
    dIdt = gamma * E - (1 - mu) * I - mu * I
    dRdt = (1 - mu) * I
    dDdt = mu * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# Initial conditions vector
y0 = [S0, 0, I0, R0, D0]

# Solve the SEIRD model
sol = odeint(seird_model, y0, t)

# Extracting variables from the solution
S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deaths')
plt.xlabel('Time (Days)')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()