import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data from CSV file
df = pd.read_csv("COVIDseir.csv")

# set model parameters
N = 167420951          # total population
I0 = 3                 # initial number of infected individuals
E0 = 2.1333            # initial number of Exposed individuals
R0 = 0                 # initial number of recovered individuals
S0 = N-I0-E0-R0        # initial number of susceptible individuals
beta = 0.158           # infection rate
gamma = 0.013          # recovery rate
sigma = 0.1515         # rate of progression from exposed to infectious

# define SEIR function
def SEIR_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# set initial conditions
y0 = S0, 0, I0, R0

# set time points
t = np.arange(365)

# solve SEIR model
from scipy.integrate import odeint
sol = odeint(SEIR_model, y0, t, args=(beta, gamma, sigma))

# plot the results
plt.plot(t, sol[:, 0], label='Susceptible')
plt.plot(t, sol[:, 1], label='Exposed')
plt.plot(t, sol[:, 2], label='Infected')
plt.plot(t, sol[:, 3], label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of individuals')
plt.title('SEIR Model of COVID 19 (Bangladesh)')
plt.legend()
plt.show()