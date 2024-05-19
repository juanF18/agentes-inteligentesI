import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear el entorno
env = gym.make("Acrobot-v1")


# Función para discretizar el espacio de estados
def discretize_state(state, bins):
    state_disc = []
    for i in range(len(state)):
        state_disc.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_disc)


# Función para crear bins de discretización
def create_bins(num_bins, lower_bounds, upper_bounds):
    bins = []
    for l, u in zip(lower_bounds, upper_bounds):
        bins.append(np.linspace(l, u, num_bins))
    return bins


# Parámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento que determina la importancia de recompensas futuras
epsilon = 0.1  # Parámetro epsilon para la política epsilon-greedy que controla la exploración vs la explotación
num_episodes = 1000  # Número total de episodios de entrenamiento
max_steps = 500  # Número máximo de pasos por episodio
num_bins = 10  # Número de bins para discretizar cada dimensión del espacio de estados

# Crear bins para la discretización
lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
upper_bounds[1] = 1  # Ajustar límites superiores para las velocidades
upper_bounds[3] = 1
bins = create_bins(num_bins, lower_bounds, upper_bounds)

# Inicializar la Q-Table
Q = defaultdict(lambda: np.zeros(env.action_space.n))


# Función para elegir acción usando la política epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Entrenamiento del agente usando TD(0) con barra de progreso y visualización en consola
rewards = []
td_errors = []  # Lista para almacenar los errores TD

for episode in tqdm(range(num_episodes), desc="Episodios de entrenamiento"):
    state, _ = env.reset()
    state = discretize_state(state, bins)
    total_reward = 0
    episode_td_errors = []  # Lista para almacenar los errores TD de un episodio

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state, bins)
        best_next_action = np.argmax(Q[next_state])

        # Calcular el TD error
        td_error = reward + gamma * Q[next_state][best_next_action] - Q[state][action]
        episode_td_errors.append(td_error)

        # Actualizar la Q-Table
        Q[state][action] += alpha * td_error

        state = next_state
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)
    td_errors.append(
        np.mean(episode_td_errors)
    )  # Promedio de los errores TD del episodio

# Graficar las recompensas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")

# Graficar el TD error
plt.subplot(1, 2, 2)
plt.plot(range(num_episodes), td_errors)
plt.xlabel("Episodes")
plt.ylabel("TD Error")
plt.title("TD Error vs Episodes")

plt.tight_layout()
plt.show()

# Cerrar el entorno
env.close()
