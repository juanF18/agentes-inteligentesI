import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import time

# Crear el entorno
env = gym.make("ALE/SpaceInvaders-v5")

# Parámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.01  # Parámetro epsilon para la política epsilon-greedy
num_episodes = 1000
max_steps = 100

# Inicializar la Q-Table
Q = defaultdict(lambda: np.zeros(env.action_space.n))


# Función para elegir acción usando la política epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Función para procesar el estado
def preprocess_state(state):
    return tuple(state[::8, ::8].mean(axis=2).flatten())  # Simplificación extrema


# Entrenamiento del agente usando SARSA con barra de progreso y visualización en consola
rewards = []
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    state, _ = env.reset()
    state = preprocess_state(state)
    action = choose_action(state)
    total_reward = 0
    for step in range(max_steps):
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_state(next_state)
        next_action = choose_action(next_state)
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )
        state = next_state
        action = next_action
        total_reward += reward

        time.sleep(0.01)  # Añadir un pequeño retraso para mejor visualización

        if done:
            break
    rewards.append(total_reward)

# Graficar las recompensas
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")
plt.show()

# Cerrar el entorno
env.close()
