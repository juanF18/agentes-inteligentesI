import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear el entorno
env = gym.make("ALE/SpaceInvaders-v5")

# Parámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.1  # Parámetro epsilon para la política epsilon-greedy
epsilon_min = 0.01  # Valor mínimo de epsilon
epsilon_decay = 0.995  # Factor de decaimiento de epsilon
num_episodes = 1000
max_steps = 1000  # Aumentar los pasos máximos para un entrenamiento más exhaustivo

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
    state = np.mean(state, axis=2).astype(np.uint8)  # Convertir a escala de grises
    state = state[::8, ::8]  # Redimensionar la imagen (downsampling)
    return tuple(state.flatten())


# Entrenamiento del agente usando TD(0) con barra de progreso y visualización en consola
rewards = []
td_errors = []  # Lista para almacenar los errores TD

for episode in tqdm(range(num_episodes), desc="Episodios de entrenamiento"):
    state, _ = env.reset()
    state = preprocess_state(state)
    total_reward = 0
    episode_td_errors = []  # Lista para almacenar los errores TD de un episodio

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_state(next_state)
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

    # Reducir epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Graficar las recompensas y el TD error
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")

plt.subplot(1, 2, 2)
plt.plot(range(num_episodes), td_errors)
plt.xlabel("Episodes")
plt.ylabel("TD Error")
plt.title("TD Error vs Episodes")

plt.tight_layout()
plt.show()

# Cerrar el entorno
env.close()
