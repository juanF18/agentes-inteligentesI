import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear el entorno
env = gym.make("Taxi-v3")

# Parámetros
# Tasa de aprendizaje
alpha = 0.6
# Factor de descuento que determina la importancia de las recompensas futuras
gamma = 0.99
# Parámetro epsilon para la política epsilon-greedy que controla la exploración vs la explotación
epsilon = 0.01
# Número total de episodios de entrenamiento
num_episodes = 10000
# Número máximo de pasos por episodio
max_steps = 1000

# Inicializar la Q-Table
Q = defaultdict(lambda: np.zeros(env.action_space.n))


# Función para elegir acción usando la política epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Entrenamiento del agente usando Q-learning
rewards = []
for episode in tqdm(range(num_episodes), desc="Episodios de entrenamiento"):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][best_next_action] - Q[state][action]
        )
        state = next_state
        total_reward += reward
        if done:
            break
    rewards.append(total_reward)

# Graficar las recompensas
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")
plt.show()
