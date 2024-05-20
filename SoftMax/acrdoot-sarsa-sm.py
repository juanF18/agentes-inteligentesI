import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Crear el entorno
"""
    Crea una instancia del entorno, un entorno
    de control donde el bojetivo es balancear 
    un brazo doble invertido
"""
env = gym.make("Acrobot-v1", render_mode="human")


# Función para discretizar el espacio de estados
"""
    Convierte el estado continuo en un estado discreto
    State: Vector de estado continuo proporcionado por el entorno
    bins : Lista de bins usados para discretitzar cada dimension del estado
    np.digitize: Determian en que bin cae cada valor del estado
    - 1: Ajusta para que los indices comiencen desde 0

"""


def discretize_state(state, bins):
    state_disc = []
    for i in range(len(state)):
        state_disc.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_disc)


# Función para crear bins de discretización
"""
    Crea los bins necesario para discretizar el espacio de los
    estados.
    num_bins: Numero de bins por dimension
    lower_bound y upper_bounds: Limites inferiores y superiores
    del espacio de estados
    np.linspace(l, u, num_bins) : Divide el rango entre 'l' y 'u'
    en 'num_bins' partes iguales
"""


def create_bins(num_bins, lower_bounds, upper_bounds):
    bins = []
    for l, u in zip(lower_bounds, upper_bounds):
        bins.append(np.linspace(l, u, num_bins))
    return bins


# Parámetros
alpha = 0.01  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento que determina la importacion de recompensas futuras
num_episodes = 1000  # Numero total de episodios de entrenar
max_steps = 500  # Numero maximo de pasos por episodio
num_bins = 10  # Numero de bisn para discretizar, cada dimension del espacio de estados

# Crear bins para la discretización
"""
Limites del espacio de estados del entorno 
"""
lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
# Ajustar límites superiores para las velocidades
# Evita valores infinitos
upper_bounds[1] = 1
upper_bounds[3] = 1
# Se crean los bins con los limites ajustados
bins = create_bins(num_bins, lower_bounds, upper_bounds)

# Inicializar la Q-Table
# diccionario que almacena los valores de Q para cada par
# estado-accion
Q = defaultdict(lambda: np.zeros(env.action_space.n))


# Función para elegir acción usando la política softmax
def choose_action(state):
    q_values = Q[state]
    probabilities = softmax(q_values)
    action = np.random.choice(len(q_values), p=probabilities)
    return action


# Función de SoftMax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Entrenamiento del agente usando SARSA
rewards = []
for episode in tqdm(range(num_episodes), desc="Episodios de entrenamiento"):
    state, _ = env.reset()
    state = discretize_state(state, bins)
    action = choose_action(state)
    total_reward = 0
    for step in range(max_steps):
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state, bins)
        next_action = choose_action(next_state)
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )
        state = next_state
        action = next_action
        total_reward += reward
        print(total_reward)
        if done:
            break
    rewards.append(total_reward)
# Graficar las recompensas
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")
plt.show()
