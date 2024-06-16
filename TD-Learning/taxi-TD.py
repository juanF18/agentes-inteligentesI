import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pickle


# Hiperparámetros
alpha = 0.001  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento que determina la importacion de recompensas futuras
epsilon = 0.01  # Parámetro epsilon para la política epsilon-greedy que controla la exploracicon vs la explotacion
num_episodes = 1000  # Numero total de episodios de entrenar
max_steps = 500  # Numero maximo de pasos por episodio
# num_bins = 10  # Numero de bisn para discretizar, cada dimension del espacio de estados

# Crear el entorno
"""
    Crea una instancia del entorno, un entorno
    de control donde el bojetivo es balancear 
    un brazo doble invertido
"""
env = gym.make("Taxi-v3")


# Inicializar la Q-Table o cargar una anterior
# diccionario que almacena los valores de Q para cada par
# estado-accion
Q = None
try:
    with open("Qtx.pkl", "rb") as f:
        Q_dict = pickle.load(f)
    Q = defaultdict(lambda: np.zeros(env.action_space.n), Q_dict)
    # Q = np.load("Qtx.npy")
except Exception as e:
    print(e)

if Q == None:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # Q = np.zeros([env.observation_space.n, env.action_space.n])


# Función para elegir acción usando la política epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Entrenamiento del agente usando Q-learning
rewardsEpoch = []


def taxi():
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
        rewardsEpoch.append(total_reward)

    # Convertir el defaultdict a un diccionario normal antes de guardarlo
    Q_dict = dict(Q)
    # Guardar el defaultdict usando pickle
    with open("Qtx.pkl", "wb") as f:
        pickle.dump(Q_dict, f)


taxi()

# Graficar las recompensas
plt.plot(range(num_episodes), rewardsEpoch)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")
plt.show()
