import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Hiperparámetros
alpha = 0.6  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento que determina la importacion de recompensas futuras
epsilon = 0.001  # Parámetro epsilon para la política epsilon-greedy que controla la exploracicon vs la explotacion
num_episodes = 10000  # Numero total de episodios de entrenar
max_steps = 1000  # Numero maximo de pasos por episodio
# num_bins = 10  # Numero de bisn para discretizar, cada dimension del espacio de estados
method = ["Q-learning", "SARSA", "T(0)"]

# Crear el entorno
"""
    Crea una instancia del entorno, un entorno
    de control donde el bojetivo es balancear 
    un brazo doble invertido
"""
env = gym.make("Taxi-v3", render_mode="human")


# Inicializar la Q-Table o cargar una anterior
# diccionario que almacena los valores de Q para cada par
# estado-accion
Q = None
try:
    Q = np.load("Q-learning.npy")
except Exception as e:
    print(e)

if not np.any(Q):
    Q = np.zeros([env.observation_space.n, env.action_space.n])


# Función para elegir acción usando la política epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Entrenamiento del agente usando Q-learning
rewardsEpoch = []


# Funcion de entrenamiento en el ambiente de taxi
def taxi():
    for i in tqdm(range(num_episodes)):
        observacion, info = env.reset()
        terminated = False
        totalRewards = 0
        while not terminated:
            action = choose_action(observacion)
            next_observation, reward, terminated, truncated, info = env.step(action)
            totalRewards += reward

            Q[observacion, action] += alpha * (
                reward + gamma * np.max(Q[next_observation, :]) - Q[observacion, action]
            )
            observacion = next_observation
        rewardsEpoch.append(totalRewards)

    np.save("Q-learning.npy", Q)


# llamamos la función
taxi()

# Graficar las recompensas
plt.plot(range(num_episodes), rewardsEpoch)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")
plt.legend()
plt.savefig(f"gph/{method[0]}_a{alpha}_g{gamma}_e{epsilon}.png")
plt.show()
