import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


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


def softmax(q_values, tau):
    preferences = q_values / tau
    max_preference = np.max(preferences)
    exp_preferences = np.exp(
        preferences - max_preference
    )  # Subtract max_preference for numerical stability
    probabilities = exp_preferences / np.sum(exp_preferences)
    return np.random.choice(len(q_values), p=probabilities)


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


def plot_rewards(rewards):
    sns.set_theme(style="darkgrid")
    rewards = np.array(rewards)
    block_size = 100
    num_blocks = len(rewards) // block_size
    avg_rewards = np.mean(
        rewards[: num_blocks * block_size].reshape(-1, block_size), axis=1
    )
    max_rewards = np.max(
        rewards[: num_blocks * block_size].reshape(-1, block_size), axis=1
    )
    min_rewards = np.min(
        rewards[: num_blocks * block_size].reshape(-1, block_size), axis=1
    )
    x = np.arange(1, len(avg_rewards) + 1) * block_size

    plt.figure(figsize=(10, 6))
    plt.plot(x, avg_rewards, label="Average Reward", color="blue")
    plt.plot(x, max_rewards, label="Max Reward", color="orange", linestyle="--")
    plt.plot(x, min_rewards, label="Min Reward", color="green", linestyle=":")
    plt.fill_between(x, min_rewards, max_rewards, color="blue", alpha=0.1)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(f"gph/{method[0]}_a{alpha}_g{gamma}_e{epsilon}.png")
    plt.show()


# llamamos la función
taxi()
plot_rewards(rewardsEpoch)

# Graficar las recompensas
# plt.plot(range(num_episodes), rewardsEpoch)
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Rewards vs Episodes")
# plt.legend()
# plt.savefig(f"gph/{method[0]}_a{alpha}_g{gamma}_e{epsilon}.png")
# plt.show()
