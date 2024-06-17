import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pickle
import seaborn as sns


# Hiperparámetros
alpha = 0.6  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento que determina la importacion de recompensas futuras
epsilon = 0.01  # Parámetro epsilon para la política epsilon-greedy que controla la exploracicon vs la explotacion
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
env = gym.make("Taxi-v3")


# Inicializar la Q-Table o cargar una anterior
# diccionario que almacena los valores de Q para cada par
# estado-accion
Q = None
try:
    with open("Q_SARSA.pkl", "rb") as f:
        Q_dict = pickle.load(f)
    Q = defaultdict(lambda: np.zeros(env.action_space.n), Q_dict)
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
    with open("Q_SARSA.pkl", "wb") as f:
        pickle.dump(Q_dict, f)


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
    plt.savefig(f"gph/{method[1]}_{alpha}_{gamma}_{epsilon}.png")
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
# plt.savefig(f"gph/{method[1]}_{alpha}_{gamma}_{epsilon}.png")
# plt.show()
