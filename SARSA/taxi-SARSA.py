import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pickle
import seaborn as sns


# Hiperparámetros
alpha = 0.001  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento que determina la importacion de recompensas futuras
epsilon = 0.01  # Parámetro epsilon para la política epsilon-greedy que controla la exploracicon vs la explotacion
num_episodes = 10000  # Numero total de episodios de entrenar
max_steps = 1000  # Numero maximo de pasos por episodio
# num_bins = 10  # Numero de bisn para discretizar, cada dimension del espacio de estados
method = ["Q-learning", "SARSA", "T(0)"]
heat = 0.5

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
    with open("Q_SARSAsoft.pkl", "rb") as f:
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
errorsEpoch = []


# Funcion de entrenamiento en el ambiente de taxi
def taxi(soft: bool):
    for episode in tqdm(range(num_episodes), desc="Episodios de entrenamiento"):
        state, _ = env.reset()
        total_reward = 0
        total_errors = 0
        steps = 0
        for step in range(max_steps):
            action = None
            if soft:
                action = softmax(Q[state], heat)
            else:
                action = choose_action(state)

            next_state, reward, done, _, _ = env.step(action)
            steps += 1
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][best_next_action] - Q[state][action]
            )

            td_error = (
                reward + gamma * Q[next_state][best_next_action] - Q[state][action]
            )
            Q[state][action] += alpha * td_error
            state = next_state
            total_reward += reward
            total_errors += abs(td_error)

            if done:
                break
        rewardsEpoch.append(total_reward)
        errorsEpoch.append(total_errors / steps)

    # Convertir el defaultdict a un diccionario normal antes de guardarlo
    Q_dict = dict(Q)
    # Guardar el defaultdict usando pickle
    with open("Q_SARSAsoft.pkl", "wb") as f:
        pickle.dump(Q_dict, f)


def plot_metrics(rewards, errors):
    sns.set_theme(style="darkgrid")
    rewards = np.array(rewards)
    errors = np.array(errors)
    block_size = 100  # Aumentar el tamaño del bloque para suavizar más la curva
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
    avg_errors = np.mean(
        errors[: num_blocks * block_size].reshape(-1, block_size), axis=1
    )

    x = np.arange(1, len(avg_rewards) + 1) * block_size

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # Gráfica de Recompensas
    axs[0].plot(x, avg_rewards, label="Average Reward", color="blue")
    axs[0].plot(x, max_rewards, label="Max Reward", color="orange", linestyle="--")
    axs[0].plot(x, min_rewards, label="Min Reward", color="green", linestyle=":")
    axs[0].fill_between(x, min_rewards, max_rewards, color="blue", alpha=0.1)
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Learning Curve - Rewards")
    axs[0].legend()

    # Gráfica de Errores TD
    axs[1].plot(x, avg_errors, label="Average TD Error", color="red")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Error")
    axs[1].set_title("Learning Curve - Errors SARSA ")
    axs[1].legend()

    plt.savefig(f"gph/SARSA/{method[1]}_al{alpha}_ga{gamma}_ep{epsilon}.png")
    plt.show()


# llamamos la función
taxi(soft=True)
plot_metrics(rewardsEpoch, errorsEpoch)

# Graficar las recompensas
# plt.plot(range(num_episodes), rewardsEpoch)
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Rewards vs Episodes")
# plt.legend()
# plt.savefig(f"gph/{method[1]}_{alpha}_{gamma}_{epsilon}.png")
# plt.show()