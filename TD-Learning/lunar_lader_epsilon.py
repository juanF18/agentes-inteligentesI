import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Definir el entorno
env = gym.make("LunarLander-v2")


# Definir función para serializar estados
def serialize_state(state):
    x, y, vx, vy, theta, vtheta, left_contact, right_contact = state
    x_discrete = np.clip(int(np.digitize(x, np.linspace(-1, 1, 10))), 0, 9)
    y_discrete = np.clip(int(np.digitize(y, np.linspace(-1, 1, 10))), 0, 9)
    vx_discrete = np.clip(int(np.digitize(vx, np.linspace(-1, 1, 10))), 0, 9)
    vy_discrete = np.clip(int(np.digitize(vy, np.linspace(-1, 1, 10))), 0, 9)
    theta_discrete = np.clip(int(np.digitize(theta, np.linspace(-1, 1, 10))), 0, 9)
    vtheta_discrete = np.clip(int(np.digitize(vtheta, np.linspace(-1, 1, 10))), 0, 9)
    left_contact_discrete = int(left_contact)
    right_contact_discrete = int(right_contact)

    state_discrete = (
        x_discrete,
        y_discrete,
        vx_discrete,
        vy_discrete,
        theta_discrete,
        vtheta_discrete,
        left_contact_discrete,
        right_contact_discrete,
    )

    return state_discrete


def plot_metrics(rewards, td_errors):
    sns.set_theme(style="darkgrid")
    rewards = np.array(rewards)
    td_errors = np.array(td_errors)
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
    avg_td_errors = np.mean(
        td_errors[: num_blocks * block_size].reshape(-1, block_size), axis=1
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
    axs[1].plot(x, avg_td_errors, label="Average TD Error", color="red")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("TD Error")
    axs[1].set_title("Learning Curve - TD Errors")
    axs[1].legend()

    plt.savefig("./assets/Metrics_TD0.png")
    plt.show()


# Inicializar la tabla Q con valores aleatorios o cargar una tabla existente
state_bins = [10] * 8  # Número de bins por dimensión del estado
num_actions = env.action_space.n  # Número de acciones
Q = np.zeros(state_bins + [num_actions])

try:
    with open("./TD-Learning/QLunarTD0.pkl", "rb") as f:
        Q_loaded = pickle.load(f)
    print("Tabla Q cargada exitosamente.")
    Q = Q_loaded
except FileNotFoundError:
    print("No se encontró una tabla Q guardada, se inicializa una nueva.")

# Parámetros de aprendizaje
alpha = 0.3  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.0001  # Probabilidad de exploración


# Función para seleccionar acciones usando la estrategia Epsilon Greedy
def choose_action(state):
    if random.random() > epsilon:
        return np.argmax(Q[state])
    else:
        return random.randrange(env.action_space.n)


# Almacenar las recompensas y errores TD por episodio
rewards = []
td_errors = []

# Entrenamiento del agente
num_episodios = 10000  # Número de episodios de entrenamiento
max_steps_per_episode = 200  # Número máximo de pasos por episodio

for episode in tqdm(range(num_episodios), desc="# Episodios"):
    state = serialize_state(env.reset()[0])
    done = False
    steps = 0
    total_reward = 0
    episode_td_error = 0

    while not done and steps < max_steps_per_episode:
        action = choose_action(state)
        new_state, reward, done, _, _ = env.step(action)
        new_state_discrete = serialize_state(new_state)

        # Actualizar la tabla Q utilizando la ecuación de TD(0)
        best_next_action = np.argmax(Q[new_state_discrete])
        td_error = (
            reward + gamma * Q[new_state_discrete][best_next_action] - Q[state][action]
        )
        Q[state][action] += alpha * td_error

        state = new_state_discrete
        steps += 1
        total_reward += reward
        episode_td_error += abs(td_error)

    rewards.append(total_reward)
    td_errors.append(episode_td_error / steps)  # Error TD promedio por episodio

# Guardar la tabla Q actualizada
with open("./TD-Learning/QLunarTD0.pkl", "wb") as f:
    pickle.dump(Q, f)
    print("Tabla Q guardada exitosamente.")


plot_metrics(rewards, td_errors)
