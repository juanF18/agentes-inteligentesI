import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Configuración del entorno
env = gym.make("LunarLander-v2", continuous=False)

# Parámetros del TD(0)
alpha = 0.2  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Tasa de exploración inicial
epsilon_decay = 0.0095  # Decaimiento de epsilon
epsilon_min = 0.1  # Valor mínimo de epsilon
num_episodes = 5000  # Número de episodios de entrenamiento

# Discretización del espacio de estados
state_bins = [
    np.linspace(-1.5, 1.5, 10),  # Posición X
    np.linspace(-1.5, 1.5, 10),  # Posición Y
    np.linspace(-3.0, 3.0, 10),  # Velocidad X
    np.linspace(-3.0, 3.0, 10),  # Velocidad Y
    np.linspace(-np.pi, np.pi, 10),  # Ángulo
    np.linspace(-5.0, 5.0, 10),  # Velocidad Angular
    np.array([0, 1]),  # Pierna izquierda contacto
    np.array([0, 1]),  # Pierna derecha contacto
]


def discretize_state(state):
    state_idx = []
    for i, s in enumerate(state):
        idx = np.digitize(s, state_bins[i]) - 1
        state_idx.append(idx)
    return tuple(state_idx)


# Inicialización de la tabla V
v_table = np.zeros([len(b) for b in state_bins])

# Intentar cargar la tabla V existente
try:
    with open("./TD-Learning/V_table_epsilon.pkl", "rb") as f:
        v_table = pickle.load(f)
    print("Tabla V cargada exitosamente.")
except FileNotFoundError:
    print("No se encontró una tabla V guardada, se inicializa una nueva.")


# Función de selección de acción usando la política epsilon-greedy
def epsilon_greedy_policy(state, epsilon):
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(v_table[state])
    else:
        return random.randrange(env.action_space.n)


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
    axs[1].set_ylabel("TD Error")
    axs[1].set_title("Learning Curve - Errors TD(0) epsilon-greedy")
    axs[1].legend()

    plt.savefig("./assets/Metrics_TD0_EpsilonGreedy.png")
    plt.show()


# Entrenamiento con TD(0)
rewards = []
errors = []

for episode in tqdm(range(num_episodes), desc="# Episodios"):
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    episode_error = 0

    done = False
    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Actualización de la tabla V
        td_error = reward + gamma * v_table[next_state] - v_table[state]
        v_table[state] += alpha * td_error

        state = next_state
        total_reward += reward
        episode_error += abs(td_error)

        # Terminar el episodio si la recompensa total es menor que el umbral
        if total_reward < -400:
            break

    # Decaimiento de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    rewards.append(total_reward)
    errors.append(episode_error)

    if (episode + 1) % 100 == 0:
        print(
            f"Episodio: {episode + 1}, Recompensa Total: {total_reward}, epsilon: {epsilon:.4f}"
        )

# Guardar la tabla V actualizada
with open("./TD-Learning/v_table.pkl", "wb") as f:
    pickle.dump(v_table, f)
    print("Tabla V guardada exitosamente.")

# Visualizar la curva de aprendizaje
plot_metrics(rewards, errors)
