import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Configuración del entorno
env = gym.make("LunarLander-v2", continuous=False)

# Parámetros del TD(0)
alpha = 0.2  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
tau = 6.0  # Parámetro de temperatura para Softmax
tau_decay = 0.995  # Decaimiento de tau
tau_min = 0.1  # Valor mínimo de tau
num_episodes = 5000  # Número de episodios de entrenamiento
reward_threshold = -400  # Umbral de recompensa para terminar el episodio temprano

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


# Inicialización de la tabla Q
v_table = np.zeros([len(b) for b in state_bins] + [env.action_space.n])

# Intentar cargar la tabla Q existente
try:
    with open("./TD-Learning/Q_table_softmax.pkl", "rb") as f:
        v_table = pickle.load(f)
    print("Tabla Q cargada exitosamente.")
except FileNotFoundError:
    print("No se encontró una tabla Q guardada, se inicializa una nueva.")


# Función de selección de acción usando la política Softmax
def softmax_policy(state, tau):
    q_values = v_table[state]
    e_x = np.exp(
        (q_values - np.max(q_values)) / tau
    )  # Subtracción para estabilidad numérica
    probabilities = e_x / np.sum(e_x)
    return np.random.choice(len(q_values), p=probabilities)


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
    axs[1].set_title("Learning Curve - Errors TD(0) Softmax")
    axs[1].set_ylim(0, max(avg_errors) + 10)  # Ajuste para evitar recortes
    axs[1].legend()

    plt.savefig("./assets/Metrics_TD0_Softmax.png")
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
    steps = 0  # Contador de pasos para monitoreo
    while (
        not done and steps < 1000
    ):  # Limitar los pasos por episodio para evitar bucles infinitos
        action = softmax_policy(state, tau)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Cálculo del objetivo TD
        td_target = reward + gamma * np.max(v_table[next_state])

        # Cálculo del error TD
        td_error = td_target - v_table[state][action]

        # Actualización de la tabla Q
        v_table[state][action] += alpha * td_error

        state = next_state
        total_reward += reward
        episode_error += abs(td_error)
        steps += 1

        # Terminar el episodio si la recompensa total es menor que el umbral
        if total_reward < reward_threshold:
            break

    # Decaimiento de tau
    if tau > tau_min:
        tau *= tau_decay

    rewards.append(total_reward)
    errors.append(episode_error)

    if (episode + 1) % 100 == 0:
        print(
            f"Episodio: {episode + 1}, Recompensa Total: {total_reward}, tau: {tau:.4f}, steps: {steps}"
        )

# Guardar la tabla Q actualizada
with open("./TD-Learning/v_table_softmax.pkl", "wb") as f:
    pickle.dump(v_table, f)
    print("Tabla Q guardada exitosamente.")

# Visualizar la curva de aprendizaje
plot_metrics(rewards, errors)
