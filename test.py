from mpi4py import MPI
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import pickle
import os

# Parámetros
alpha = 0.2  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.001  # Epsilon estático para la política epsilon-greedy
num_episodes = 1000  # Número de episodios
max_steps = 100
num_bins = 30  # Número de bins para discretización
q_table_filename = "Qtx.pkl"  # Nombre del archivo para guardar la tabla Q

# Discretizar el espacio de acciones (steer, gas, brake)
action_bins = [
    np.linspace(-1, 1, num_bins).astype(np.float16),
    np.linspace(0, 1, num_bins).astype(np.float16),
    np.linspace(0, 1, num_bins).astype(np.float16),
]


# Función para crear bins
def create_bins(low, high, num_bins):
    return [np.linspace(l, h, num_bins).astype(np.float16) for l, h in zip(low, high)]


# Crear bins para el espacio de estados
state_low = np.zeros((48, 48), dtype=np.float16).flatten()  # Reducido a 48x48
state_high = np.ones((48, 48), dtype=np.float16).flatten()
bins = create_bins(state_low, state_high, num_bins)


# Función para discretizar el estado
def discretize_state(state, bins):
    state = (
        state[::2, ::2].mean(axis=2).flatten().astype(np.float16)
    )  # Reducir resolución a 48x48
    return tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))


# Discretizar acción
def discretize_action(action):
    return tuple(np.digitize(a, b) - 1 for a, b in zip(action, action_bins))


# Obtener acción continua a partir de acción discretizada
def get_continuous_action(discretized_action):
    return np.array(
        [b[a] for a, b in zip(discretized_action, action_bins)], dtype=np.float16
    )


# Función para elegir acción usando la política epsilon-greedy
def epsilon_greedy_action(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return tuple(np.random.randint(bin_size) for bin_size in [num_bins] * 3)
    else:
        return np.unravel_index(np.argmax(Q[state]), (num_bins, num_bins, num_bins))


# Inicialización de la Q-Table
def initialize_q_table():
    return np.zeros((num_bins, num_bins, num_bins), dtype=np.float16)


# Guardar la tabla Q en un archivo
def save_q_table(Q, filename):
    with open(filename, "wb") as f:
        pickle.dump(dict(Q), f, protocol=pickle.HIGHEST_PROTOCOL)


# Cargar la tabla Q desde un archivo
def load_q_table(filename):
    if os.path.exists(f"./{filename}"):
        try:
            with open(filename, "rb") as f:
                return defaultdict(initialize_q_table, pickle.load(f))
        except (OSError, EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading Q-table: {e}. Initializing new Q-table.")
            os.remove(f"./{filename}")
            return defaultdict(initialize_q_table)
    else:
        return defaultdict(initialize_q_table)


# Función de entrenamiento Q-learning
def q_learning(
    env_name, num_episodes, alpha, gamma, epsilon, max_steps, bins, q_table_filename
):
    env = gym.make(env_name)
    Q = load_q_table(q_table_filename)
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes", leave=False):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        total_reward = 0
        done = False

        for _ in range(max_steps):
            action_discrete = epsilon_greedy_action(Q, state, epsilon)
            action = get_continuous_action(action_discrete)
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, bins)

            best_next_action = np.unravel_index(
                np.argmax(Q[next_state]), (num_bins, num_bins, num_bins)
            )
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action_discrete]
            Q[state][action_discrete] += alpha * td_delta

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")

    env.close()
    save_q_table(Q, q_table_filename)
    return dict(Q), rewards  # Convertir defaultdict a dict para serialización


# Función para combinar resultados
def combine_results(Q_list):
    combined_Q = defaultdict(initialize_q_table)
    num_results = len(Q_list)

    for Q in Q_list:
        for state, actions in Q.items():
            combined_Q[state] += actions

    for state in combined_Q:
        combined_Q[state] /= num_results

    return combined_Q


# Convertir defaultdict a dict con listas para MPI
def convert_q_table_for_serialization(Q):
    return {state: actions.tolist() for state, actions in Q.items()}


# Convertir dict con listas a defaultdict con np.array
def convert_q_table_from_serialization(Q_serialized):
    Q = defaultdict(initialize_q_table)
    for state, actions in Q_serialized.items():
        Q[state] = np.array(actions, dtype=np.float16)
    return Q


# Función para graficar las recompensas
def plot_rewards(rewards):
    sns.set_theme(style="darkgrid")
    rewards = np.array(rewards, dtype=np.float16)
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
    plt.show()


# Configuración de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    env_name = "CarRacing-v2"
    episodes_per_process = num_episodes // size
    Q_local, rewards_local = q_learning(
        env_name,
        episodes_per_process,
        alpha,
        gamma,
        epsilon,
        max_steps,
        bins,
        q_table_filename,
    )

    # Convertir Q_local a un formato serializable
    Q_local_serialized = convert_q_table_for_serialization(Q_local)

    # Añadir impresión de depuración
    print(f"Rank {rank} Q_local_serialized size: {len(Q_local_serialized)}")

    # Recopilación de las Q-Tables y recompensas
    Q_list_serialized = comm.gather(Q_local_serialized, root=0)
    rewards_list = comm.gather(rewards_local, root=0)

    if rank == 0:
        Q_list = [convert_q_table_from_serialization(Q) for Q in Q_list_serialized]
        combined_Q = combine_results(Q_list)
        combined_rewards = [reward for sublist in rewards_list for reward in sublist]
        plot_rewards(combined_rewards)
        save_q_table(combined_Q, q_table_filename)
