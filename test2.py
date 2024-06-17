import os
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from collections import defaultdict

# Definir el entorno
env = gym.make("LunarLander-v2")

# Parámetros de aprendizaje
alpha = 0.3  # Tasa de aprendizaje ajustada
gamma = 0.99  # Factor de descuento ajustado
epsilon = 0.1  # Probabilidad de exploración
epsilon_decay = 0.995  # Decaimiento de epsilon por episodio
min_epsilon = 0.01  # Epsilon mínimo
num_episodios = 5000  # Número de episodios de entrenamiento
max_steps_per_episode = 300  # Número máximo de pasos por episodio
q_table_filename = "q_table.pkl"  # Nombre del archivo para guardar la tabla Q


def initialize_q_table():
    return np.zeros(env.action_space.n)


def serialize_state(state):
    x, y, vx, vy, theta, vtheta, left_contact, right_contact = state
    x_discrete = np.clip(int(np.digitize(x, np.linspace(-1, 1, 20))), 0, 19)
    y_discrete = np.clip(int(np.digitize(y, np.linspace(-1, 1, 20))), 0, 19)
    vx_discrete = np.clip(int(np.digitize(vx, np.linspace(-1, 1, 20))), 0, 19)
    vy_discrete = np.clip(int(np.digitize(vy, np.linspace(-1, 1, 20))), 0, 19)
    theta_discrete = np.clip(int(np.digitize(theta, np.linspace(-1, 1, 20))), 0, 19)
    vtheta_discrete = np.clip(int(np.digitize(vtheta, np.linspace(-1, 1, 20))), 0, 19)
    left_contact_discrete = int(left_contact)
    right_contact_discrete = int(right_contact)
    return (
        x_discrete,
        y_discrete,
        vx_discrete,
        vy_discrete,
        theta_discrete,
        vtheta_discrete,
        left_contact_discrete,
        right_contact_discrete,
    )


def plot_rewards(rewards):
    sns.set_theme(style="darkgrid")
    rewards = np.array(rewards)
    block_size = 100  # Tamaño del bloque para suavizar la curva
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


def save_q_table(Q, filename):
    with open(filename, "wb") as f:
        pickle.dump(dict(Q), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_q_table(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return defaultdict(initialize_q_table, pickle.load(f))
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading Q-table: {e}. Initializing new Q-table.")
            return defaultdict(initialize_q_table)
    else:
        return defaultdict(initialize_q_table)


def combine_q_tables(Q1, Q2):
    combined_Q = defaultdict(initialize_q_table)
    for state, actions in Q1.items():
        if state in Q2:
            combined_Q[state] = (Q1[state] + Q2[state]) / 2
        else:
            combined_Q[state] = Q1[state]

    for state, actions in Q2.items():
        if state not in Q1:
            combined_Q[state] = Q2[state]

    return combined_Q


def train_agent(
    env,
    num_episodes,
    max_steps,
    alpha,
    gamma,
    epsilon,
    epsilon_decay,
    min_epsilon,
    q_table_filename,
):
    Q = load_q_table(q_table_filename)
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = serialize_state(env.reset()[0])
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < max_steps:
            if random.random() > epsilon:
                action = np.argmax(Q[state])
            else:
                action = random.randrange(env.action_space.n)

            new_state, reward, done, _, _ = env.step(action)
            new_state_discrete = serialize_state(new_state)

            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * np.max(Q[new_state_discrete]) - Q[state][action]
            )

            state = new_state_discrete
            steps += 1
            total_reward += reward

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        rewards.append(total_reward)

        if episode % 100 == 0:
            save_q_table(Q, q_table_filename)

    save_q_table(Q, q_table_filename)
    return Q, rewards


def evaluate_agent(env, Q, num_eval_episodes, max_steps):
    total_reward = 0

    for episode in range(num_eval_episodes):
        state = serialize_state(env.reset()[0])
        done = False

        while not done:
            action = np.argmax(Q[state])
            new_state, reward, done, _, _ = env.step(action)
            new_state_discrete = serialize_state(new_state)
            total_reward += reward
            state = new_state_discrete

    average_reward = total_reward / num_eval_episodes
    print("Recompensa promedio:", average_reward)
    return average_reward


if __name__ == "__main__":
    Q_old = load_q_table(q_table_filename)
    Q_new, rewards = train_agent(
        env,
        num_episodios,
        max_steps_per_episode,
        alpha,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
        q_table_filename,
    )
    Q_combined = combine_q_tables(Q_old, Q_new)
    save_q_table(Q_combined, q_table_filename)
    average_reward = evaluate_agent(env, Q_combined, 10, max_steps_per_episode)
    plot_rewards(rewards)
