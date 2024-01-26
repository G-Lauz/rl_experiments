import gymnasium
import matplotlib.pyplot as plt
import numpy
import torch
import time

from reinforce import reinforce
from agent import MLPPolicyAgent


def evaluate(agent, environment, n_episodes, max_steps, render=False):
    agent.eval()

    with torch.no_grad():
        episodes_rewards = []

        for i in range(n_episodes):
            state = environment.reset()[0]

            episode_reward = 0
            for _ in range(max_steps):
                if render:
                    environment.render()
                    time.sleep(0.01)

                state = torch.from_numpy(state).float().to(agent.device)
                action, _ = agent.act(state)

                state, reward, done, _, _ = environment.step(action)

                episode_reward += reward

                if done:
                    break

            print(f"[{i+1}/{n_episodes}] Episode reward: {episode_reward:.2f}")
            episodes_rewards.append(episode_reward)

            mean_reward = numpy.mean(episodes_rewards)
            std_reward = numpy.std(episodes_rewards)

        return mean_reward, std_reward


def plot_reward_history(reward_history):
    plt.figure()

    plt.plot(reward_history)

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.tight_layout()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_environment = gymnasium.make("CartPole-v1")

    state_size = training_environment.observation_space.shape[0]
    action_size = training_environment.action_space.n

    agent = MLPPolicyAgent(state_size, action_size, gamma=0.99, alpha=0.0001, device=device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=agent.alpha)

    print("_____TRAINING_____")
    reward_history = reinforce(
        agent,
        training_environment,
        optimizer,
        n_episodes=1000,
        max_steps=1000,
        verbose=True
    )
    plot_reward_history(reward_history)
    training_environment.close()
    print("__________________\n")

    torch.save(agent.state_dict(), "mlp_agent_cartpole_checkpoint.pth")

    print("_____TESTING_____")
    testing_environment = gymnasium.make("CartPole-v1", render_mode="human")
    mean, std = evaluate(
        agent,
        testing_environment,
        n_episodes=10,
        max_steps=1000,
        render=True
    )
    print(f"Mean reward: {mean:.2f} +/- {std:.2f}")
    testing_environment.close()
    print("__________________\n")

    plt.show()


if __name__ == "__main__":
    main()
