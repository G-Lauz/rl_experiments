import gymnasium as gym
import numpy as np
import torch
import time


env = gym.make('CartPole-v1', render_mode='human')


class LinearPolicyAgent():
    def __init__(self, state_size, action_size, alpha=0.0001, gamma=1, device='cpu'):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.action_size = action_size
        self.state_size = state_size
        self.device = device

        self.reset()

    def reset(self):
        self.weights = torch.rand((self.action_size, self.state_size), device=self.device)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        probabilities = torch.matmul(self.weights, state.T) # compute action preferences eq 13.3
        policy = torch.functional.F.softmax(probabilities, dim=0).T # softmax in action preferences eq 13.2

        action_idx = torch.multinomial(policy, 1).item() # sample action from policy

        return action_idx

    def update_policy(self, weights):
        self.weights = weights


def reinforce(environment, agent, n_episodes, max_steps, render=False, device='cpu'):
    total_rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        print(f"Episode {i+1}/{n_episodes}")

        rewards_history = []
        action_history = []
        state_history = []

        state = env.reset()[0]

        # Generate an episode S0, A0, R1, ..., ST-1, AT-1, RT, following pi(.|., theta)
        for j in range(max_steps):
            message = f"Step {j+1}/{max_steps}"
            print(message, end="\r", flush=True)

            if render:
                env.render()
                time.sleep(0.01)

            action_idx = agent.act(state)

            next_state, reward, done, _, _ = env.step(action_idx)

            action_history.append(action_idx)
            state_history.append(state)
            rewards_history.append(reward)

            state = next_state

            if done:
                break

        total_rewards[i] = sum(rewards_history)

        # Update policy parameters, eq. 13.8
        for t in range(len(state_history))[::-1]: # exclude the goal state from the update
            state = torch.from_numpy(state_history[t]).float().unsqueeze(0).to(device)
            action_idx = action_history[t]

            # compute the return following time t
            gt = agent.gamma ** np.arange(len(rewards_history[t:])) * np.array(rewards_history[t:])
            gt = agent.gamma ** t * np.sum(gt)

            # compute the gradient of the log policy
            probabilities = torch.matmul(agent.weights, state.T) # compute action preferences eq 13.3
            policy = torch.functional.F.softmax(probabilities, dim=0) # softmax in action preferences eq 13.2
            current_features = state[:, action_idx]
            grad_log_pi = current_features - torch.dot(policy[action_idx], current_features)

            # update the policy parameters
            agent.update_policy(agent.weights + agent.alpha * agent.gamma ** t * gt * grad_log_pi)

    return total_rewards


def main():
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    linear_policy_agent = LinearPolicyAgent(state_size, action_size, alpha=0.01, gamma=1)
    scores = reinforce(env, linear_policy_agent, n_episodes=1000, max_steps=1000, render=True)

    print(f"Average score over 1000 episodes: {np.mean(scores)}")


if __name__ == '__main__':
    main()
