import time
import torch

from agent import Agent


def generate_episode(agent: Agent, environment, max_steps, render=False):
    state_history = []
    action_history = []
    reward_history = []
    policy_history = []

    state = environment.reset()[0]

    for _ in range(max_steps):
        if render:
            environment.render()
            time.sleep(0.01)

        state = torch.from_numpy(state).float().to(agent.device)
        action, policy = agent.act(state)

        state, reward, done, _, _ = environment.step(action)

        state_history.append(state)
        action_history.append(action)
        reward_history.append(reward)
        policy_history.append(policy)

        if done:
            break

    return state_history, action_history, reward_history, policy_history


def reinforce(agent: Agent, environment, optimizer: torch.optim.Optimizer, n_episodes, max_steps, verbose=False):
    """
    REINFORCE algorithm for policy gradient from Sutton and Barto, 2018.
    
    Args:
        agent: a differentiable policy parameterization pi(a|s, theta)
        environment: an environment with discrete actions
        n_episodes: number of episodes to run
        max_steps: maximum number of steps per episode
        device: device to use for computations
    """
    cumulative_rewards = []

    for i in range(n_episodes):
        # Generate an episode S0, A0, R1, ..., ST-1, AT-1, RT, following pi(.|., theta)
        state_history, action_history, reward_history, log_probability = generate_episode(
            agent, environment, max_steps
        )

        episode_reward = sum(reward_history)
        cumulative_rewards.append(episode_reward)

        if verbose and (i + 1) % 100 == 0:
            print(f"[{i+1}/{n_episodes}] Episode - Reward: {episode_reward:.2f}")

        # For each step of the episode t = 0, 1, ..., T-1:
        n_step = len(state_history)
        for t in range(n_step)[::-1]: # exclude the goal state from the update
            # Compute Gt = R_t+1 + gamma * R_t+2 + ... + gamma^(T-1-t) * R_T
            discount_factors = agent.gamma ** torch.arange(n_step - t)
            gt = torch.sum(torch.tensor(reward_history[t:]) * discount_factors)

            optimizer.zero_grad() # reset the gradients

            # Compute the gradient of the log policy
            state = torch.from_numpy(state_history[t]).float().to(agent.device)
            policy = agent.policy(state)
            action = torch.multinomial(policy, 1).item()
            log_pi = torch.log(policy[action])

            # Update the policy parameters
            loss = -log_pi * gt # eq. 13.8 (note the minus sign account for the gradient ascent because of PyTorch's optimizer)
            loss.backward()
            optimizer.step()

    return cumulative_rewards