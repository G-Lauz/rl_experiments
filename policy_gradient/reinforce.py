import time
import numpy
import torch

from collections import deque

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


def reinforce_hf(policy, env, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = numpy.finfo(numpy.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, numpy.mean(scores_deque)))

    return scores
