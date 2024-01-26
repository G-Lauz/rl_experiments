import abc

from typing import Tuple

import torch

class Agent(abc.ABC, torch.nn.Module):
    def __init__(self, gamma, alpha, device="cpu"):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

        self.device = device
        self.to(device)

    @abc.abstractmethod
    def policy(self, state):
        pass

    @abc.abstractmethod
    def act(self, state) -> Tuple[int, float]:
        """
        Return an action and its log probability
        """


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.mlp(x)


class MLPPolicyAgent(Agent):
    def __init__(self, state_size, action_size, gamma, alpha, hidden_size=16, device="cpu"):
        super().__init__(gamma, alpha, device)

        self.mlp = MLP(state_size, action_size, hidden_size).to(device)

        self.apply(self.init_weights)

    def init_weights(self, net):
        if isinstance(net, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(net.weight)
            torch.nn.init.zeros_(net.bias)

    def policy(self, state):
        logits = self.mlp(state)
        policy = torch.functional.F.softmax(logits, dim=0) # softmax in action preferences eq 13.2
        return policy

    def act(self, state) -> Tuple[int, float]:
        policy = self.policy(state)
        action = torch.multinomial(policy, 1).item()
        log_probability = torch.log(policy[action])

        return action, log_probability
