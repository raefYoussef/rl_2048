# DQN agent designed using code from the Torch tutorial here:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Adjustments made to all this to more easily be used with the 2048 environment we wrote

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .dqn_helpers import Transition, ReplayBuffer


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        # TODO wondering if it would be worth try to add a convolutional layer for one experiment?
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class BasicDQN:
    def __init__(
        self,
        dqn_module=DQN,
        torch_device=None,
        observations_in_state=9,
        num_actions=4,
        buffer_size=10000,
        batch_size=128,
        e_start=0.9,
        e_end=0.05,
        e_decay=1000,
        discount=0.99,
        learn_rate=1e-4,
        update_rate=.0005,
        debug=False,
    ):
        self.reward = 0
        self.terminal = 0
        self.state = []

        self.observations_in_state = observations_in_state
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.e_start = e_start
        self.e_end = e_end
        self.e_decay = e_decay  # num steps needed before we only use e_end
        self.discount = discount
        self.learn_rate = learn_rate
        self.update_rate = update_rate
        self.steps_done = 0
        self.replay_buffer = ReplayBuffer(buffer_size)
        if torch_device:
            self.device = torch_device
        else:
            # TODO here is where I could swap for torch_directml
            print("Setting device to 'default' as none was provided")
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if debug:
            self.rng = np.random.default_rng(seed=100)
        else:
            self.rng = np.random.default_rng(seed=None)

        self.policy_net = dqn_module(observations_in_state, num_actions).to(self.device)
        self.target_net = dqn_module(observations_in_state, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learn_rate, amsgrad=True
        )

    def reset_game(self, initial_state):
        self.state = initial_state
        self.terminal = False

    # TODO Maybe switch this to softmax? -- or test different values for the epsilon decay
    def epsilon_decay(self, state):
        # epsilon decay
        sample = self.rng.random()
        epsilon = self.e_end + (self.e_start - self.e_end) * np.exp(
            -1.0 * self.steps_done / self.e_decay
        )
        self.steps_done += 1
        if sample > epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.rng.integers(0, 4)]], device=self.device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.discount
        ) + reward_batch

        # TODO decide if we want a different loss function
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()# .to(self.device)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def act_and_learn(self, state, env):
        action = self.select_action(state)
        stat_obs, reward, score, end, win = env.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        if end:
            next_state = None
        else:
            next_state = torch.tensor(stat_obs.flatten(), dtype=torch.float, device=self.device).unsqueeze(0)

        # Store the transition in memory
        self.replay_buffer.push(state, action, next_state, reward)
        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.update_rate + target_net_state_dict[key]*(1-self.update_rate)
        self.target_net.load_state_dict(target_net_state_dict)

        if next_state is None:
            high_tile = torch.max(state)
        else:
            high_tile = torch.max(next_state)
        return state, action, next_state, reward, score, high_tile, end, win

    def select_action(self, state):
        if self.terminal:
            print(f"Currently in terminal state {self.state}, can't act")
            return -1
        action = self.epsilon_decay(state)
        return action

    def update_state(self, new_state, reward, is_terminal=False):
        self.last_state = self.state
        self.state = new_state
        self.reward = reward
        self.terminal = is_terminal
