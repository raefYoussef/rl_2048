from pathlib import Path
import typing
import numpy.typing as npt
import numpy as np
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.distributions import Categorical


class AgentPPO:
    """
    PPO RL Agent to Play 2048
    """

    def __init__(
        self,
        env,
        policy,
        seed: typing.Optional[float] = None,
        lr: float = 0.005,
        gamma: float = 0.95,
        clip: float = 0.2,
        num_updates: int = 5,
        max_eps_moves: int = 1600,
        max_batch_moves: int = 2048,
        actor_path: typing.Union[Path, str] = "./model/ppo_actor.pth",
        critic_path: typing.Union[Path, str] = "./model/ppo_critic.pth",
        save_freq: int = 10,
    ) -> None:
        """
        Initializes the PPO model, including hyperInputs.

        Inputs:
            env:                Environment class (e.g. Env2048)
            policy:             Policy class to use for actor/critic networks (e.g. PolicyMLP)
            seed:               RNG seed (default None, doesn't init RNG)
            lr:                 Actor/Critic Learning rate (default 0.005)
            gamma:              Discount factor to be applied when calculating Rewards-To-Go (default 0.95)
            clip:               PPO clipping factor (default .2)
            num_updates:        Number of updates per iteration (default: 5)
            max_eps_moves:      Maximum number of moves per episode (default 1600)
            max_batch_moves:    Maximum number of moves per batch (default 2400)
            actor_path:         Path to actor weights (default: ./model/ppo_actor.pth)
            critic_path:        Path to critic weights (default: ./model/ppo_critic.pth)
            save_freq:          Iterations interval to save model weights (default: 10)

        Returns:
            None
        """

        # Initialize default values for hyperInputs
        # PPO Configs
        self.seed = seed
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.num_updates = num_updates
        self.max_batch_moves = max_batch_moves
        self.max_eps_moves = max_eps_moves

        # Model Configs
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.save_freq = save_freq

        # Init RNG
        if self.seed != None:
            # Check if our seed is valid first
            assert type(self.seed) == int

            # Set the seed
            torch.manual_seed(self.seed)

        # Extract environment information
        self.env = env
        self.state_dim = env.get_state_dim()
        self.action_dim = env.get_action_dim()

        # Initialize actor and critic networks
        self.actor = policy(self.state_dim, self.action_dim)
        self.critic = policy(self.state_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.Inputs(), lr=self.lr)
        self.critic_optim = Adam(self.critic.Inputs(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "move_cnt": 0,  # moves so far
            "iter_cnt": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rewards": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

    def get_action(
        self, state: typing.Union[npt.NDArray[np.float_], Tensor]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Queries an action from the actor network

        Inputs:
            state:      The state at the current time step

        Outputs:
            action:     The action to take
            log_prob:   The log probability of the selected action in the distribution
        """

        # Query the actor network for the policy (prob of selecting each action)
        policy = self.actor(state)

        # Sample an action from the distribution
        dist = Categorical(policy)
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return (action.detach().numpy(), log_prob.detach().numpy())

    def get_value(
        self, states: Tensor, actions: Tensor
    ) -> typing.Tuple[Tensor, Tensor]:
        """
        Get the value function of each state

        Inputs:
            states:     The states from the most recently collected batch as a tensor.
                        Shape: (dimension of batch, dimension of state)
            actions:    The actions from the most recently collected batch as a tensor.
                        Shape: (dimension of batch, dimension of action)

        Return:
            V:          The predicted value for each state
            log_probs:  The log probabilities of the actions taken in givens each (state, action) pair
        """
        # Query critic network for the state value, V(s)
        V = self.critic(states).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        policy = self.actor(states)
        dist = Categorical(policy)
        log_probs = dist.log_prob(actions)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return (V, log_probs)

    def compute_rtgs(self, batch_rewards: Tensor) -> Tensor:
        """
        Compute the Reward-To-Go of each move in a batch given the rewards.

        Inputs:
            batch_rewards:  The rewards in a batch
                            Shape: (number of episodes, number of moves per episode)

        Outputs:
            batch_rtgs: The rewards to go
                        Shape: (number of moves in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num moves per episode)
        batch_rtgs = []

        # Iterate through each episode
        for eps_rewards in reversed(batch_rewards):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for reward in reversed(eps_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def collect_batch(
        self,
    ) -> typing.Tuple[Tensor, Tensor, Tensor, Tensor, typing.List]:
        """
        Collect the batch of data from simulation.
        Since this is an on-policy algorithm, we'll need to collect a fresh
        batch of data each time we iterate the actor/critic networks.

        Inputs:
            None

        Outputs:
            batch_states:       The states collected this batch. Shape: (batch dim, state dim)
            batch_actions:      The actions collected this batch. Shape: (batch dim, action dim)
            batch_log_probs:    The log probabilities of each action taken this batch. Shape: (batch dim)
            batch_rtgs:         The Rewards-To-Go of each move in this batch. Shape: (batch dim)
            batch_lens:         The lengths of each episode this batch. Shape: (number of episodes)
        """

        # Batch data. For more details, check function header.
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []
        batch_lens = []

        # Keep simulating until we've run more than or equal to specified moves per batch
        batch_move_cnt = 0
        while batch_move_cnt < self.max_batch_moves:
            eps_rewards = []  # rewards collected per episode

            # Reset the environment
            state = self.env.reset()
            end = False

            # Run an episode for a maximum of max_eps_moves moves
            for ep_move_cnt in range(self.max_eps_moves):
                # TODO: add option for rendering env

                batch_move_cnt += 1  # Increment moves ran this batch so far

                # Track states in this batch
                batch_states.append(state)

                # Calculate action and make a step in the env.
                action, log_prob = self.get_action(state)
                state, reward, score, end, win = self.env.step(action)

                # Track recent reward, action, and action log probability
                eps_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if end:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_move_cnt + 1)
            batch_rewards.append(eps_rewards)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_states = torch.tensor(batch_states, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rewards)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rewards"] = batch_rewards
        self.logger["batch_lens"] = batch_lens

        return batch_states, batch_actions, batch_log_probs, batch_rtgs, batch_lens

    def learn(self, num_moves: int) -> None:
        """
        Train the actor and critic networks. Main PPO algorithm.

        Inputs:
            num_moves:    The total number of episodes to train form

        Outputs:
            None
        """

        move_cnt = 0  # Episode counter
        iter_cnt = 0  # Iterations counter

        while move_cnt < num_moves:  # ALG STEP 2
            # Collecting a batch of simulations
            batch_states, batch_actions, batch_log_probs, batch_rtgs, batch_lens = (
                self.collect_batch()
            )  # ALG STEP 3

            # Calculate how many moves we collected this batch
            move_cnt += np.sum(batch_lens)

            # Increment the number of iterations
            iter_cnt += 1

            # Logging moves so far and iterations so far
            self.logger["move_cnt"] = move_cnt
            self.logger["iter_cnt"] = iter_cnt

            # Calculate advantage at k-th iteration
            V, _ = self.get_value(batch_states, batch_actions)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.num_updates):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.get_value(batch_states, batch_actions)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger["actor_losses"].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if iter_cnt % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./ppo_critic.pth")

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Inputs:
                None

        Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        move_cnt = self.logger["move_cnt"]
        iter_cnt = self.logger["iter_cnt"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_eps_rewards = np.mean(
            [np.sum(eps_rewards) for eps_rewards in self.logger["batch_rewards"]]
        )
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger["actor_losses"]]
        )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_eps_rewards = str(round(avg_eps_rewards, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{iter_cnt} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_eps_rewards}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {move_cnt}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rewards"] = []
        self.logger["actor_losses"] = []
