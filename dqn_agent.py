import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork, FromPixelsQNetwork, FromPixelsDuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PER_UPDATE_EVERY = 1    # how often to update the network

A = 0.5 # for the importance sampling weight computation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn, dueling_dqn, from_pixels):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_dqn (bool): id True, use double_dqn implementation
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.from_pixels = from_pixels

        # Q-Network
        if not self.from_pixels:
            if not self.dueling_dqn:
                self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            else:
                self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
                self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            if not self.dueling_dqn:
                self.qnetwork_local = FromPixelsQNetwork(action_size, seed).to(device)
                self.qnetwork_target = FromPixelsQNetwork(action_size, seed).to(device)
            else:
                self.qnetwork_local = FromPixelsDuelingQNetwork(action_size, seed).to(device)
                self.qnetwork_target = FromPixelsDuelingQNetwork(action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) # optimizer is applied to the qnetwork_local parameters, since they are updated at each learning step!

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
       
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if not self.double_dqn:
                    self.learn(experiences, GAMMA)
                else:
                    self.double_dqn_learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # exploitation
        else:
            return random.choice(np.arange(self.action_size)) # exploration

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences # dones: Flags indicating whether each episode ended after the action (1 if done, 0 if not).

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # (1 - dones) ensures that if the episode is done (i.e., dones = 1), the future reward is ignored by multiplying the Q-value by 0

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) # This loss measures how far the network’s predictions are from the desired targets.
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # soft update is used to implement the fixed qnetwork_target (it involves smoothly updating the target network weights from the local network) (a periodic update should be an alternative)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

    def double_dqn_learn(self, experiences, gamma):
        """Double DQN implementation: select the best action using one set of parameters w, but evaluate it using a different set of parameters w'.
        When using DQNs with fixed Q targets, we already have an alternate set of parameters w−. 

        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences # dones: Flags indicating whether each episode ended after the action (1 if done, 0 if not).

        # Use local model to select the best action for next_states (argmax)
        best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)

        # Use target model to evaluate the Q-value of the best actions
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # (1 - dones) ensures that if the episode is done (i.e., dones = 1), the future reward is ignored by multiplying the Q-value by 0

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) # This loss measures how far the network’s predictions are from the desired targets.
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # soft_update is used to implement the fixed qnetwork_target (it involves smoothly updating the target network weights from the local network) (a periodic update should be an alternative)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                   

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizedExperienceReplayAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn, e, dueling_dqn, from_pixels):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_dqn (bool): id True, use double_dqn implementation
            e (float):  for the importance sampling weight computation
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_dqn = double_dqn
        self.e = e
        self.dueling_dqn = dueling_dqn
        self.from_pixels = from_pixels

        # Q-Network
        if not from_pixels:
            if not self.dueling_dqn:
                self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            else:
                self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
                self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            if not self.dueling_dqn:
                self.qnetwork_local = FromPixelsQNetwork(action_size, seed).to(device)
                self.qnetwork_target = FromPixelsQNetwork(action_size, seed).to(device)
            else:
                self.qnetwork_local = FromPixelsDuelingQNetwork(action_size, seed).to(device)
                self.qnetwork_target = FromPixelsDuelingQNetwork(action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) # optimizer is applied to the qnetwork_local parameters, since they are updated at each learning step!

        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, A, seed)
       
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, b):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, self.e, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % PER_UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if not self.double_dqn:
                    self.learn(experiences, GAMMA, b)
                else:
                    self.double_dqn_learn(experiences, GAMMA, b)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # exploitation
        else:
            return random.choice(np.arange(self.action_size)) # exploration

    def learn(self, experiences, gamma, b):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): tuple of (idx, s, a, r, s', priority, done) tuples 
            gamma (float): discount factor
            b (float): for the importance sampling weight computation
        """

        indices, states, actions, rewards, next_states, dones = experiences # dones: Flags indicating whether each episode ended after the action (1 if done, 0 if not).

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # (1 - dones) ensures that if the episode is done (i.e., dones = 1), the future reward is ignored by multiplying the Q-value by 0

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 
        assert Q_targets.shape == Q_expected.shape, "[ERROR] Q_targets and Q_expected shapes do not match!"

        TD_errors = torch.abs(Q_expected - Q_targets).detach() + self.e
        assert TD_errors.shape == Q_expected.shape, "[ERROR] Td_errors and Q_expected shapes do not match!"
        self.memory.update_priorities(indices, TD_errors)

        importance_sampling_weights = self.memory.compute_importance_sampling_weight(indices, b) # b is increased over time
        assert importance_sampling_weights.shape[0] == len(indices), "[ERROR] Importance sampling weights do not match batch size!"

        # Compute loss
        loss = (importance_sampling_weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean() # This loss measures how far the network’s predictions are from the desired targets.
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # soft update is used to implement the fixed qnetwork_target (it involves smoothly updating the target network weights from the local network) (a periodic update should be an alternative)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

    def double_dqn_learn(self, experiences, gamma, b):
        """Double DQN implementation: select the best action using one set of parameters w, but evaluate it using a different set of parameters w'.
        When using DQNs with fixed Q targets, we already have an alternate set of parameters w−. 

        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): tuple of (idx, s, a, r, s', priority, done) tuples 
            gamma (float): discount factor
            b (float): for the importance sampling weight computation
        """ 

        indices, states, actions, rewards, next_states, dones = experiences # dones: Flags indicating whether each episode ended after the action (1 if done, 0 if not).

        # Use local model to select the best action for next_states (argmax)
        best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)

        # Use target model to evaluate the Q-value of the best actions
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # (1 - dones) ensures that if the episode is done (i.e., dones = 1), the future reward is ignored by multiplying the Q-value by 0

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 
        assert Q_targets.shape == Q_expected.shape, "[ERROR] Q_targets and Q_expected shapes do not match!"

        TD_errors = torch.abs(Q_expected - Q_targets).detach() + self.e
        assert TD_errors.shape == Q_expected.shape, "[ERROR] Td_errors and Q_expected shapes do not match!"
        self.memory.update_priorities(indices, TD_errors)

        importance_sampling_weights = self.memory.compute_importance_sampling_weight(indices, b) # b is increased over time
        assert importance_sampling_weights.shape[0] == len(indices), "[ERROR] Importance sampling weights do not match batch size!"

        # Compute loss
        loss = (importance_sampling_weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean() # This loss measures how far the network’s predictions are from the desired targets.
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # soft_update is used to implement the fixed qnetwork_target (it involves smoothly updating the target network weights from the local network) (a periodic update should be an alternative)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                   

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store prioritized experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, a, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            a (float): hyperparameter for the importance sampling weight computation
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "priority", "done"])
        self.a = a
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, priority, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, priority, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = random.sample(range(len(self.memory)), k=self.batch_size)

        experiences = [self.memory[idx] for idx in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (indices, states, actions, rewards, next_states, dones)
    
    def update_priorities(self, indices, TD_errors):
        """Update priorities for the selected experiences (sampled experiences for which the TD error has been computed)"""
        assert len(indices) == len(TD_errors), "[ERROR] Indices and TD_errors lengths do not match!"
        for i, idx in enumerate(indices):
            experience = self.memory[idx]
            updated_experience = self.experience(experience.state, experience.action, experience.reward, experience.next_state, TD_errors[i], experience.done)
            self.memory[idx] = updated_experience
    
    def compute_importance_sampling_weight(self, indices, b):
        """Compute the importance sampling weight for the experiences related to the selected indeces"""
        n = len(self.memory)
        denominator = sum(self.memory[i].priority**self.a for i in range(n))
        
        experiences = [self.memory[idx] for idx in indices]
        importance_sampling_weight = torch.stack([((1 / (n * (e.priority ** self.a) / denominator)) ** b) for e in experiences if e is not None]).float().to(device)

        return importance_sampling_weight

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)