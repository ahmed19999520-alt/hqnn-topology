import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class TopologyState:
    def __init__(self, adjacency: np.ndarray,
                 fidelity: float,
                 entanglement: float,
                 euler_drift: int,
                 noise_level: float):
        self.adjacency = adjacency.copy()
        self.fidelity = fidelity
        self.entanglement = entanglement
        self.euler_drift = euler_drift
        self.noise_level = noise_level

    def to_tensor(self) -> torch.Tensor:
        n = self.adjacency.shape[0]
        flat_adj = np.abs(self.adjacency).flatten()
        scalar_features = np.array([
            self.fidelity,
            self.entanglement,
            float(self.euler_drift) / 10.0,
            self.noise_level,
        ])
        features = np.concatenate([flat_adj, scalar_features])
        return torch.FloatTensor(features)

    @property
    def dim(self) -> int:
        n = self.adjacency.shape[0]
        return n * n + 4


class TopologyAction:
    ADD_EDGE = 0
    REMOVE_EDGE = 1
    STRENGTHEN_EDGE = 2
    WEAKEN_EDGE = 3
    N_ACTIONS = 4

    def __init__(self, action_type: int, edge: Tuple[int, int]):
        self.action_type = action_type
        self.edge = edge


class DQNPolicy(nn.Module):
    def __init__(self, state_dim: int, n_nodes: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_nodes = n_nodes
        n_edge_actions = n_nodes * (n_nodes - 1) // 2
        self.n_actions = TopologyAction.N_ACTIONS * n_edge_actions

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.n_actions),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        advantage = self.advantage_stream(encoded)
        value = self.value_stream(encoded)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int,
              reward: float, next_state: torch.Tensor, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.stack(next_states),
                torch.FloatTensor(dones))

    def __len__(self) -> int:
        return len(self.buffer)


class TopologyRLOptimizer:
    def __init__(self, n_nodes: int,
                 state_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 5000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        self.n_nodes = n_nodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNPolicy(state_dim, n_nodes, hidden_dim).to(self.device)
        self.target_net = DQNPolicy(state_dim, n_nodes, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                      lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6)
        self.replay_buffer = ReplayBuffer()

        self.edge_list = [(i, j) for i in range(n_nodes)
                          for j in range(i+1, n_nodes)]
        self.n_edges = len(self.edge_list)
        self.steps_done = 0
        self.loss_history: List[float] = []
        self.reward_history: List[float] = []

    def _get_epsilon(self) -> float:
        return (self.epsilon_end +
                (self.epsilon_start - self.epsilon_end) *
                np.exp(-self.steps_done / self.epsilon_decay))

    def select_action(self, state: torch.Tensor) -> int:
        epsilon = self._get_epsilon()
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randint(0, self.policy_net.n_actions - 1)
        with torch.no_grad():
            q_values = self.policy_net(state.to(self.device).unsqueeze(0))
            return int(q_values.argmax().item())

    def decode_action(self, action_idx: int) -> TopologyAction:
        action_type = action_idx // self.n_edges
        edge_idx = action_idx % self.n_edges
        edge = self.edge_list[edge_idx]
        return TopologyAction(action_type, edge)

    def compute_reward(self, state: TopologyState,
                        next_state: TopologyState) -> float:
        fidelity_reward = (next_state.fidelity - state.fidelity) * 5.0
        entanglement_reward = (next_state.entanglement - state.entanglement) * 3.0
        euler_penalty = -abs(next_state.euler_drift) * 0.5
        noise_bonus = max(0.0, (state.noise_level - next_state.noise_level) * 2.0)
        return fidelity_reward + entanglement_reward + euler_penalty + noise_bonus

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_state'])
        self.target_net.load_state_dict(ckpt['target_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.steps_done = ckpt['steps_done']
        self.loss_history = ckpt['loss_history']
        self.reward_history = ckpt['reward_history']


class RLTopologyTrainer:
    def __init__(self, network, corrector, slime, n_episodes: int = 500):
        self.network = network
        self.corrector = corrector
        self.slime = slime
        self.n_episodes = n_episodes

        n_nodes = network.n
        sample_state = TopologyState(
            adjacency=np.zeros((n_nodes, n_nodes)),
            fidelity=1.0, entanglement=0.5,
            euler_drift=0, noise_level=0.05
        )
        state_dim = sample_state.dim
        self.rl_optimizer = TopologyRLOptimizer(n_nodes, state_dim)

    def get_current_state(self, noise_level: float) -> TopologyState:
        snap = self.network.snapshot()
        A = np.abs(np.real(self.network.get_adjacency_matrix()))
        return TopologyState(
            adjacency=A,
            fidelity=snap['network_fidelity'],
            entanglement=snap['average_entanglement'],
            euler_drift=snap['euler_drift'],
            noise_level=noise_level,
        )

    def apply_action(self, action: TopologyAction) -> None:
        i, j = action.edge
        key = (min(i,j), max(i,j))
        if key not in self.network.edges:
            from hqnn.core.quantum_edge import QuantumEdge
            self.network.edges[key] = QuantumEdge(i, j)
            return
        edge = self.network.edges[key]
        import cmath
        mag = abs(edge.weight)
        phase = cmath.phase(edge.weight)
        if action.action_type == TopologyAction.ADD_EDGE:
            edge.weight = complex(0.1, 0.0)
        elif action.action_type == TopologyAction.REMOVE_EDGE:
            edge.weight = complex(1e-6, 0.0)
        elif action.action_type == TopologyAction.STRENGTHEN_EDGE:
            edge.weight = min(mag * 1.2, 5.0) * (np.cos(phase) + 1j*np.sin(phase))
        elif action.action_type == TopologyAction.WEAKEN_EDGE:
            edge.weight = max(mag * 0.8, 1e-6) * (np.cos(phase) + 1j*np.sin(phase))

    def train(self, noise_profile: np.ndarray) -> Dict:
        results = {
            'episode_rewards': [],
            'episode_fidelities': [],
            'loss_history': [],
        }
        for episode in range(self.n_episodes):
            self.network.reset()
            noise = float(noise_profile[episode % len(noise_profile)])
            state_obj = self.get_current_state(noise)
            state_tensor = state_obj.to_tensor()
            episode_reward = 0.0
            episode_fidelities = []

            for step in range(20):
                action_idx = self.rl_optimizer.select_action(state_tensor)
                action = self.rl_optimizer.decode_action(action_idx)
                self.apply_action(action)
                self.network.apply_global_decoherence(noise, 0.01)
                self.network.update_entanglement_all()
                self.slime.step(noise)
                diag = self.corrector.diagnose()
                self.corrector.correct(diag)

                next_state_obj = self.get_current_state(noise)
                next_tensor = next_state_obj.to_tensor()
                reward = self.rl_optimizer.compute_reward(state_obj, next_state_obj)
                episode_reward += reward
                done = step == 19

                self.rl_optimizer.replay_buffer.push(
                    state_tensor, action_idx, reward, next_tensor, float(done))
                loss = self.rl_optimizer.train_step()
                if loss is not None:
                    results['loss_history'].append(loss)

                state_obj = next_state_obj
                state_tensor = next_tensor
                episode_fidelities.append(next_state_obj.fidelity)

            results['episode_rewards'].append(episode_reward)
            results['episode_fidelities'].append(np.mean(episode_fidelities))

            if episode % 50 == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:.3f} | "
                      f"Fidelity: {np.mean(episode_fidelities):.4f} | "
                      f"Epsilon: {self.rl_optimizer._get_epsilon():.3f}")

        return results