from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time as ostime
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import original_environment as oe


@dataclass
class Config:
    """全局配置参数"""
    # seed: int = 42
    seed: int = random.randint(1, 10000)
    n_actions: int = 9
    # 状态维度: grid_norm, minute_norm, cyc_sin, cyc_cos, price_level, pickup_prob, seek_ratio, trans_ratio
    state_dim: int = 8
    seq_len: int = 6
    max_minutes: int = 60
    episodes: int = 50000
    eval_episodes: int = 500
    train_log_every: int = 2000
    checkpoint_every: int = 10000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 5e-4
    tau: float = 0.005
    epsilon_start: float = 0.6
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 500000
    replay_size: int = 400_000
    warmup_steps: int = 20_000
    update_every: int = 1
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lambda_price: float = 0.2
    lambda_idle_penalty: float = 0.04
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-5
    # 对抗训练
    adv_train_enabled: bool = True
    adv_train_always: bool = False          # 概率触发
    adv_train_prob: float = 0.45
    adv_log_every: int = 50
    # 训练提速
    adv_enable_pgd_after_ratio: float = 0.6
    adv_pgd_prob_after_warm: float = 0.7
    # 攻击参数
    fgsm_eps: float = 0.1
    pgd_eps: float = 0.05
    pgd_step: float = 0.02
    pgd_steps: int = 3
    noise_std: float = 0.1
    boundary_ratio: float = 0.2
    eval_clean_only: bool = False
    # 奖励缩放因子
    reward_scale: float = 1.0


def apply_fast_safe_profile(cfg: Config) -> None:
    cfg.episodes = 30000
    cfg.eval_episodes = 300
    cfg.train_log_every = 2000
    cfg.checkpoint_every = 10000
    cfg.warmup_steps = 8000
    # PGD 只在后期加入，前期只 fgsm/noise
    cfg.adv_enable_pgd_after_ratio = 0.7
    cfg.adv_pgd_prob_after_warm = 0.7
    # 降低日志 I/O
    cfg.adv_log_every = 500



def default_eval_scenarios(cfg: Config, agent_type: str = "dqn") -> List[Tuple[str, str, float]]:
    if cfg.eval_clean_only:
        return [("clean", "none", 0.0)]
    base = [
        ("clean", "none", 0.0),
        ("whitebox_fgsm_low", "fgsm", cfg.fgsm_eps),
        ("whitebox_fgsm_high", "fgsm", cfg.fgsm_eps * 2),
        ("whitebox_pgd_low", "pgd", cfg.pgd_eps),
        ("whitebox_pgd_high", "pgd", cfg.pgd_eps * 2),
        ("blackbox_noise_low", "random_noise", cfg.noise_std),
        ("blackbox_noise_high", "random_noise", cfg.noise_std * 2.5),
        ("blackbox_boundary_low", "boundary", cfg.boundary_ratio),
        ("blackbox_boundary_high", "boundary", cfg.boundary_ratio * 2),
    ]
    return base


def normalize_output_suffix(suffix: str) -> str:
    s = (suffix or "").strip()
    if not s:
        return ""
    return s if s.startswith("_") else f"_{s}"


def with_suffix(base_filename: str, suffix: str) -> str:
    suf = normalize_output_suffix(suffix)
    if not suf:
        return base_filename
    root, ext = os.path.splitext(base_filename)
    return f"{root}{suf}{ext}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 (PER)"""
    def __init__(self, capacity: int, alpha: float) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        self.storage: List[Optional[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, float]]] = [
            None
        ] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state_seq: np.ndarray,
        state_now: np.ndarray,
        action: int,
        reward: float,
        next_state_seq: np.ndarray,
        next_state_now: np.ndarray,
        done: float,
        bonus_priority: float = 0.0,
    ) -> None:
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.storage[self.pos] = (
            state_seq.astype(np.float32),
            state_now.astype(np.float32),
            int(action),
            float(reward),
            next_state_seq.astype(np.float32),
            next_state_now.astype(np.float32),
            float(done),
        )
        self.priorities[self.pos] = max_prio + bonus_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Dict[str, np.ndarray]:
        if self.size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.size]
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.storage[idx] for idx in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        state_seq = np.stack([s[0] for s in samples], axis=0)
        state_now = np.stack([s[1] for s in samples], axis=0)
        actions = np.array([s[2] for s in samples], dtype=np.int64)
        rewards = np.array([s[3] for s in samples], dtype=np.float32)
        next_state_seq = np.stack([s[4] for s in samples], axis=0)
        next_state_now = np.stack([s[5] for s in samples], axis=0)
        dones = np.array([s[6] for s in samples], dtype=np.float32)
        return {
            "state_seq": state_seq,
            "state_now": state_now,
            "actions": actions,
            "rewards": rewards,
            "next_state_seq": next_state_seq,
            "next_state_now": next_state_now,
            "dones": dones,
            "indices": indices,
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities


class SpatioTemporalEncoder(nn.Module):
    """CNN + GRU + Attention 的时空特征提取模块"""
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=48, hidden_size=hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.now_proj = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.out_dim = hidden_dim * 2

    def forward(self, seq_x: torch.Tensor, now_x: torch.Tensor) -> torch.Tensor:
        x = seq_x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        h, _ = self.gru(x)
        w = torch.softmax(self.attn(h), dim=1)
        context = (h * w).sum(dim=1)
        now_feat = self.now_proj(now_x)
        return torch.cat([context, now_feat], dim=-1)


class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.encoder = SpatioTemporalEncoder(state_dim=state_dim, hidden_dim=64)
        self.q_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, seq_x: torch.Tensor, now_x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(seq_x, now_x)
        return self.q_head(z)


class AttackEngine:
    """对抗攻击方法集合，支持训练时的正则化和评估时的动作翻转攻击"""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def _clamp_like(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.max(torch.min(x_adv, x + eps), x - eps)

    # ========== 训练时使用的攻击 ==========
    def fgsm_reg(self, model, seq_x, now_x, target_q, eps):
        seq_adv = seq_x.detach().clone().requires_grad_(True)
        now_adv = now_x.detach().clone().requires_grad_(True)
        pred = model(seq_adv, now_adv)
        loss = F.mse_loss(pred, target_q.detach())
        loss.backward()
        seq_adv = seq_adv + eps * seq_adv.grad.sign()
        now_adv = now_adv + eps * now_adv.grad.sign()
        return seq_adv.detach(), now_adv.detach()

    def pgd_reg(self, model, seq_x, now_x, target_q, eps, step, k):
        seq_adv = seq_x.detach().clone()
        now_adv = now_x.detach().clone()
        seq_start, now_start = seq_x.detach().clone(), now_x.detach().clone()
        for _ in range(k):
            seq_adv.requires_grad_(True)
            now_adv.requires_grad_(True)
            pred = model(seq_adv, now_adv)
            loss = F.mse_loss(pred, target_q.detach())
            loss.backward()
            with torch.no_grad():
                seq_adv = seq_adv + step * seq_adv.grad.sign()
                now_adv = now_adv + step * now_adv.grad.sign()
                seq_adv = self._clamp_like(seq_adv, seq_start, eps)
                now_adv = self._clamp_like(now_adv, now_start, eps)
        return seq_adv.detach(), now_adv.detach()


    def fgsm_action(self, model, seq_x, now_x, action, eps):
        """FGSM攻击"""
        seq_adv = seq_x.detach().clone().requires_grad_(True)
        now_adv = now_x.detach().clone().requires_grad_(True)
        q = model(seq_adv, now_adv)
        loss = -q.gather(1, action.unsqueeze(1)).squeeze().sum()   # 梯度上升
        loss.backward()
        seq_adv = seq_adv + eps * seq_adv.grad.sign()
        now_adv = now_adv + eps * now_adv.grad.sign()
        return seq_adv.detach(), now_adv.detach()

    def pgd_action(self, model, seq_x, now_x, action, eps, step, k):
        """PGD攻击"""
        seq_adv = seq_x.detach().clone()
        now_adv = now_x.detach().clone()
        seq_start, now_start = seq_x.detach().clone(), now_x.detach().clone()
        for _ in range(k):
            seq_adv.requires_grad_(True)
            now_adv.requires_grad_(True)
            q = model(seq_adv, now_adv)
            loss = -q.gather(1, action.unsqueeze(1)).squeeze().sum()
            loss.backward()
            with torch.no_grad():
                seq_adv = seq_adv + step * seq_adv.grad.sign()
                now_adv = now_adv + step * now_adv.grad.sign()
                seq_adv = self._clamp_like(seq_adv, seq_start, eps)
                now_adv = self._clamp_like(now_adv, now_start, eps)
        return seq_adv.detach(), now_adv.detach()

    # ========== 黑盒攻击 ==========
    def random_noise(self, seq_x, now_x, std):
        return (seq_x + torch.randn_like(seq_x) * std).detach(), (now_x + torch.randn_like(now_x) * std).detach()

    def boundary_like(self, seq_x, now_x, ratio):
        seq_adv = seq_x.detach().clone()
        now_adv = now_x.detach().clone()
        seq_mask = (torch.rand_like(seq_adv) < ratio).float()
        now_mask = (torch.rand_like(now_adv) < ratio).float()
        seq_adv = seq_adv * (1.0 - seq_mask) + (1.0 - seq_adv) * seq_mask
        now_adv = now_adv * (1.0 - now_mask) + (1.0 - now_adv) * now_mask
        return seq_adv, now_adv


class EnvAdapter:
    """环境适配器：预计算静态数据，状态构造，奖励缩放关闭"""
    def __init__(self, env: oe.Orignal_Env, cfg: Config) -> None:
        self.env = env
        self.cfg = cfg
        self.last_reward = 0.0
        self.seek_time = 0.0
        self.trans_time = 0.0
        self.history: Deque[np.ndarray] = deque(maxlen=cfg.seq_len)

        # 预计算价格水平和接客概率
        self.price_levels = np.zeros(900, dtype=np.float32)
        self.pickup_probs = np.zeros(900, dtype=np.float32)
        try:
            for gid in range(1, 901):
                conf = float(oe.dy.iloc[gid - 1, 5])
                self.price_levels[gid - 1] = max(0.0, min((conf - 1.0) / 0.8, 1.0))
                self.pickup_probs[gid - 1] = float(env.find_prob(gid))
        except Exception as e:
            print(f"[Warning] 预计算静态数据失败: {e}，将使用动态读取方式")

    def reset(self, start_grid: int) -> Tuple[np.ndarray, np.ndarray]:
        self.env.environment_change(start_grid)
        self.last_reward = 0.0
        self.seek_time = 0.0
        self.trans_time = 0.0
        now = self._build_state(grid_id=start_grid, minute=0)
        self.history.clear()
        for _ in range(self.cfg.seq_len):
            self.history.append(now.copy())
        return np.stack(self.history, axis=0), now

    def _grid_price_level(self, grid_id: int) -> float:
        return self.price_levels[grid_id - 1]

    def _pickup_prob(self, grid_id: int) -> float:
        return self.pickup_probs[grid_id - 1]

    def _build_state(self, grid_id: int, minute: int) -> np.ndarray:
        minute_norm = minute / self.cfg.max_minutes
        cyc_sin = math.sin(2.0 * math.pi * minute_norm)
        cyc_cos = math.cos(2.0 * math.pi * minute_norm)
        grid_norm = (grid_id - 1) / 899.0
        price_level = self._grid_price_level(grid_id)
        pickup_prob = self._pickup_prob(grid_id)
        seek_ratio = min(1.0, self.seek_time / self.cfg.max_minutes)
        trans_ratio = min(1.0, self.trans_time / self.cfg.max_minutes)
        # 移除reward_proxy，状态维度 8
        return np.array(
            [grid_norm, minute_norm, cyc_sin, cyc_cos, price_level, pickup_prob, seek_ratio, trans_ratio],
            dtype=np.float32,
        )

    def step(self, action_idx: int, t: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, float]]:
        action = action_idx + 1
        next_grid = self.env.get_next_grid(action)
        if next_grid < 1 or next_grid > 900:
            valid_actions = [a for a in range(1, 10) if 1 <= self.env.get_next_grid(a) <= 900]
            action = random.choice(valid_actions)
        s, t_next, reward_raw, done, seek_time, trans_time = self.env.step_amend(
            action, t, self.seek_time, self.trans_time
        )
        self.seek_time = float(seek_time)
        self.trans_time = float(trans_time)

        price_level = self._grid_price_level(s)
        idle_penalty = self.cfg.lambda_idle_penalty * (1.0 if reward_raw <= 0 else 0.0)
        shaped_reward = float(reward_raw + self.cfg.lambda_price * price_level - idle_penalty)
        shaped_reward *= self.cfg.reward_scale   # 缩放因子为 1.0，无影响
        self.last_reward = shaped_reward

        now = self._build_state(grid_id=s, minute=int(t_next))
        self.history.append(now.copy())
        seq = np.stack(self.history, axis=0)
        info = {
            "raw_reward": float(reward_raw),
            "price_level": float(price_level),
            "seek_time": float(self.seek_time),
            "trans_time": float(self.trans_time),
            "grid_id": float(s),
            "minute": float(t_next),
        }
        return seq, now, shaped_reward, bool(done), info


class DQNAgent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.online = DQN(cfg.state_dim, cfg.n_actions).to(self.device)
        self.target = DQN(cfg.state_dim, cfg.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplayBuffer(cfg.replay_size, alpha=cfg.per_alpha)
        self.attack = AttackEngine(cfg)
        self.train_steps = 0
        self.train_episodes = 0   # 用于epsilon衰减
        self.env_steps = 0
        self.adv_trigger_count = 0
        self.adv_log_every = cfg.adv_log_every

    def epsilon(self) -> float:
        # 基于已完成episode比例衰减
        frac = min(1.0, self.env_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def record_env_step(self):
        self.env_steps += 1

    def beta(self) -> float:
        # beta仍基于 train_steps
        frac = min(1.0, self.train_steps / float(self.cfg.epsilon_decay_steps)) if hasattr(self.cfg, 'epsilon_decay_steps') else 1.0
        # 使用默认衰减，保留原逻辑
        return self.cfg.per_beta_start + frac * (self.cfg.per_beta_end - self.cfg.per_beta_start)

    def act(self, state_seq: np.ndarray, state_now: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randint(0, self.cfg.n_actions - 1)
        with torch.no_grad():
            seq_t = torch.from_numpy(state_seq).float().unsqueeze(0).to(self.device)
            now_t = torch.from_numpy(state_now).float().unsqueeze(0).to(self.device)
            q_values = self.online(seq_t, now_t)
            return int(torch.argmax(q_values, dim=1).item())

    def push_transition(self, state_seq, state_now, action, reward, next_state_seq, next_state_now, done, attacked):
        bonus = 0.2 if attacked else 0.0
        self.buffer.push(state_seq, state_now, action, reward, next_state_seq, next_state_now, float(done), bonus)

    def _soft_update(self):
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * op.data)

    def update(self) -> Dict[str, float]:
        self.train_steps += 1
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return {}
        if self.train_steps % self.cfg.update_every != 0:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size, beta=self.beta())
        seq = torch.from_numpy(batch["state_seq"]).float().to(self.device)
        now = torch.from_numpy(batch["state_now"]).float().to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        next_seq = torch.from_numpy(batch["next_state_seq"]).float().to(self.device)
        next_now = torch.from_numpy(batch["next_state_now"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)
        weights = torch.from_numpy(batch["weights"]).float().to(self.device)

        q = self.online(seq, now).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_seq, next_now), dim=1)
            next_q = self.target(next_seq, next_now).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        td_error = td_target - q
        base_loss = (weights * td_error.pow(2)).mean()

        robust_loss = torch.tensor(0.0, device=self.device)
        use_adv = self.cfg.adv_train_enabled and (
            self.cfg.adv_train_always or (random.random() < self.cfg.adv_train_prob)
        )
        if use_adv:
            self.adv_trigger_count += 1
            # fast-safe: 前期只做 fgsm/noise；后期按概率混入 pgd，降低总耗时
            episode_ratio = (self.train_episodes / max(1, self.cfg.episodes))
            allow_pgd = episode_ratio >= self.cfg.adv_enable_pgd_after_ratio
            if allow_pgd and random.random() < self.cfg.adv_pgd_prob_after_warm:
                mode = "pgd"
            else:
                mode = random.choice(["fgsm", "noise"])
            if self.adv_trigger_count % self.adv_log_every == 1:
                print(f"[AdvTrain] Triggered (count={self.adv_trigger_count}, steps={self.train_steps}), mode={mode}, eps={self.epsilon():.3f}")
            with torch.no_grad():
                full_q_target = self.online(seq, now)
            if mode == "fgsm":
                seq_adv, now_adv = self.attack.fgsm_reg(self.online, seq, now, full_q_target, self.cfg.fgsm_eps)
            elif mode == "pgd":
                seq_adv, now_adv = self.attack.pgd_reg(
                    self.online, seq, now, full_q_target, self.cfg.pgd_eps, self.cfg.pgd_step, self.cfg.pgd_steps
                )
            else:
                seq_adv, now_adv = self.attack.random_noise(seq, now, self.cfg.noise_std)
            robust_q = self.online(seq_adv, now_adv)
            robust_loss = F.mse_loss(robust_q, full_q_target.detach())

        loss = base_loss + 0.35 * robust_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self._soft_update()

        new_prios = td_error.detach().abs().cpu().numpy() + self.cfg.per_eps
        self.buffer.update_priorities(batch["indices"], new_prios)

        return {
            "loss": float(loss.item()),
            "td_abs": float(td_error.detach().abs().mean().item()),
            "eps": float(self.epsilon()),
        }


class QLearningBaseline:
    """表格型 Q-learning，epsilon 衰减基于 episode 比例，与 DQN 对齐"""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lr = 0.02
        self.gamma = 0.7
        self.epsilon_start = 0.9
        self.epsilon_end = cfg.epsilon_end
        self.q = np.zeros((900, cfg.max_minutes, cfg.n_actions), dtype=np.float32)
        self.train_episodes = 0
        self.total_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.cfg.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _state_idx(self, grid_id: int, minute: int) -> Tuple[int, int]:
        g = int(np.clip(grid_id - 1, 0, 899))
        m = int(np.clip(minute, 0, self.cfg.max_minutes - 1))
        return g, m

    def choose_action(self, grid_id: int, minute: int, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        g, m = self._state_idx(grid_id, minute)
        if random.random() < eps:
            return random.randint(0, self.cfg.n_actions - 1)
        qv = self.q[g, m]
        max_v = np.max(qv)
        candidates = np.where(qv == max_v)[0]
        return int(np.random.choice(candidates))

    def learn(self, grid_id, minute, action, reward, next_grid, next_minute, done):
        self.total_steps += 1
        g, m = self._state_idx(grid_id, minute)
        ng, nm = self._state_idx(next_grid, next_minute)
        q_pred = self.q[g, m, action]
        q_next = 0.0 if done else float(np.max(self.q[ng, nm]))
        q_tgt = reward + self.gamma * q_next
        self.q[g, m, action] += self.lr * (q_tgt - q_pred)


def apply_discrete_attack_to_obs(
    obs_grid: int,
    obs_minute: int,
    attack_mode: Optional[str],
    attack_strength: float,
    cfg: Config,
) -> Tuple[int, int]:
    if attack_mode is None or attack_mode == "none":
        return obs_grid, obs_minute
    g, t = obs_grid, obs_minute

    if attack_mode in ("fgsm", "pgd"):
        print(f"[Warning] Q-learning 不支持白盒攻击 {attack_mode}，已回退为 random_noise 强度 {attack_strength}")
        attack_mode = "random_noise"

    if attack_mode == "random_noise":
        delta = max(1, int(attack_strength * 120))
        g = int(np.clip(g + np.random.randint(-delta, delta + 1), 1, 900))
        t = int(np.clip(t + np.random.randint(-2, 3), 0, cfg.max_minutes - 1))
    elif attack_mode == "boundary":
        if random.random() < min(1.0, attack_strength * 2.0):
            g = 1 if random.random() < 0.5 else 900
            t = cfg.max_minutes - 1
    return g, t


def run_episode_generic(
    env_adp_or_raw: Union[EnvAdapter, oe.Orignal_Env],
    agent: Union[DQNAgent, QLearningBaseline],
    cfg: Config,
    start_grid: int,
    training: bool,
    attack_mode: Optional[str] = None,
    attack_strength: float = 0.0,
) -> Dict[str, float]:
    is_dqn = isinstance(agent, DQNAgent)

    if is_dqn:
        env_adp = env_adp_or_raw  # type: EnvAdapter
        seq, now = env_adp.reset(start_grid)
        t = 0
    else:
        raw_env = env_adp_or_raw  # type: oe.Orignal_Env
        raw_env.environment_change(start_grid)
        s_id = start_grid
        t = 0
        seek_time = 0.0
        trans_time = 0.0

    rewards: List[float] = []
    order_count = 0
    attacked_steps = 0
    losses: List[float] = []

    while True:
        if is_dqn:
            # --- DQN 分支 ---
            seq_in, now_in = seq.copy(), now.copy()
            if attack_mode is not None and attack_mode != "none":
                seq_t = torch.from_numpy(seq_in).float().unsqueeze(0).to(agent.device)
                now_t = torch.from_numpy(now_in).float().unsqueeze(0).to(agent.device)
                # 获取当前动作
                with torch.no_grad():
                    q_vals = agent.online(seq_t, now_t)
                    action_taken = torch.argmax(q_vals, dim=1).item()
                if attack_mode == "random_noise":
                    seq_a, now_a = agent.attack.random_noise(seq_t, now_t, std=attack_strength)
                elif attack_mode == "boundary":
                    seq_a, now_a = agent.attack.boundary_like(seq_t, now_t, ratio=attack_strength)
                elif attack_mode == "fgsm":
                    seq_a, now_a = agent.attack.fgsm_action(
                        agent.online, seq_t, now_t, torch.tensor([action_taken], device=agent.device), eps=attack_strength
                    )
                elif attack_mode == "pgd":
                    seq_a, now_a = agent.attack.pgd_action(
                        agent.online, seq_t, now_t, torch.tensor([action_taken], device=agent.device),
                        eps=attack_strength, step=max(attack_strength / 3.0, 1e-3), k=3
                    )
                else:
                    seq_a, now_a = seq_t, now_t
                seq_in = seq_a.squeeze(0).cpu().numpy()
                now_in = now_a.squeeze(0).cpu().numpy()
                attacked_steps += 1

            action = agent.act(seq_in, now_in, greedy=not training)
            if training:
                agent.record_env_step()  
            next_seq, next_now, reward, done, info = env_adp.step(action, t)
            if info["raw_reward"] > 0:
                order_count += 1
            rewards.append(reward)

            if training:
                attacked = attack_mode is not None and attack_mode != "none"
                agent.push_transition(seq, now, action, reward, next_seq, next_now, done, attacked=attacked)
                stats = agent.update()
                if "loss" in stats:
                    losses.append(stats["loss"])

            seq, now = next_seq, next_now
            t = int(info["minute"])
            if done:
                break
        else:
            # ---Q-learning分支 ---
            s_obs, t_obs = apply_discrete_attack_to_obs(s_id, t, attack_mode, attack_strength, cfg)
            if attack_mode is not None and attack_mode != "none":
                attacked_steps += 1

            valid_actions = [a for a in range(1, 10) if 1 <= raw_env.get_next_grid(a) <= 900]
            if not valid_actions:
                valid_actions = [5]

            chosen = agent.choose_action(s_obs, t_obs, greedy=not training) + 1
            if chosen not in valid_actions:
                if training:
                    action = random.choice(valid_actions)
                else:
                    g, m = agent._state_idx(s_obs, t_obs)
                    qv = agent.q[g, m]
                    action = max(valid_actions, key=lambda a: float(qv[a - 1]))
            else:
                action = chosen

            s_next, t_next, reward, done, seek_time, trans_time = raw_env.step_amend(action, t, seek_time, trans_time)
            rewards.append(float(reward))
            if reward > 0:
                order_count += 1

            if training:
                agent.learn(s_id, t, action - 1, float(reward), int(s_next), int(t_next), bool(done))

            s_id = int(s_next)
            t = int(t_next)
            if done:
                break

    score = float(np.sum(rewards))
    if is_dqn:
        total_time = max(1.0, env_adp.seek_time + env_adp.trans_time)
        sum_seek = env_adp.seek_time
        sum_trans = env_adp.trans_time
        final_minute = t
    else:
        total_time = max(1.0, seek_time + trans_time)
        sum_seek = seek_time
        sum_trans = trans_time
        final_minute = t

    order_rev = score / max(1, order_count)
    hour_rev = score / max(1.0, final_minute / 60.0)
    return {
        "score": score,
        "n_orders": float(order_count),
        "peer_order_revenue": float(order_rev),
        "peer_hour_revenue": float(hour_rev),
        "sum_seek_time": float(sum_seek),
        "sum_trans_time": float(sum_trans),
        "revenue_efficiency": float(score / total_time),
        "utilization_rate": float(sum_trans / total_time),
        "loss": float(np.mean(losses) if losses else 0.0),
        "attacked_steps": float(attacked_steps),
    }


def summarize(records: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(records)
    return {c: float(df[c].mean()) for c in df.columns}


def save_model(agent: DQNAgent, suffix: str, episode: int):
    filename = with_suffix(f"dqn_checkpoint_{episode}.pt", suffix)
    torch.save(agent.online.state_dict(), filename)
    print(f"[Save] Model saved to {filename}")


def save_q_table(q_agent: QLearningBaseline, suffix: str, episode: int):
    filename = with_suffix(f"qtable_checkpoint_{episode}.npy", suffix)
    np.save(filename, q_agent.q)
    print(f"[Save] Q-table saved to {filename}")


def train_and_evaluate(cfg: Config, suffix: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.seed)
    raw_env = oe.Orignal_Env(start_l=250, reward_dy=1)
    env_adp = EnvAdapter(env=raw_env, cfg=cfg)
    agent = DQNAgent(cfg=cfg)

    train_records: List[Dict[str, float]] = []
    for ep in range(cfg.episodes):
        start_grid = np.random.randint(1, 901)
        rec = run_episode_generic(env_adp, agent, cfg, start_grid, training=True, attack_mode=None)
        train_records.append(rec)
        agent.train_episodes = ep + 1   # 更新 episode 计数
        if (ep + 1) % cfg.checkpoint_every == 0:
            save_model(agent, suffix, ep+1)
        if (ep + 1) % cfg.train_log_every == 0:
            window = train_records[-cfg.train_log_every:]
            m = summarize(window)
            print(
                f"[Train DQN] Episode {ep+1:4d}/{cfg.episodes} | "
                f"score={m['score']:.2f}, orders={m['n_orders']:.2f}, "
                f"hour_rev={m['peer_hour_revenue']:.2f}, loss={m['loss']:.4f}, eps={agent.epsilon():.3f}"
            )

    scenarios = default_eval_scenarios(cfg, agent_type="dqn")
    eval_rows: List[Dict[str, float]] = []
    for name, mode, strength in scenarios:
        records = []
        for _ in range(cfg.eval_episodes):
            start_grid = np.random.randint(1, 901)
            rec = run_episode_generic(
                env_adp, agent, cfg, start_grid,
                training=False, attack_mode=mode, attack_strength=strength
            )
            records.append(rec)
        avg = summarize(records)
        avg["scenario"] = name
        eval_rows.append(avg)
        print(
            f"[Eval DQN] {name:24s} | score={avg['score']:.2f}, "
            f"hour_rev={avg['peer_hour_revenue']:.2f}, util={avg['utilization_rate']:.3f}"
        )

    train_df = pd.DataFrame(train_records)
    eval_df = pd.DataFrame(eval_rows)
    return train_df, eval_df


def train_and_evaluate_qlearning(cfg: Config, suffix: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.seed)
    raw_env = oe.Orignal_Env(start_l=250, reward_dy=1)
    q_agent = QLearningBaseline(cfg=cfg)

    train_records: List[Dict[str, float]] = []
    for ep in range(cfg.episodes):
        start_grid = np.random.randint(1, 901)
        rec = run_episode_generic(raw_env, q_agent, cfg, start_grid, training=True, attack_mode=None)
        train_records.append(rec)
        q_agent.train_episodes = ep + 1
        if (ep + 1) % cfg.checkpoint_every == 0:
            save_q_table(q_agent, suffix, ep+1)
        if (ep + 1) % cfg.train_log_every == 0:
            m = summarize(train_records[-cfg.train_log_every:])
            print(
                f"[Train QL] Episode {ep+1:4d}/{cfg.episodes} | "
                f"score={m['score']:.2f}, orders={m['n_orders']:.2f}, "
                f"hour_rev={m['peer_hour_revenue']:.2f}, eps={q_agent.epsilon():.3f}"
            )

    scenarios = default_eval_scenarios(cfg, agent_type="qlearning")
    eval_rows: List[Dict[str, float]] = []
    for name, mode, strength in scenarios:
        records = []
        for _ in range(cfg.eval_episodes):
            start_grid = np.random.randint(1, 901)
            rec = run_episode_generic(
                raw_env, q_agent, cfg, start_grid,
                training=False, attack_mode=mode, attack_strength=strength
            )
            records.append(rec)
        avg = summarize(records)
        avg["scenario"] = name
        eval_rows.append(avg)
        print(
            f"[Eval QL] {name:24s} | score={avg['score']:.2f}, "
            f"hour_rev={avg['peer_hour_revenue']:.2f}, util={avg['utilization_rate']:.3f}"
        )

    train_df = pd.DataFrame(train_records)
    eval_df = pd.DataFrame(eval_rows)
    return train_df, eval_df


def robustness_report(eval_df: pd.DataFrame) -> pd.DataFrame:
    clean = eval_df[eval_df["scenario"] == "clean"].iloc[0]
    rows = []
    for _, row in eval_df.iterrows():
        damage = 0.0
        if clean["score"] != 0:
            damage = (clean["score"] - row["score"]) / abs(clean["score"])
        rows.append({
            "scenario": row["scenario"],
            "score": row["score"],
            "peer_hour_revenue": row["peer_hour_revenue"],
            "revenue_efficiency": row["revenue_efficiency"],
            "utilization_rate": row["utilization_rate"],
            "performance_loss_ratio": max(0.0, damage),
        })
    return pd.DataFrame(rows).sort_values("performance_loss_ratio")


def build_comparison_table(dqn_report: pd.DataFrame, q_report: pd.DataFrame) -> pd.DataFrame:
    merged = dqn_report.merge(q_report, on="scenario", suffixes=("_dqn", "_qlearning"), how="inner")
    merged["score_gain_vs_q"] = merged["score_dqn"] - merged["score_qlearning"]
    merged["score_gain_ratio_vs_q"] = np.where(
        merged["score_qlearning"] == 0, 0.0,
        merged["score_gain_vs_q"] / np.abs(merged["score_qlearning"])
    )
    merged["robustness_advantage"] = merged["performance_loss_ratio_qlearning"] - merged["performance_loss_ratio_dqn"]
    return merged.sort_values("scenario")


def plot_revenue_curve(dqn_train_df, q_train_df, out_path, smooth_window=30):
    dqn_s = dqn_train_df["score"].rolling(window=smooth_window, min_periods=1).mean()
    q_s = q_train_df["score"].rolling(window=smooth_window, min_periods=1).mean()
    plt.figure(figsize=(10,5))
    plt.plot(dqn_s.values, label="DQN (rolling mean)", linewidth=2.0)
    plt.plot(q_s.values, label="Q-learning (rolling mean)", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Revenue score")
    plt.title("Training Revenue Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_attack_strength_curve(dqn_report, q_report, out_path):
    family_map = {
        "FGSM": ["clean", "whitebox_fgsm_low", "whitebox_fgsm_high"],
        "PGD": ["clean", "whitebox_pgd_low", "whitebox_pgd_high"],
        "Noise": ["clean", "blackbox_noise_low", "blackbox_noise_high"],
        "Boundary": ["clean", "blackbox_boundary_low", "blackbox_boundary_high"],
    }
    x = [0,1,2]
    plt.figure(figsize=(10,6))
    for family, scenarios in family_map.items():
        try:
            d = dqn_report.set_index("scenario").loc[scenarios, "performance_loss_ratio"].values
            q = q_report.set_index("scenario").loc[scenarios, "performance_loss_ratio"].values
        except KeyError:
            continue
        plt.plot(x, d, marker="o", linewidth=2.0, label=f"DQN-{family}")
        plt.plot(x, q, marker="x", linestyle="--", linewidth=1.8, label=f"Q-{family}")
    plt.xticks(x, ["clean", "low", "high"])
    plt.xlabel("Attack strength level")
    plt.ylabel("Performance loss ratio")
    plt.title("Attack Strength vs Performance Loss")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_clean_eval_only_bar(dqn_report, q_report, out_path):
    ds = float(dqn_report[dqn_report["scenario"] == "clean"].iloc[0]["score"])
    qs = float(q_report[q_report["scenario"] == "clean"].iloc[0]["score"])
    plt.figure(figsize=(5,5))
    plt.bar([0,1], [ds,qs], width=0.55, color=["#4C72B0","#55A868"])
    plt.xticks([0,1], ["DQN","Q-learning"])
    plt.ylabel("Score (clean eval)")
    plt.title("Clean-only evaluation: revenue score")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_no_adv_vs_adv_table(report_no_adv, report_adv):
    merged = report_no_adv.merge(report_adv, on="scenario", suffixes=("_no_adv", "_adv"), how="inner")
    merged["score_delta_adv_minus_no_adv"] = merged["score_adv"] - merged["score_no_adv"]
    merged["performance_loss_ratio_delta_no_adv_minus_adv"] = (
        merged["performance_loss_ratio_no_adv"] - merged["performance_loss_ratio_adv"]
    )
    return merged.sort_values("scenario")


def plot_no_adv_vs_adv(report_no_adv, report_adv, out_path):
    scenarios = report_no_adv["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.35
    fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)
    axes[0].bar(x - w/2, report_no_adv["score"].values, width=w, label="DQN no adv-train", color="#4C72B0")
    axes[0].bar(x + w/2, report_adv["score"].values, width=w, label="DQN with adv-train", color="#DD8452")
    axes[0].set_ylabel("Score")
    axes[0].set_title("DQN: clean & attack scenarios — revenue score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - w/2, report_no_adv["performance_loss_ratio"].values, width=w, label="no adv-train", color="#4C72B0")
    axes[1].bar(x + w/2, report_adv["performance_loss_ratio"].values, width=w, label="with adv-train", color="#DD8452")
    axes[1].set_ylabel("Performance loss ratio (vs clean)")
    axes[1].set_title("Relative performance drop (lower is better under attack)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios, rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    start = ostime.perf_counter()
    parser = argparse.ArgumentParser(description="DQN vs Q-learning 寻客鲁棒性实验")
    parser.add_argument("--no-adv-train", action="store_true", help="关闭训练阶段对抗正则")
    parser.add_argument("--adv-probabilistic", action="store_true", help="对抗训练按概率触发")
    parser.add_argument("--adv-prob", type=float, default=None, help="对抗训练概率")
    parser.add_argument("--output-suffix", type=str, default="", help="输出文件后缀")
    parser.add_argument("--compare-dqn-reports", nargs=2, metavar=("NO_ADV_REPORT_CSV", "ADV_REPORT_CSV"), help="对比两份报告")
    parser.add_argument("--eval-clean-only", action="store_true", help="评估只跑 clean")
    parser.add_argument(
        "--fast-safe",
        action="store_true",
        help="快速稳妥模式：在不降低攻击强度前提下减少迭代耗时（缩短训练预算、分阶段启用PGD、降低日志频率）",
    )
    args = parser.parse_args()
    if args.compare_dqn_reports:
        start_cmp = ostime.perf_counter()
        r_no = pd.read_csv(args.compare_dqn_reports[0])
        r_adv = pd.read_csv(args.compare_dqn_reports[1])
        cmp_tbl = build_no_adv_vs_adv_table(r_no, r_adv)
        suf = args.output_suffix
        out_csv = with_suffix("dqn_robustness_compare_no_adv_vs_adv.csv", suf)
        out_png = with_suffix("figure_no_adv_vs_adv_robustness.png", suf)
        cmp_tbl.to_csv(out_csv, index=False, encoding="utf-8-sig")
        plot_no_adv_vs_adv(r_no, r_adv, out_path=out_png)
        print("\n========= DQN no-adv-train vs adv-train =========")
        print(cmp_tbl.to_string(index=False))
        print(f"Saved: {out_csv}, {out_png}, elapsed {ostime.perf_counter()-start_cmp:.1f}s")
        sys.exit(0)

    cfg = Config()
    if args.fast_safe:
        apply_fast_safe_profile(cfg)
    cfg.eval_clean_only = args.eval_clean_only
    out_suf = args.output_suffix or ""
    if cfg.eval_clean_only and not out_suf:
        out_suf = "_eval_clean_only"
    if cfg.eval_clean_only:
        print("[Config] 评估阶段: 仅 clean 基线")
    if args.no_adv_train:
        cfg.adv_train_enabled = False
    if args.adv_probabilistic:
        if not cfg.adv_train_enabled:
            parser.error("--adv-probabilistic 与 --no-adv-train 冲突")
        cfg.adv_train_always = False
        if args.adv_prob is not None:
            cfg.adv_train_prob = args.adv_prob
    elif args.adv_prob is not None:
        parser.error("--adv-prob 需与 --adv-probabilistic 同时使用")

    print(f"[Config] 种子参数：seed={cfg.seed}")
    print("[Config] 训练期对抗正则:",
          "关闭" if not cfg.adv_train_enabled else
          f"全程触发" if cfg.adv_train_always else f"概率 {cfg.adv_train_prob}")
    print(f"[Config] 奖励缩放因子: {cfg.reward_scale} (1.0 表示不缩放)")
    print(f"[Config] 攻击强度: fgsm_eps={cfg.fgsm_eps}, pgd_eps={cfg.pgd_eps}")
    if args.fast_safe:
        print(
            "[Config] fast-safe 已启用: "
            f"episodes={cfg.episodes}, eval_episodes={cfg.eval_episodes}, "
            f"checkpoint_every={cfg.checkpoint_every}, adv_log_every={cfg.adv_log_every}, "
            f"pgd_after_ratio={cfg.adv_enable_pgd_after_ratio}, pgd_prob_after_warm={cfg.adv_pgd_prob_after_warm}"
        )

    dqn_train_df, dqn_eval_df = train_and_evaluate(cfg, suffix=out_suf)
    dqn_report_df = robustness_report(dqn_eval_df)

    q_train_df, q_eval_df = train_and_evaluate_qlearning(cfg, suffix=out_suf)
    q_report_df = robustness_report(q_eval_df)
    compare_df = build_comparison_table(dqn_report_df, q_report_df)

    f_train = with_suffix("dqn_train_log.csv", out_suf)
    f_eval = with_suffix("dqn_eval_scenarios.csv", out_suf)
    f_rep = with_suffix("dqn_robustness_report.csv", out_suf)
    f_qtrain = with_suffix("qlearning_train_log.csv", out_suf)
    f_qeval = with_suffix("qlearning_eval_scenarios.csv", out_suf)
    f_qrep = with_suffix("qlearning_robustness_report.csv", out_suf)
    f_cmp = with_suffix("dqn_vs_qlearning_comparison.csv", out_suf)
    f_fig_rev = with_suffix("figure_revenue_curve.png", out_suf)
    f_fig_atk = with_suffix("figure_attack_strength_loss_curve.png", out_suf)
    f_fig_clean = with_suffix("figure_eval_clean_only_dqn_vs_q.png", out_suf)

    dqn_train_df.to_csv(f_train, index=False, encoding="utf-8-sig")
    dqn_eval_df.to_csv(f_eval, index=False, encoding="utf-8-sig")
    dqn_report_df.to_csv(f_rep, index=False, encoding="utf-8-sig")
    q_train_df.to_csv(f_qtrain, index=False, encoding="utf-8-sig")
    q_eval_df.to_csv(f_qeval, index=False, encoding="utf-8-sig")
    q_report_df.to_csv(f_qrep, index=False, encoding="utf-8-sig")
    compare_df.to_csv(f_cmp, index=False, encoding="utf-8-sig")

    plot_revenue_curve(dqn_train_df, q_train_df, out_path=f_fig_rev)
    if cfg.eval_clean_only:
        plot_clean_eval_only_bar(dqn_report_df, q_report_df, out_path=f_fig_clean)
    else:
        plot_attack_strength_curve(dqn_report_df, q_report_df, out_path=f_fig_atk)

    print("\n========= DQN Robustness Report =========")
    print(dqn_report_df.to_string(index=False))
    print("\n========= Q-learning Robustness Report =========")
    print(q_report_df.to_string(index=False))
    print("\n========= DQN vs Q-learning Comparison =========")
    print(compare_df.to_string(index=False))
    print("Saved CSV:", ", ".join([f_train, f_eval, f_rep, f_qtrain, f_qeval, f_qrep, f_cmp]))
    print("Saved Figures:", ", ".join([f_fig_rev, f_fig_clean if cfg.eval_clean_only else f_fig_atk]))
    print(f"Elapsed: {ostime.perf_counter()-start:.1f}s")