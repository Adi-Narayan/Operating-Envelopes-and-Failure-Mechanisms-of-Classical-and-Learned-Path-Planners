"""
Stronger learned baselines for the 2D comparison: TD3 and SAC with the same LSTM
observation encoder, observation (12-D), action (heading change), reward and
training loop as ARC. Only the learning algorithm differs.

Both agents expose the ARCAgent interface used by train.py and the evaluation
harness: select_action(history, hidden, noise_scale) -> (action, hidden),
update(batch, gamma, tau) -> dict, save(path), load(path), .device, eval().
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Encoder(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, lstm_h, mlp_h, stochastic=False):
        super().__init__()
        self.enc = Encoder(obs_dim, lstm_h)
        self.body = nn.Sequential(nn.Linear(lstm_h, mlp_h), nn.ReLU(), nn.Linear(mlp_h, mlp_h), nn.ReLU())
        self.mu = nn.Linear(mlp_h, act_dim)
        self.log_std = nn.Linear(mlp_h, act_dim) if stochastic else None

    def forward(self, h):
        z = self.body(self.enc(h))
        if self.log_std is None:
            return torch.tanh(self.mu(z))
        return self.mu(z), torch.clamp(self.log_std(z), LOG_STD_MIN, LOG_STD_MAX)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, lstm_h, mlp_h):
        super().__init__()
        self.enc = Encoder(obs_dim, lstm_h)
        self.body = nn.Sequential(nn.Linear(lstm_h + act_dim, mlp_h), nn.ReLU(),
                                  nn.Linear(mlp_h, mlp_h), nn.ReLU(), nn.Linear(mlp_h, 1))

    def forward(self, h, a):
        return self.body(torch.cat([self.enc(h), a], dim=1))


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1 - tau).add_(tau * s.data)


class _Base:
    algo = None

    def __init__(self, obs_dim, act_dim, config, device):
        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.cfg = config
        self.lstm_h = config["lstm_hidden_size"]
        self.mlp_h = config["actor_hidden_size"]
        self.lr = config.get("learning_rate", 3e-4)
        self.n_updates = 0

    def eval(self):
        for m in self._modules():
            m.eval()

    def save(self, path):
        torch.save({"algo": self.algo, "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                    "cfg": {k: self.cfg[k] for k in ("lstm_hidden_size", "actor_hidden_size")},
                    "state": {k: v.state_dict() for k, v in self._state().items()}}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        assert ck["algo"] == self.algo, (ck["algo"], self.algo)
        for k, v in self._state().items():
            if k in ck["state"]:
                v.load_state_dict(ck["state"][k])
            elif not ck.get("actor_only"):
                raise KeyError(f"{path}: missing '{k}'")
        # actor-only files (export_actor) hold just what evaluation uses: the deterministic actor


class TD3Agent(_Base):
    algo = "td3"

    def __init__(self, obs_dim, act_dim, config, device, policy_delay=2, target_noise=0.2, noise_clip=0.5):
        super().__init__(obs_dim, act_dim, config, device)
        self.actor = Actor(obs_dim, act_dim, self.lstm_h, self.mlp_h).to(device)
        self.q1 = Critic(obs_dim, act_dim, self.lstm_h, self.mlp_h).to(device)
        self.q2 = Critic(obs_dim, act_dim, self.lstm_h, self.mlp_h).to(device)
        self.actor_t, self.q1_t, self.q2_t = (copy.deepcopy(m) for m in (self.actor, self.q1, self.q2))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.opt_c = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr)
        self.policy_delay, self.target_noise, self.noise_clip = policy_delay, target_noise, noise_clip
        self._last_actor_loss = 0.0

    def _modules(self):
        return [self.actor, self.q1, self.q2]

    def _state(self):
        return {"actor": self.actor, "q1": self.q1, "q2": self.q2, "actor_t": self.actor_t,
                "q1_t": self.q1_t, "q2_t": self.q2_t, "opt_a": self.opt_a, "opt_c": self.opt_c}

    @torch.no_grad()
    def select_action(self, history, hidden=None, noise_scale=0.0):
        if history.dim() == 2:
            history = history.unsqueeze(0)
        a = self.actor(history)
        if noise_scale > 0:
            a = torch.clamp(a + noise_scale * torch.randn_like(a), -1, 1)
        return a.cpu().numpy()[0], None

    def update(self, batch, gamma, tau):
        h, a, r, nh, d = batch["history"], batch["action"], batch["reward"], batch["next_history"], batch["done"]
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_t(nh) + noise).clamp(-1, 1)
            tq = r + (1 - d) * gamma * torch.min(self.q1_t(nh, na), self.q2_t(nh, na))
        q1, q2 = self.q1(h, a), self.q2(h, a)
        lc = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
        self.opt_c.zero_grad(); lc.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.opt_c.step()
        self.n_updates += 1
        if self.n_updates % self.policy_delay == 0:
            la = -self.q1(h, self.actor(h)).mean()
            self.opt_a.zero_grad(); la.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()
            self._last_actor_loss = la.item()
            for t, s in ((self.actor_t, self.actor), (self.q1_t, self.q1), (self.q2_t, self.q2)):
                soft_update(t, s, tau)
        return {"critic_loss": lc.item(), "actor_loss": self._last_actor_loss, "avg_q": q1.mean().item()}


class SACAgent(_Base):
    algo = "sac"

    def __init__(self, obs_dim, act_dim, config, device):
        super().__init__(obs_dim, act_dim, config, device)
        self.actor = Actor(obs_dim, act_dim, self.lstm_h, self.mlp_h, stochastic=True).to(device)
        self.q1 = Critic(obs_dim, act_dim, self.lstm_h, self.mlp_h).to(device)
        self.q2 = Critic(obs_dim, act_dim, self.lstm_h, self.mlp_h).to(device)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -float(act_dim)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.opt_c = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def _modules(self):
        return [self.actor, self.q1, self.q2]

    def _state(self):
        return {"actor": self.actor, "q1": self.q1, "q2": self.q2, "q1_t": self.q1_t, "q2_t": self.q2_t,
                "opt_a": self.opt_a, "opt_c": self.opt_c, "opt_al": self.opt_al,
                "log_alpha": _AlphaHolder(self)}

    def _sample(self, h):
        mu, log_std = self.actor(h)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(1, keepdim=True)
        return a, logp

    @torch.no_grad()
    def select_action(self, history, hidden=None, noise_scale=0.0):
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if noise_scale > 0:
            a, _ = self._sample(history)
        else:
            a = torch.tanh(self.actor(history)[0])
        return a.cpu().numpy()[0], None

    def update(self, batch, gamma, tau):
        h, a, r, nh, d = batch["history"], batch["action"], batch["reward"], batch["next_history"], batch["done"]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlogp = self._sample(nh)
            tq = r + (1 - d) * gamma * (torch.min(self.q1_t(nh, na), self.q2_t(nh, na)) - alpha * nlogp)
        q1, q2 = self.q1(h, a), self.q2(h, a)
        lc = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
        self.opt_c.zero_grad(); lc.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.opt_c.step()
        pa, logp = self._sample(h)
        la = (alpha * logp - torch.min(self.q1(h, pa), self.q2(h, pa))).mean()
        self.opt_a.zero_grad(); la.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_a.step()
        lal = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()
        soft_update(self.q1_t, self.q1, tau); soft_update(self.q2_t, self.q2, tau)
        self.n_updates += 1
        return {"critic_loss": lc.item(), "actor_loss": la.item(), "avg_q": q1.mean().item()}


class _AlphaHolder:
    """state_dict / load_state_dict wrapper so log_alpha is saved with the agent."""

    def __init__(self, agent):
        self.agent = agent

    def state_dict(self):
        return {"log_alpha": self.agent.log_alpha.detach().clone()}

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.agent.log_alpha.copy_(sd["log_alpha"])


ALGOS = {"td3": TD3Agent, "sac": SACAgent}

# Hyperparameters used for the stronger baselines (standard TD3/SAC choices; same
# encoder sizes, observation, reward, action and 300-episode budget as ARC).
BASELINE_TRAIN_CONFIG = {
    "learning_rate": 3e-4,     # Adam
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "memory_capacity": 50_000,
    "exploration_noise": 0.1,  # TD3 Gaussian action noise; SAC samples its own policy
}


def export_actor(src, dst):
    """Write an inference-only copy of a TD3/SAC checkpoint (actor weights only, ~1/10 the size).
    Evaluation (select_action with noise 0) uses only the actor, so results are unchanged;
    training cannot be resumed from the exported file."""
    ck = torch.load(src, map_location="cpu", weights_only=True)
    out = {k: ck[k] for k in ("algo", "obs_dim", "act_dim", "cfg")}
    out["state"] = {"actor": ck["state"]["actor"]}
    out["actor_only"] = True
    torch.save(out, dst)


def load_agent(path, device=None):
    """Load an ARC, TD3 or SAC checkpoint and return an agent with the ARCAgent interface."""
    device = device or torch.device("cpu")
    ck = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ck, dict) and "algo" in ck:
        cfg = dict(ck["cfg"])
        agent = ALGOS[ck["algo"]](ck["obs_dim"], ck["act_dim"], cfg, device)
        agent.load(path)
        agent.eval()
        return agent
    return None
