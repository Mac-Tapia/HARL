"""
harl_wrapper.py
==============
Adapta LimaTransporteEnv (PettingZoo AEC) a la interfaz paralela de HARL.

Interfaz que HARL espera:
  reset() → (obs_list, share_obs_list, avail_actions_list)
  step(actions) → (obs_list, share_obs_list, rewards, dones, infos, avail_actions)

Conversión AEC → paralelo:
  Un step() del wrapper completa una ronda completa (los 3 agentes actúan una vez).
"""

import numpy as np
from gymnasium import spaces

from harl.envs.lima_transporte.lima_env import LimaTransporteEnv, AGENTES, OBS_DIM

GLOBAL_DIM = OBS_DIM * len(AGENTES)   # 57 = 19x3: estado global base sin demanda predictiva
COMMUNICATION_DIM = 6
REWARD_MODES = {"individual", "team_mean", "team_sum"}
COMMUNICATION_MODES = {"none", "ctde_state", "ctde_message"}


class LimaHARLWrapper:
    """
    Wrapper de LimaTransporteEnv compatible con OnPolicyHARunner de HARL.

    Atributos requeridos por HARL:
      n_agents              : int
      observation_space     : list[gym.Space]
      action_space          : list[gym.Space]
      share_observation_space: list[gym.Space]
    """

    def __init__(self, env_args: dict | None = None):
        env_args = env_args or {}
        self.reward_mode = str(env_args.get("reward_mode", "individual")).lower().strip()
        if self.reward_mode == "team":
            self.reward_mode = "team_mean"
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode invalido: {self.reward_mode}. "
                f"Usa uno de: {sorted(REWARD_MODES)}."
            )

        self.communication_mode = str(env_args.get("communication_mode", "ctde_state")).lower().strip()
        if self.communication_mode not in COMMUNICATION_MODES:
            raise ValueError(
                f"communication_mode invalido: {self.communication_mode}. "
                f"Usa uno de: {sorted(COMMUNICATION_MODES)}."
            )

        self._env = LimaTransporteEnv(
            max_steps=env_args.get("max_steps", 672),
            data_dir=env_args.get("data_dir", "data/processed"),
            dataset_split=env_args.get("dataset_split", env_args.get("split", "train")),
            causal_obs=env_args.get("causal_obs", env_args.get("production_obs", False)),
            use_demand_forecast=env_args.get("use_demand_forecast", False),
            seed=env_args.get("seed", 42),
        )
        self.n_agents = len(AGENTES)
        self.agents   = AGENTES.copy()
        self._seed    = env_args.get("seed", 42)
        self.obs_dim = int(getattr(self._env, "obs_dim", OBS_DIM))
        self.global_dim = self.obs_dim * self.n_agents
        self.share_dim = self.global_dim + (
            COMMUNICATION_DIM if self.communication_mode == "ctde_message" else 0
        )

        # Espacios por agente (heterogéneos)
        self.observation_space = [
            self._env.observation_spaces[ag] for ag in AGENTES
        ]
        self.action_space = [
            self._env.action_spaces[ag] for ag in AGENTES
        ]
        # Estado compartido para CTDE. El actor conserva obs local;
        # el mensaje agregado solo alimenta al critico centralizado.
        _glob = spaces.Box(low=-np.inf, high=np.inf,
                           shape=(self.share_dim,), dtype=np.float32)
        self.share_observation_space = [_glob] * self.n_agents

    # ── Interfaz requerida por HARL ─────────────────────────────────────────

    def seed(self, seed: int):
        self._seed = seed

    def reset(self):
        """Reinicia el entorno y devuelve observaciones iniciales."""
        self._env.reset(seed=self._seed)
        self._terminal_infos = [{} for _ in AGENTES]  # caché de infos del paso terminal
        obs       = [self._env.observe(ag).copy() for ag in AGENTES]
        gs        = self._shared_state()
        share_obs = [gs.copy() for _ in AGENTES]
        return obs, share_obs, self._avail_actions()

    def step(self, actions):
        """
        Ejecuta una ronda completa (los 3 agentes actúan una vez en orden AEC).

        Parámetros
        ----------
        actions : array-like de longitud n_agents (una acción por agente)

        Devuelve
        --------
        obs, share_obs, rewards (n_agents,1), dones (n_agents,), infos, avail_actions
        """
        rewards = [0.0] * self.n_agents
        dones   = [False] * self.n_agents
        infos   = [{} for _ in range(self.n_agents)]

        for i, ag in enumerate(AGENTES):
            if not self._env.agents:
                # Episodio ya terminó — devolver infos del paso terminal cacheadas
                dones = [True] * self.n_agents
                infos = [d.copy() for d in self._terminal_infos]
                break
            self._env.step(self._coerce_action(actions[i], self.action_space[i]))
            rewards[i] = float(self._env.rewards.get(ag, 0.0))
            dones[i]   = bool(self._env.terminations.get(ag, False))
            infos[i]   = self._env.infos.get(ag, {})

        # Termination simultánea: cuando el entorno marca a todos los agentes
        # como terminados, HARL espera dones=True para toda la ronda.
        env_done = bool(self._env.agents) and all(
            self._env.terminations.get(ag, False)
            or self._env.truncations.get(ag, False)
            for ag in self._env.agents
        )
        if (not self._env.agents or env_done) and any(infos):
            dones = [True] * self.n_agents
            self._terminal_infos = [
                self._env.infos.get(ag, {}).copy() for ag in AGENTES
            ]

        local_rewards = [float(r) for r in rewards]
        rewards, team_reward = self._apply_reward_mode(local_rewards)
        message = self._communication_vector()
        for i, info in enumerate(infos):
            info["reward_mode"] = self.reward_mode
            info["local_reward_round"] = local_rewards[i]
            info["team_reward_round"] = team_reward
            info["communication_mode"] = self.communication_mode
            info["shared_message"] = message.tolist()

        obs       = [self._env.observe(ag).copy() for ag in AGENTES]
        gs        = self._shared_state(message)
        share_obs = [gs.copy() for _ in AGENTES]
        rewards_out = [[r] for r in rewards]   # (n_agents, 1)

        return obs, share_obs, rewards_out, dones, infos, self._avail_actions()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_action(action, space):
        arr = np.asarray(action)
        if space.__class__.__name__ == "Box":
            out = np.zeros(space.shape, dtype=np.float32)
            flat = arr.astype(np.float32, copy=False).reshape(-1)
            n = min(flat.size, out.size)
            if n:
                out.reshape(-1)[:n] = flat[:n]
            return np.clip(out, space.low, space.high).astype(np.float32)

        if arr.ndim == 0:
            return int(arr.item())
        flat = arr.reshape(-1)
        if flat.size == 1:
            return int(flat[0])
        return int(np.argmax(flat))

    def _avail_actions(self):
        """
        Máscara de acciones disponibles, rellena hasta max_action_dim para que
        np.array() pueda convertir la lista en un array homogéneo.

        buses     : [1,1,1,1,1,1, 0,0,0,0,0,0]   (6 válidas + 6 relleno)
        combis    : [1,1,1,1,1,1, 1,1,1,1,1,1]   (12 válidas)
        mototaxis : [1,1,1,1,1,1, 1,1,1,1,1,1]   (12 válidas)
        """
        if not all(self._env.action_spaces[ag].__class__.__name__ == "Discrete" for ag in AGENTES):
            return None

        max_dim = max(self._env.action_spaces[ag].n for ag in AGENTES)
        result = []
        for ag in AGENTES:
            n = self._env.action_spaces[ag].n
            avail = [1] * n + [0] * (max_dim - n)
            result.append(avail)
        return result

    def _apply_reward_mode(self, rewards: list[float]) -> tuple[list[float], float | None]:
        local_rewards = [float(r) for r in rewards]
        if self.reward_mode == "individual":
            return local_rewards, None

        if self.reward_mode == "team_sum":
            team_reward = float(np.sum(local_rewards))
        else:
            team_reward = float(np.mean(local_rewards))
        return [team_reward] * self.n_agents, team_reward

    def _communication_vector(self) -> np.ndarray:
        stats = self._env.get_episode_stats()["sistema"]
        max_agent_steps = max(int(self._env.max_steps) * self.n_agents, 1)
        alertas_emitidas = float(stats.get("alertas_emitidas", 0))
        compliance_fallos = float(stats.get("compliance_fallos", 0))
        return np.array(
            [
                float(stats.get("tasa_deteccion", 0.0)),
                float(stats.get("fp_por_ataque", 0.0)),
                float(stats.get("tasa_alerta_efectiva", 0.0)),
                min(alertas_emitidas / max_agent_steps, 1.0),
                compliance_fallos / max(alertas_emitidas, 1.0),
                min(
                    float(getattr(self._env, "_paso_actual", 0))
                    / max(int(self._env.max_steps), 1),
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    def _shared_state(self, message: np.ndarray | None = None) -> np.ndarray:
        base_state = self._env.get_global_state()
        if self.communication_mode != "ctde_message":
            return base_state.astype(np.float32, copy=False)

        if message is None:
            message = self._communication_vector()
        return np.concatenate([base_state, message], dtype=np.float32)

    def get_episode_stats(self) -> dict:
        return self._env.get_episode_stats()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
