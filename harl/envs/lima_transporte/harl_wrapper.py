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

GLOBAL_DIM = OBS_DIM * len(AGENTES)   # 57 = 19×3 — estado global para el crítico centralizado


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
        self._env = LimaTransporteEnv(
            max_steps=env_args.get("max_steps", 672),
            data_dir=env_args.get("data_dir", "data/processed"),
            seed=env_args.get("seed", 42),
        )
        self.n_agents = len(AGENTES)
        self.agents   = AGENTES.copy()
        self._seed    = env_args.get("seed", 42)

        # Espacios por agente (heterogéneos)
        self.observation_space = [
            self._env.observation_spaces[ag] for ag in AGENTES
        ]
        self.action_space = [
            self._env.action_spaces[ag] for ag in AGENTES
        ]
        # Estado global compartido dim=57 para el crítico centralizado (OBS_DIM=19 × 3 agentes)
        _glob = spaces.Box(low=-np.inf, high=np.inf,
                           shape=(GLOBAL_DIM,), dtype=np.float32)
        self.share_observation_space = [_glob] * self.n_agents

    # ── Interfaz requerida por HARL ─────────────────────────────────────────

    def seed(self, seed: int):
        self._seed = seed

    def reset(self):
        """Reinicia el entorno y devuelve observaciones iniciales."""
        self._env.reset(seed=self._seed)
        self._terminal_infos = [{} for _ in AGENTES]  # caché de infos del paso terminal
        obs       = [self._env.observe(ag).copy() for ag in AGENTES]
        gs        = self._env.get_global_state()
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

        # Termination simultánea: cuando el último agente termina (agents=[]),
        # todos los agentes terminan al mismo tiempo → forzar dones y cachear infos.
        if not self._env.agents and any(infos):
            dones = [True] * self.n_agents
            self._terminal_infos = [d.copy() for d in infos]

        obs       = [self._env.observe(ag).copy() for ag in AGENTES]
        gs        = self._env.get_global_state()
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

    def get_episode_stats(self) -> dict:
        return self._env.get_episode_stats()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
