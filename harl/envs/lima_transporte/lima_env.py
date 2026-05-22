"""
lima_env.py
===========
Entorno Dec-POMDP de Lima Metropolitana — API AEC de PettingZoo.

Tres agentes heterogéneos cooperativos bajo paradigma CTDE:
  - buses:     observa rutas troncales (Lima Norte, Breña, SJL, VES)
  - combis:    observa rutas informales (Comas, SMP, Los Olivos)
  - mototaxis: observa zonas periféricas (VES, SJL, Puente Piedra)

Cada agente decide si emitir alerta, con qué severidad, y cómo priorizar
apoyo, despacho, cobertura y coordinación entre ejes.
El crítico centralizado observa el estado global de los 3 agentes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESPACIO DE OBSERVACIÓN — OBS_DIM = 19 base / 20 con demanda predictiva
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Contexto espacio-temporal (base):
  [0]  zona_id_n       — distrito normalizado ∈ [0,1]
  [1]  hora_sin        — sin(2π·hora/24)  codificación cíclica
  [2]  hora_cos        — cos(2π·hora/24)  evita discontinuidad hora 23→0
  [3]  dia_sin         — sin(2π·dia/7)    codificación cíclica
  [4]  dia_cos         — cos(2π·dia/7)    lunes=0 … domingo=6
  [5]  hist7d_n        — ataques últimos 7 días normalizados ∈ [0,1]
  [6]  alerta_prev     — alerta emitida en el paso anterior (binario)
  [7]  riesgo_zona     — factor de riesgo del distrito ∈ [0,1]
  [8]  flag_visual     — señal de visión artificial: TP≈0.88, FP≈0.05

  Gravedad del evento (del dataset scrapeado):
  [9]  homicidio       — evento letal en registro ∈ {0,1}
  [10] herido          — víctima herida (sin homicidio) ∈ {0,1}
  [11] arma_id_n       — tipo arma normalizado: AF=0.33, Blanca=0.66, Otros=1.0

  Contexto de extorsión:
  [12] extorsion_rel   — ataque relacionado con extorsión ∈ {0,1}
  [13] monto_cupo_n    — monto cupo normalizado ∈ [0,1]  (máx=100000 soles)
  [14] tipo_ext_n      — tipo extorsión normalizado ∈ [0,1]
                         0=no_id, 0.2=amenaza, 0.4=extorsion_gral,
                         0.6=cobro_cupo, 0.8=represalia, 1.0=atentado

  Modus operandi del atacante:
  [15] vehiculo_id_n   — vehículo sicario normalizado ∈ [0,1]
                         0=no_id, 0.25=a_pie, 0.5=auto, 0.75=moto_lineal, 1.0=mototaxi

  Perfil de la víctima:
  [16] victima_sexo_n  — sexo víctima: 0=no_id/masculino, 1=femenino
  [17] victima_edad_n  — edad normalizada ∈ [0,1]  (máx=90 años; -1→0)
  [18] victima_menor   — víctima menor de edad ∈ {0,1}
  [19] demanda_pred_n  — demanda pronosticada normalizada ∈ [0,1]
                         Solo se incluye si use_demand_forecast=True.

  Justificación codificación cíclica: hora=23 y hora=0 son adyacentes;
  la normalización lineal crea distancia artificial de 23 unidades.
  (Ref: spatiotemporal crime RL, ScienceDirect 2025)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESPACIO DE ACCIÓN — continuo por rol operativo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cada agente usa Box([-1]*6, [1]*6):
    a[0] intensidad_alerta     : <=0 no alerta, >0 emite alerta
    a[1] severidad             : continuo reescalado a severidad 1..3 si hay alerta
    a[2] accion_auxiliar       : >0 activa apoyo/vigilancia/maniobra auxiliar
    a[3] prioridad_despacho    : [-1,1] -> [0,1] urgencia operativa del recurso
    a[4] cobertura_zonal       : [-1,1] redistribuye cobertura territorial
    a[5] coordinacion_intereje : [-1,1] solicita apoyo de otro eje/modalidad

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMPENSA MULTI-OBJETIVO — R ∈ ℝ³  (calibrada a datos Lima MP 2026)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  obj[0] — Detección : TP=+5.0, FN=-3.0   ratio FP:FN = 1:6
  obj[1] — Precisión : TN=+0.2, FP=-0.9   penaliza falsas alarmas
  obj[2] — Gravedad  : homicidio +10/-10   escala costo real del evento letal

  Ratio FP:FN = 1:6 (cyber-defense literature, arXiv:2410.17351)
  Escala TP homicidio = 10× vs ataque común (vida > recursos)

Pesos de escalarización  w · R = r_scalar  (heterogéneos por rol):
  buses:     w = [0.45, 0.20, 0.35]  — mayor peso gravedad (rutas troncales)
  combis:    w = [0.40, 0.25, 0.35]  — equilibrado con acento en gravedad
  mototaxis: w = [0.35, 0.35, 0.30]  — mayor peso precisión (menos recursos)

  Σw = 1.0 por agente. Diseño basado en:
  - Zhong et al. (2024) HARL/HAPPO — JMLR 25(32)
  - MO-MIX: Multi-Objective MARL — IEEE TPAMI 2023
  - MORL Decomposition Taxonomy — arXiv:2311.12495 (ESR approach)
  - Cyber Network Defense MARL — arXiv:2410.17351 (penalty ratio)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces


# ── Constantes del entorno ────────────────────────────────────────────────

AGENTES   = ["buses", "combis", "mototaxis"]
N_ZONAS   = 24
OBS_DIM_BASE = 19
DEMAND_FORECAST_OBS_IDX = 19
OBS_DIM   = OBS_DIM_BASE   # Compatibilidad: dimension base sin demanda predictiva.
OBS_DIM_WITH_DEMAND = OBS_DIM_BASE + 1
ACTION_DIM = 6
ACT_DIM   = {ag: ACTION_DIM for ag in AGENTES}
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
NON_CAUSAL_OBS_IDX = (9, 10, 11, 16, 17, 18)

# ── Recompensa Multi-Objetivo  R ∈ ℝ³ = [deteccion, precision, gravedad] ─────
# Calibrada a datos Lima MP 2026 y literatura MARL de ciberseguridad:
#   ratio FP:FN = 1:6  (arXiv:2410.17351 — cyber defense)
#   término de gravedad homicidio = ±10  (costo de vida > costo recurso)
#
#         evento              [deteccion, precision, gravedad]
RECOMPENSAS_MO = {
    "homicidio_prevenido":    np.array([+5.0,  0.0, +10.0]),  # TP + homicidio
    "ataque_prevenido":       np.array([+5.0,  0.0,   0.0]),  # TP sin homicidio
    "falsa_alarma":           np.array([ 0.0, -0.9,   0.0]),  # FP: penalidad reforzada anti-sobreactuacion
    "homicidio_no_detectado": np.array([-3.0,  0.0, -10.0]),  # FN + homicidio
    "ataque_no_detectado":    np.array([-3.0,  0.0,   0.0]),  # FN sin homicidio
    "silencio_ok":            np.array([ 0.0, +0.2,   0.0]),  # TN correcto
}

N_OBJETIVOS       = 3
NOMBRES_OBJETIVOS = ["deteccion", "precision", "gravedad"]

# Pesos de escalarización  w · R = r_scalar  (Σw = 1.0 por agente)
# Diseño heterogéneo por rol operativo y capacidad de cada modalidad:
PESOS_AGENTE = {
    "buses":     np.array([0.45, 0.20, 0.35]),  # rutas troncales: acento en gravedad
    "combis":    np.array([0.40, 0.25, 0.35]),  # equilibrado con acento en gravedad
    "mototaxis": np.array([0.35, 0.35, 0.30]),  # periferias: acento en precisión
}

# Recompensas escalares legacy (render/logs, usando pesos de combis como referencia)
RECOMPENSAS = {k: float(np.dot(v, PESOS_AGENTE["combis"])) for k, v in RECOMPENSAS_MO.items()}

ZONAS_POR_AGENTE = {
    "buses":     [0, 1, 3, 4, 7, 8, 12, 13],
    "combis":    [1, 2, 5, 6, 10, 11, 14, 15],
    "mototaxis": [3, 6, 9, 14, 15, 16, 0, 2],
}

FACTOR_RIESGO = {
    0: 1.00, 1: 0.90, 2: 0.85, 3: 0.82, 4: 0.78,
    5: 0.75, 6: 0.72, 7: 0.70, 8: 0.68, 9: 0.65,
    10: 0.60, 11: 0.55, 12: 0.50, 13: 0.48, 14: 0.45,
    15: 0.42, 16: 0.40, 17: 0.38, 18: 0.10, 19: 0.08,
    20: 0.09, 21: 0.07, 22: 0.12, 23: 0.10,
}

HORAS_PICO   = {5, 6, 7, 17, 18, 19, 20, 21}


class LimaTransporteEnv(AECEnv):
    """
    Entorno PettingZoo AEC para el sistema MADRL de Lima.

    Parámetros
    ----------
    max_steps : int
        Pasos por episodio (default 672 = 28 días × 24 horas).
    data_dir : str
        Directorio con los datasets procesados.
    dataset_split : str
        Split a cargar: "train" para entrenamiento o "val" para evaluacion holdout.
    causal_obs : bool
        Si True, elimina variables post-evento que no existen antes de decidir
        una alerta real: homicidio, herido, arma y perfil de victima.
    seed : int
        Semilla para reproducibilidad.
    render_mode : str | None
        "human" para imprimir estado en consola.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "lima_transporte_v1",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        max_steps: int = 672,
        data_dir: str = "data/processed",
        dataset_split: str = "train",
        causal_obs: bool = False,
        use_demand_forecast: bool = False,
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.max_steps    = max_steps
        self.data_dir     = Path(data_dir)
        self.dataset_split = str(dataset_split or "train").lower().strip()
        if self.dataset_split not in {"train", "val"}:
            raise ValueError("dataset_split debe ser 'train' o 'val'")
        self.causal_obs   = bool(causal_obs)
        self.use_demand_forecast = bool(use_demand_forecast)
        self.obs_dim = OBS_DIM_WITH_DEMAND if self.use_demand_forecast else OBS_DIM_BASE
        self.seed_val     = seed
        self.render_mode  = render_mode
        self._rng         = np.random.default_rng(seed)

        # Agentes
        self.possible_agents = AGENTES.copy()
        self.agents          = AGENTES.copy()

        # Espacios de observación — Vector base de 19 dimensiones.
        # [zona, h_sin, h_cos, d_sin, d_cos, hist, alerta, riesgo, visual,
        #  homicidio, herido, arma,
        #  extorsion, monto, tipo_ext,
        #  vehiculo,
        #  sexo, edad, menor,
        #  demanda_pred_n opcional]
        obs_low  = np.array(
            [0, -1, -1, -1, -1, 0, 0, 0, 0,
             0,  0,  0,
             0,  0,  0,
             0,
             0,  0,  0], dtype=np.float32)
        obs_high = np.array(
            [1,  1,  1,  1,  1, 1, 1, 1, 1,
             1,  1,  1,
             1,  1,  1,
             1,
             1,  1,  1], dtype=np.float32)
        if self.use_demand_forecast:
            obs_low = np.append(obs_low, np.float32(0.0)).astype(np.float32)
            obs_high = np.append(obs_high, np.float32(1.0)).astype(np.float32)
        self.observation_spaces = {
            ag: spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            for ag in self.possible_agents
        }

        # Espacios de acción continuos:
        # [alerta, severidad, apoyo_auxiliar, prioridad_despacho,
        #  cobertura_zonal, coordinacion_intereje]
        self.action_spaces = {
            ag: spaces.Box(
                low=ACTION_LOW,
                high=ACTION_HIGH,
                shape=(ACTION_DIM,),
                dtype=np.float32,
            )
            for ag in self.possible_agents
        }

        # Cargar datos de entrenamiento
        self._datos = self._cargar_datos()
        self._indice = {ag: 0 for ag in AGENTES}

        # Estado interno
        self._paso_actual   = 0
        self._hora_actual   = 0
        self._dia_actual    = 0
        self._historial     = {ag: [] for ag in AGENTES}
        self._alerta_previa = {ag: 0 for ag in AGENTES}

        # Selector de agentes (turno AEC)
        self._selector = agent_selector(self.possible_agents)

        # Acumuladores de episodio
        self._ep_recompensas   = {ag: 0.0 for ag in AGENTES}
        self._ep_homicidios_detectados   = {ag: 0 for ag in AGENTES}
        self._ep_homicidios_no_detectados = {ag: 0 for ag in AGENTES}
        self._ep_ataques_detectados   = {ag: 0 for ag in AGENTES}
        self._ep_ataques_no_detectados = {ag: 0 for ag in AGENTES}
        self._ep_ataques_alertados_modelo = {ag: 0 for ag in AGENTES}
        self._ep_ataques_no_alertados_modelo = {ag: 0 for ag in AGENTES}
        self._ep_homicidios_alertados_modelo = {ag: 0 for ag in AGENTES}
        self._ep_falsas_alarmas = {ag: 0 for ag in AGENTES}
        self._ep_falsas_alarmas_modelo = {ag: 0 for ag in AGENTES}
        self._ep_alertas_emitidas = {ag: 0 for ag in AGENTES}
        self._ep_alertas_efectivas = {ag: 0 for ag in AGENTES}
        self._ep_compliance_fallos = {ag: 0 for ag in AGENTES}
        self._ep_prioridad_despacho_sum = {ag: 0.0 for ag in AGENTES}
        self._ep_cobertura_zonal_abs_sum = {ag: 0.0 for ag in AGENTES}
        self._ep_coord_intereje_sum = {ag: 0.0 for ag in AGENTES}
        self._ep_despachos_priorizados = {ag: 0 for ag in AGENTES}
        self._ep_redistribuciones_cobertura = {ag: 0 for ag in AGENTES}
        self._ep_solicitudes_intereje = {ag: 0 for ag in AGENTES}
        # Acumuladores por objetivo (multi-objetivo)
        self._ep_obj = {ag: np.zeros(N_OBJETIVOS) for ag in AGENTES}
        # Última recompensa vectorial por agente
        self.reward_vectors: dict[str, np.ndarray] = {
            ag: np.zeros(N_OBJETIVOS) for ag in AGENTES
        }

    # ── Carga de datos ─────────────────────────────────────────────────────

    def _cargar_datos(self) -> dict[str, pd.DataFrame]:
        datos = {}
        for ag in AGENTES:
            modalidad = ag  # buses, combis, mototaxis
            path_clean = self.data_dir / f"dataset_{modalidad}_{self.dataset_split}.csv"
            if not path_clean.exists():
                raise FileNotFoundError(
                    f"No se encontró el dataset requerido: {path_clean}. "
                    "Ejecuta primero: python data/preparar_dataset.py"
                )

            df = pd.read_csv(path_clean)
            if df.empty:
                raise ValueError(
                    f"El dataset está vacío: {path_clean}. "
                    "Regenera con: python data/preparar_dataset.py"
                )

            if "timestamp" not in df.columns:
                raise ValueError(
                    f"Esquema incorrecto: {path_clean} "
                    "no incluye 'timestamp'. Regenera con preparar_dataset.py"
                )

            fechas = pd.to_datetime(df["timestamp"], errors="coerce")
            if not fechas.notna().any():
                raise ValueError(
                    f"Sin timestamps válidos: {path_clean}. "
                    "Regenera con: python data/preparar_dataset.py"
                )

            df = df.sample(frac=1, random_state=self.seed_val).reset_index(drop=True)
            datos[ag] = df
        return datos

    # ── API PettingZoo AEC ─────────────────────────────────────────────────

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents     = self.possible_agents.copy()
        self._selector  = agent_selector(self.possible_agents)
        self.agent_selection = self._selector.reset()

        self._paso_actual = 0
        self._hora_actual = int(self._rng.integers(0, 24))
        self._dia_actual  = int(self._rng.integers(0, 7))
        self._alerta_previa = {ag: 0 for ag in AGENTES}
        self._historial     = {ag: [] for ag in AGENTES}

        # Resetear índices de datos
        self._indice = {ag: 0 for ag in AGENTES}

        # Inicializar buffers PettingZoo
        self.rewards      = {ag: 0.0  for ag in self.agents}
        self._cumulative_rewards = {ag: 0.0 for ag in self.agents}
        self.terminations = {ag: False for ag in self.agents}
        self.truncations  = {ag: False for ag in self.agents}
        self.infos        = {ag: {}    for ag in self.agents}

        # Acumuladores
        self._ep_recompensas           = {ag: 0.0 for ag in AGENTES}
        self._ep_homicidios_detectados = {ag: 0   for ag in AGENTES}
        self._ep_homicidios_no_detectados = {ag: 0 for ag in AGENTES}
        self._ep_ataques_detectados    = {ag: 0   for ag in AGENTES}
        self._ep_ataques_no_detectados = {ag: 0   for ag in AGENTES}
        self._ep_ataques_alertados_modelo = {ag: 0 for ag in AGENTES}
        self._ep_ataques_no_alertados_modelo = {ag: 0 for ag in AGENTES}
        self._ep_homicidios_alertados_modelo = {ag: 0 for ag in AGENTES}
        self._ep_falsas_alarmas        = {ag: 0   for ag in AGENTES}
        self._ep_falsas_alarmas_modelo = {ag: 0 for ag in AGENTES}
        self._ep_alertas_emitidas      = {ag: 0   for ag in AGENTES}
        self._ep_alertas_efectivas     = {ag: 0   for ag in AGENTES}
        self._ep_compliance_fallos     = {ag: 0   for ag in AGENTES}
        self._ep_prioridad_despacho_sum = {ag: 0.0 for ag in AGENTES}
        self._ep_cobertura_zonal_abs_sum = {ag: 0.0 for ag in AGENTES}
        self._ep_coord_intereje_sum = {ag: 0.0 for ag in AGENTES}
        self._ep_despachos_priorizados = {ag: 0 for ag in AGENTES}
        self._ep_redistribuciones_cobertura = {ag: 0 for ag in AGENTES}
        self._ep_solicitudes_intereje = {ag: 0 for ag in AGENTES}
        self._ep_obj = {ag: np.zeros(N_OBJETIVOS) for ag in AGENTES}
        self.reward_vectors = {ag: np.zeros(N_OBJETIVOS) for ag in AGENTES}

        # Nuevos acumuladores para métricas avanzadas (HITL, Equidad y Reward Shaping)
        self._fp_consecutivos          = {ag: 0   for ag in AGENTES}
        self._ep_detecciones_por_zona  = {z: 0    for z in range(N_ZONAS)}
        self._ep_ataques_por_zona      = {z: 0    for z in range(N_ZONAS)}

        # Observación inicial
        self._obs_buffer = {ag: self._get_obs(ag) for ag in self.agents}

        return {ag: self._obs_buffer[ag] for ag in self.agents}

    def observe(self, agent: str) -> np.ndarray:
        return self._obs_buffer.get(agent, self._get_obs(agent))

    def _get_obs(self, agente: str) -> np.ndarray:
        """Construye el vector de observación local del agente."""
        df   = self._datos[agente]
        idx  = self._indice[agente] % len(df)
        fila = df.iloc[idx]

        # ── Contexto espacio-temporal (campos 0-8) ──────────────────────────
        zona_n = float(fila.get("zona_id", 0)) / float(N_ZONAS - 1)  # [0,23] → [0,1]

        hora = max(0.0, min(float(fila.get("hora", 12)), 23.0))
        hora_sin = float(np.sin(2 * np.pi * hora / 24))
        hora_cos = float(np.cos(2 * np.pi * hora / 24))

        dia = max(0.0, min(float(fila.get("dia", 0)), 6.0))
        dia_sin = float(np.sin(2 * np.pi * dia / 7))
        dia_cos = float(np.cos(2 * np.pi * dia / 7))

        hist_n   = min(float(fila.get("hist_7d", 0)) / 7.0, 1.0)
        alerta_p = float(self._alerta_previa[agente])
        riesgo   = float(fila.get("riesgo_zona", 0.5))

        # flag_visual: si el dataset trae evidencia YOLO validada se usa como
        # señal causal; si no existe, se mantiene la simulacion reproducible.
        ataque = int(fila.get("ataque_ocurrido", 0))
        if "flag_visual" in fila.index:
            try:
                flag_visual = float(fila.get("flag_visual", 0.0))
            except (TypeError, ValueError):
                flag_visual = 0.0
            flag_visual = max(0.0, min(flag_visual, 1.0))
        else:
            p_vis = 0.88 if ataque == 1 else 0.05
            flag_visual = float(self._rng.random() < p_vis)

        # ── Gravedad del evento (campos 9-11) ───────────────────────────────
        homicidio = float(fila.get("homicidio",  0))
        herido    = float(fila.get("herido",     0))
        arma_n    = float(fila.get("arma_id_n",  0.0))

        # ── Contexto de extorsión (campos 12-14) ────────────────────────────
        ext_rel   = float(fila.get("extorsion_relacionada", 0))
        monto_n   = float(fila.get("monto_cupo_n",         0.0))
        tipo_ext  = float(fila.get("tipo_ext_n",            0.0))

        # ── Modus operandi (campo 15) ────────────────────────────────────────
        vehiculo_n = float(fila.get("vehiculo_id_n", 0.0))

        # ── Perfil de la víctima (campos 16-18) ─────────────────────────────
        sexo_n  = float(fila.get("victima_sexo_n",   0.0))
        edad_n  = float(fila.get("victima_edad_n",   0.0))
        menor   = float(fila.get("victima_menor_edad", 0))

        obs = np.array(
            [zona_n, hora_sin, hora_cos, dia_sin, dia_cos,
             hist_n, alerta_p, riesgo, flag_visual,
             homicidio, herido, arma_n,
             ext_rel, monto_n, tipo_ext,
             vehiculo_n,
             sexo_n, edad_n, menor],
            dtype=np.float32,
        )
        if self.use_demand_forecast:
            demanda_pred_n = float(fila.get("demanda_pred_n", 0.0))
            demanda_pred_n = max(0.0, min(demanda_pred_n, 1.0))
            obs = np.append(obs, np.float32(demanda_pred_n)).astype(np.float32)
        if self.causal_obs:
            obs[list(NON_CAUSAL_OBS_IDX)] = 0.0
        return obs

    def _interpretar_accion(self, agente: str, accion) -> tuple[int, int, int, float, float, float]:
        """
        Interpreta acción continua Box(6) y devuelve:
        (alerta, severidad, apoyo_auxiliar, prioridad_despacho,
         cobertura_zonal, coordinacion_intereje).
        """
        arr = np.asarray(accion, dtype=np.float32).reshape(-1)
        if arr.size < ACTION_DIM:
            arr = np.pad(arr, (0, ACTION_DIM - arr.size), constant_values=0.0)
        arr = np.clip(arr[:ACTION_DIM], ACTION_LOW, ACTION_HIGH)

        alerta = int(float(arr[0]) > 0.0)
        if alerta:
            sev_norm = (float(arr[1]) + 1.0) / 2.0
            severidad = 1 + min(2, int(sev_norm * 3.0))
        else:
            severidad = 0
        extra = int(float(arr[2]) > 0.0)
        prioridad_despacho = float((arr[3] + 1.0) / 2.0)
        cobertura_zonal = float(arr[4])
        coordinacion_intereje = float(arr[5])
        if alerta == 0:
            prioridad_despacho *= 0.25
            coordinacion_intereje = min(coordinacion_intereje, 0.0)
        return (
            alerta,
            severidad,
            extra,
            prioridad_despacho,
            cobertura_zonal,
            coordinacion_intereje,
        )

    def _calcular_recompensa(
        self, agente: str, alerta: int, ataque_ocurrido: int, homicidio: int
    ) -> np.ndarray:
        """Devuelve vector R ∈ ℝ³ = [deteccion, precision, gravedad]."""
        r = RECOMPENSAS_MO
        if alerta == 1 and homicidio == 1:
            return r["homicidio_prevenido"].copy()
        elif alerta == 1 and ataque_ocurrido == 1:
            return r["ataque_prevenido"].copy()
        elif alerta == 1 and ataque_ocurrido == 0:
            return r["falsa_alarma"].copy()
        elif alerta == 0 and homicidio == 1:
            return r["homicidio_no_detectado"].copy()
        elif alerta == 0 and ataque_ocurrido == 1:
            return r["ataque_no_detectado"].copy()
        else:
            return r["silencio_ok"].copy()

    def _calcular_recompensa_decision(
        self,
        agente: str,
        alerta: int,
        alerta_efectiva: int,
        ataque_ocurrido: int,
        homicidio: int,
    ) -> np.ndarray:
        """
        Recompensa desacoplada para tesis/operación real.

        La detección y precisión miden la decisión de la IA. La gravedad mide
        el resultado operativo efectivo después del cumplimiento humano (HITL).
        Esto evita castigar al actor por fallas externas de ejecución, pero
        conserva el costo físico si una alerta correcta no se ejecuta.
        """
        reward_vec = np.zeros(N_OBJETIVOS, dtype=np.float32)

        if ataque_ocurrido == 1:
            reward_vec[0] += 5.0 if alerta == 1 else -3.0
        else:
            reward_vec[1] += -0.9 if alerta == 1 else 0.2

        if homicidio == 1:
            reward_vec[2] += 10.0 if alerta_efectiva == 1 else -10.0

        return reward_vec

    def _escalarizar(self, agente: str, reward_vec: np.ndarray) -> float:
        """Escalarización lineal ponderada: r = w_agente · R."""
        return float(np.dot(PESOS_AGENTE[agente], reward_vec))

    def step(self, action):
        agente = self.agent_selection

        if (
            self.terminations.get(agente, False)
            or self.truncations.get(agente, False)
        ):
            self._was_dead_step(action)
            return

        self._clear_rewards()
        self._cumulative_rewards[agente] = 0.0

        # Obtener dato real del episodio
        df  = self._datos[agente]
        idx = self._indice[agente] % len(df)
        fila = df.iloc[idx]
        ataque_ocurrido = int(fila.get("ataque_ocurrido", 0))
        homicidio = int(fila.get("homicidio", 0))
        zona_id = int(fila.get("zona_id", 0))

        # Interpretar acción
        (
            alerta,
            severidad,
            extra,
            prioridad_despacho,
            cobertura_zonal,
            coordinacion_intereje,
        ) = self._interpretar_accion(agente, action)
        redistribuye_cobertura = int(abs(cobertura_zonal) > 0.33)
        solicita_apoyo_intereje = int(coordinacion_intereje > 0.33)
        despacho_priorizado = int(alerta == 1 and prioridad_despacho > 0.66)

        # HITL: Factor Humano (Tasa de Cumplimiento / Compliance Rate)
        # Meyer (2004) -> Buses 85%; Bliss (2003) -> Combis/Mototaxis 60%
        _COMPLIANCE = {"buses": 0.85, "combis": 0.60, "mototaxis": 0.60}
        compliance_rate = _COMPLIANCE[agente]
        alerta_efectiva = 0
        _compliance_fallo = False
        if alerta == 1:
            self._ep_alertas_emitidas[agente] += 1
            if self._rng.random() <= compliance_rate:
                alerta_efectiva = 1
                self._ep_alertas_efectivas[agente] += 1
            else:
                _compliance_fallo = True  # humano no cumplio — fallo externo al agente
                self._ep_compliance_fallos[agente] += 1

        # Calcular recompensa base vectorial. Deteccion/precision se atribuyen
        # a la decision del modelo; gravedad se atribuye al resultado efectivo.
        reward_vec = self._calcular_recompensa_decision(
            agente, alerta, alerta_efectiva, ataque_ocurrido, homicidio
        )

        riesgo = float(fila.get("riesgo_zona", 0.5))

        # Reward Shaping: Multiplicador Target Function sigma (alta densidad criminal)
        if alerta_efectiva == 1 and ataque_ocurrido == 1:
            reward_vec *= (1.0 + riesgo)

        # Reward Shaping: Penalizacion dinamica por spam (Cry-wolf effect)
        # Tasa de crecimiento por agente: buses penaliza mas fuerte (alta compliance,
        # tiene mas control sobre sus FPs); combis/mototaxis mas suave (compliance menor,
        # menor control sobre outcome efectivo).
        _MAX_CRY_WOLF = 8
        _CRY_WOLF_RATE = {"buses": 0.75, "combis": 0.45, "mototaxis": 0.50}
        if alerta == 1 and ataque_ocurrido == 0:
            self._fp_consecutivos[agente] = min(self._fp_consecutivos[agente] + 1, _MAX_CRY_WOLF)
            penalty_multiplier = 1.0 + (_CRY_WOLF_RATE[agente] * self._fp_consecutivos[agente])
            reward_vec *= penalty_multiplier
        else:
            self._fp_consecutivos[agente] = 0

        # Reward Shaping: Recompensa Intrínseca por vigilancia activa
        # Mototaxis: zonas periféricas con datos escasos (Sparse Data)
        # Combis: corredores informales con alta heterogeneidad de rutas
        if extra == 1 and agente in ("mototaxis", "combis"):
            reward_vec[0] += 0.1 * riesgo

        # Reward Shaping operativo Box(6): despacho, cobertura y coordinacion.
        # Recompensa decisiones fisicamente interpretables cuando hay ataque,
        # y penaliza consumo de recursos si se activan sobre falsas alarmas.
        carga_operativa = (
            0.45 * prioridad_despacho
            + 0.25 * abs(cobertura_zonal)
            + 0.30 * max(coordinacion_intereje, 0.0)
        )
        if alerta_efectiva == 1 and ataque_ocurrido == 1:
            reward_vec[0] += 0.35 * prioridad_despacho * riesgo
            reward_vec[0] += 0.15 * redistribuye_cobertura * riesgo
            reward_vec[0] += 0.20 * solicita_apoyo_intereje * riesgo
            if homicidio == 1:
                reward_vec[2] += 0.75 * prioridad_despacho
        elif alerta == 1 and ataque_ocurrido == 0:
            reward_vec[1] -= 0.55 * carga_operativa
            reward_vec[1] -= 0.12 * float(severidad)
            if despacho_priorizado:
                reward_vec[1] -= 0.35
            if solicita_apoyo_intereje:
                reward_vec[1] -= 0.25
        elif alerta == 0 and carga_operativa > 0.35:
            reward_vec[1] -= 0.15 * carga_operativa

        self._ep_prioridad_despacho_sum[agente] += prioridad_despacho
        self._ep_cobertura_zonal_abs_sum[agente] += abs(cobertura_zonal)
        self._ep_coord_intereje_sum[agente] += max(coordinacion_intereje, 0.0)
        self._ep_despachos_priorizados[agente] += despacho_priorizado
        self._ep_redistribuciones_cobertura[agente] += redistribuye_cobertura
        self._ep_solicitudes_intereje[agente] += solicita_apoyo_intereje

        # Escalarizar recompensa final
        reward = self._escalarizar(agente, reward_vec)

        # Guardar vector para logging externo
        self.reward_vectors[agente] = reward_vec
        self._ep_obj[agente] += reward_vec

        # Actualizar contadores
        self._ep_recompensas[agente] += reward

        # Equidad Geográfica
        if ataque_ocurrido == 1:
            self._ep_ataques_por_zona[zona_id] += 1
            if alerta_efectiva == 1:
                self._ep_detecciones_por_zona[zona_id] += 1

        if alerta_efectiva == 1 and homicidio == 1:
            self._ep_homicidios_detectados[agente] += 1
        elif alerta_efectiva == 0 and homicidio == 1:
            self._ep_homicidios_no_detectados[agente] += 1

        if alerta_efectiva == 1 and ataque_ocurrido == 1:
            self._ep_ataques_detectados[agente] += 1
        elif alerta_efectiva == 0 and ataque_ocurrido == 1:
            self._ep_ataques_no_detectados[agente] += 1
        elif alerta_efectiva == 1 and ataque_ocurrido == 0:
            self._ep_falsas_alarmas[agente] += 1

        if alerta == 1 and ataque_ocurrido == 1:
            self._ep_ataques_alertados_modelo[agente] += 1
        elif alerta == 0 and ataque_ocurrido == 1:
            self._ep_ataques_no_alertados_modelo[agente] += 1
        elif alerta == 1 and ataque_ocurrido == 0:
            self._ep_falsas_alarmas_modelo[agente] += 1
        if alerta == 1 and homicidio == 1:
            self._ep_homicidios_alertados_modelo[agente] += 1

        # Actualizar estado
        self._alerta_previa[agente] = alerta
        self._indice[agente] += 1

        # Avanzar tiempo al final del turno del último agente
        if agente == AGENTES[-1]:
            self._paso_actual += 1
            self._hora_actual  = (self._hora_actual + 1) % 24
            if self._hora_actual == 0:
                self._dia_actual = (self._dia_actual + 1) % 7

        # Verificar terminación
        done = self._paso_actual >= self.max_steps
        self.terminations = {ag: done for ag in self.agents}
        self.truncations   = {ag: False for ag in self.agents}
        self.rewards[agente] = reward

        # Actualizar observación (siempre, incluido el paso terminal)
        self._obs_buffer[agente] = self._get_obs(agente)

        # Info
        self.infos[agente] = {
            "ataque_ocurrido":  ataque_ocurrido,
            "homicidio":        homicidio,
            "alerta_emitida":   alerta,  # Intención de la IA
            "alerta_efectiva":  alerta_efectiva, # Resultado tras el HITL
            "compliance_fallo":  _compliance_fallo,
            "severidad":        severidad,
            "accion_extra":     extra,
            "prioridad_despacho": prioridad_despacho,
            "cobertura_zonal": cobertura_zonal,
            "coordinacion_intereje": coordinacion_intereje,
            "despacho_priorizado": despacho_priorizado,
            "redistribuye_cobertura": redistribuye_cobertura,
            "solicita_apoyo_intereje": solicita_apoyo_intereje,
            "paso":             self._paso_actual,
            "hora":             self._hora_actual,
            "ep_reward":        self._ep_recompensas[agente],
            "ep_hom_detectados": self._ep_homicidios_detectados[agente],
            "ep_hom_no_detec":  self._ep_homicidios_no_detectados[agente],
            "ep_atk_detectados": self._ep_ataques_detectados[agente],
            "ep_atk_no_detec":  self._ep_ataques_no_detectados[agente],
            "ep_atk_alertados_modelo": self._ep_ataques_alertados_modelo[agente],
            "ep_atk_no_alertados_modelo": self._ep_ataques_no_alertados_modelo[agente],
            "ep_hom_alertados_modelo": self._ep_homicidios_alertados_modelo[agente],
            "ep_falsas":        self._ep_falsas_alarmas[agente],
            "ep_falsas_modelo": self._ep_falsas_alarmas_modelo[agente],
            "ep_alertas_emitidas": self._ep_alertas_emitidas[agente],
            "ep_alertas_efectivas": self._ep_alertas_efectivas[agente],
            "ep_compliance_fallos": self._ep_compliance_fallos[agente],
            # Métricas de Equidad
            "ataques_por_zona": self._ep_ataques_por_zona.copy(),
            "detecciones_por_zona": self._ep_detecciones_por_zona.copy(),
            # Vector multi-objetivo del paso actual
            "reward_vec":       reward_vec.tolist(),
            # Acumulado por objetivo en el episodio
            "ep_obj_deteccion": float(self._ep_obj[agente][0]),
            "ep_obj_precision": float(self._ep_obj[agente][1]),
            "ep_obj_gravedad":  float(self._ep_obj[agente][2]),
        }

        self._accumulate_rewards()

        if done:
            return

        # Siguiente agente
        self.agent_selection = self._selector.next()

        if self.render_mode == "human":
            self.render()

    def render(self):
        ag  = self.agent_selection
        r   = self.rewards.get(ag, 0)
        ep  = self._ep_recompensas.get(ag, 0)
        hdet = self._ep_homicidios_detectados.get(ag, 0)
        hno  = self._ep_homicidios_no_detectados.get(ag, 0)
        det = self._ep_ataques_detectados.get(ag, 0)
        nd  = self._ep_ataques_no_detectados.get(ag, 0)
        fa  = self._ep_falsas_alarmas.get(ag, 0)
        print(
            f"[Paso {self._paso_actual:04d}|{ag:10s}] "
            f"r={r:+.1f} | ep_r={ep:+6.1f} | "
            f"hom: prev={hdet} falló={hno} | "
            f"atk: det={det} nd={nd} fa={fa}"
        )

    def close(self):
        pass


    def _was_dead_step(self, action):
        """Maneja el paso cuando el agente ya terminó (compatibilidad PettingZoo)."""
        return super()._was_dead_step(action)

    # ── Estado global (para el crítico centralizado HAPPO) ────────────────

    def get_global_state(self) -> np.ndarray:
        """Concatena observaciones locales de los 3 agentes para el critico CTDE."""
        return np.concatenate([
            self._obs_buffer.get(ag, np.zeros(self.obs_dim, dtype=np.float32))
            for ag in AGENTES
        ], dtype=np.float32)

    def get_episode_stats(self) -> dict:
        por_eje = {}
        sistema = {
            "reward": 0.0,
            "homicidios_detectados": 0,
            "homicidios_no_detec": 0,
            "ataques_detectados": 0,
            "ataques_no_detectados": 0,
            "ataques_alertados_modelo": 0,
            "ataques_no_alertados_modelo": 0,
            "homicidios_alertados_modelo": 0,
            "falsas_alarmas": 0,
            "falsas_alarmas_modelo": 0,
            "alertas_emitidas": 0,
            "alertas_efectivas": 0,
            "compliance_fallos": 0,
            "despachos_priorizados": 0,
            "redistribuciones_cobertura": 0,
            "solicitudes_intereje": 0,
            "prioridad_despacho_sum": 0.0,
            "cobertura_zonal_abs_sum": 0.0,
            "coord_intereje_sum": 0.0,
            "obj_deteccion": 0.0,
            "obj_precision": 0.0,
            "obj_gravedad": 0.0,
        }

        for ag in AGENTES:
            ataques_detectados = int(self._ep_ataques_detectados[ag])
            ataques_no_detectados = int(self._ep_ataques_no_detectados[ag])
            homicidios_detectados = int(self._ep_homicidios_detectados[ag])
            homicidios_no_detec = int(self._ep_homicidios_no_detectados[ag])
            total_ataques = ataques_detectados + ataques_no_detectados
            total_homicidios = homicidios_detectados + homicidios_no_detec
            ataques_alertados_modelo = int(self._ep_ataques_alertados_modelo[ag])
            ataques_no_alertados_modelo = int(self._ep_ataques_no_alertados_modelo[ag])
            homicidios_alertados_modelo = int(self._ep_homicidios_alertados_modelo[ag])
            total_ataques_modelo = ataques_alertados_modelo + ataques_no_alertados_modelo
            falsas_alarmas = int(self._ep_falsas_alarmas[ag])
            falsas_alarmas_modelo = int(self._ep_falsas_alarmas_modelo[ag])
            alertas_emitidas = int(self._ep_alertas_emitidas[ag])
            alertas_efectivas = int(self._ep_alertas_efectivas[ag])
            compliance_fallos = int(self._ep_compliance_fallos[ag])
            n_decisiones = max(int(self._indice.get(ag, 0)), 1)
            prioridad_sum = float(self._ep_prioridad_despacho_sum[ag])
            cobertura_sum = float(self._ep_cobertura_zonal_abs_sum[ag])
            coord_sum = float(self._ep_coord_intereje_sum[ag])

            eje_stats = {
                "reward": float(self._ep_recompensas[ag]),
                "homicidios_detectados": homicidios_detectados,
                "homicidios_no_detec": homicidios_no_detec,
                "ataques_detectados": ataques_detectados,
                "ataques_no_detectados": ataques_no_detectados,
                "ataques_alertados_modelo": ataques_alertados_modelo,
                "ataques_no_alertados_modelo": ataques_no_alertados_modelo,
                "homicidios_alertados_modelo": homicidios_alertados_modelo,
                "falsas_alarmas": falsas_alarmas,
                "falsas_alarmas_modelo": falsas_alarmas_modelo,
                "alertas_emitidas": alertas_emitidas,
                "alertas_efectivas": alertas_efectivas,
                "compliance_fallos": compliance_fallos,
                "despachos_priorizados": int(self._ep_despachos_priorizados[ag]),
                "redistribuciones_cobertura": int(self._ep_redistribuciones_cobertura[ag]),
                "solicitudes_intereje": int(self._ep_solicitudes_intereje[ag]),
                "prioridad_despacho_media": prioridad_sum / n_decisiones,
                "cobertura_zonal_media_abs": cobertura_sum / n_decisiones,
                "coordinacion_intereje_media": coord_sum / n_decisiones,
                "prioridad_despacho_sum": prioridad_sum,
                "cobertura_zonal_abs_sum": cobertura_sum,
                "coord_intereje_sum": coord_sum,
                "total_ataques": total_ataques,
                "total_homicidios": total_homicidios,
                "tasa_deteccion": (
                    ataques_detectados / total_ataques if total_ataques else 0.0
                ),
                "tasa_deteccion_modelo": (
                    ataques_alertados_modelo / total_ataques_modelo
                    if total_ataques_modelo else 0.0
                ),
                "tasa_homicidios_prevenidos": (
                    homicidios_detectados / total_homicidios if total_homicidios else 0.0
                ),
                "tasa_homicidios_alertados_modelo": (
                    homicidios_alertados_modelo / total_homicidios
                    if total_homicidios else 0.0
                ),
                "fp_por_ataque": falsas_alarmas / max(total_ataques, 1),
                "fp_modelo_por_ataque": falsas_alarmas_modelo / max(total_ataques_modelo, 1),
                "tasa_alerta_efectiva": (
                    alertas_efectivas / alertas_emitidas if alertas_emitidas else 0.0
                ),
                "obj_deteccion": float(self._ep_obj[ag][0]),
                "obj_precision": float(self._ep_obj[ag][1]),
                "obj_gravedad": float(self._ep_obj[ag][2]),
            }
            por_eje[ag] = eje_stats

            for key in (
                "reward",
                "homicidios_detectados",
                "homicidios_no_detec",
                "ataques_detectados",
                "ataques_no_detectados",
                "ataques_alertados_modelo",
                "ataques_no_alertados_modelo",
                "homicidios_alertados_modelo",
                "falsas_alarmas",
                "falsas_alarmas_modelo",
                "alertas_emitidas",
                "alertas_efectivas",
                "compliance_fallos",
                "despachos_priorizados",
                "redistribuciones_cobertura",
                "solicitudes_intereje",
                "prioridad_despacho_sum",
                "cobertura_zonal_abs_sum",
                "coord_intereje_sum",
                "obj_deteccion",
                "obj_precision",
                "obj_gravedad",
            ):
                sistema[key] += eje_stats[key]

        total_ataques_sistema = (
            sistema["ataques_detectados"] + sistema["ataques_no_detectados"]
        )
        total_homicidios_sistema = (
            sistema["homicidios_detectados"] + sistema["homicidios_no_detec"]
        )
        total_ataques_modelo_sistema = (
            sistema["ataques_alertados_modelo"]
            + sistema["ataques_no_alertados_modelo"]
        )
        sistema["total_ataques"] = int(total_ataques_sistema)
        sistema["total_homicidios"] = int(total_homicidios_sistema)
        sistema["total_ataques_modelo"] = int(total_ataques_modelo_sistema)
        sistema["tasa_deteccion"] = (
            sistema["ataques_detectados"] / total_ataques_sistema
            if total_ataques_sistema
            else 0.0
        )
        sistema["tasa_deteccion_modelo"] = (
            sistema["ataques_alertados_modelo"] / total_ataques_modelo_sistema
            if total_ataques_modelo_sistema
            else 0.0
        )
        sistema["tasa_homicidios_prevenidos"] = (
            sistema["homicidios_detectados"] / total_homicidios_sistema
            if total_homicidios_sistema
            else 0.0
        )
        sistema["tasa_homicidios_alertados_modelo"] = (
            sistema["homicidios_alertados_modelo"] / total_homicidios_sistema
            if total_homicidios_sistema
            else 0.0
        )
        sistema["fp_por_ataque"] = sistema["falsas_alarmas"] / max(
            total_ataques_sistema, 1
        )
        sistema["fp_modelo_por_ataque"] = sistema["falsas_alarmas_modelo"] / max(
            total_ataques_modelo_sistema, 1
        )
        sistema["tasa_alerta_efectiva"] = (
            sistema["alertas_efectivas"] / sistema["alertas_emitidas"]
            if sistema["alertas_emitidas"]
            else 0.0
        )
        n_decisiones_sistema = max(sum(int(self._indice.get(ag, 0)) for ag in AGENTES), 1)
        sistema["prioridad_despacho_media"] = (
            sistema["prioridad_despacho_sum"] / n_decisiones_sistema
        )
        sistema["cobertura_zonal_media_abs"] = (
            sistema["cobertura_zonal_abs_sum"] / n_decisiones_sistema
        )
        sistema["coordinacion_intereje_media"] = (
            sistema["coord_intereje_sum"] / n_decisiones_sistema
        )

        return {
            "recompensas":           dict(self._ep_recompensas),
            "homicidios_detectados": dict(self._ep_homicidios_detectados),
            "homicidios_no_detec":   dict(self._ep_homicidios_no_detectados),
            "ataques_detectados":    dict(self._ep_ataques_detectados),
            "ataques_no_detectados": dict(self._ep_ataques_no_detectados),
            "ataques_alertados_modelo": dict(self._ep_ataques_alertados_modelo),
            "ataques_no_alertados_modelo": dict(self._ep_ataques_no_alertados_modelo),
            "homicidios_alertados_modelo": dict(self._ep_homicidios_alertados_modelo),
            "falsas_alarmas":        dict(self._ep_falsas_alarmas),
            "falsas_alarmas_modelo": dict(self._ep_falsas_alarmas_modelo),
            "alertas_emitidas":      dict(self._ep_alertas_emitidas),
            "alertas_efectivas":     dict(self._ep_alertas_efectivas),
            "compliance_fallos":     dict(self._ep_compliance_fallos),
            "despachos_priorizados": dict(self._ep_despachos_priorizados),
            "redistribuciones_cobertura": dict(self._ep_redistribuciones_cobertura),
            "solicitudes_intereje": dict(self._ep_solicitudes_intereje),
            "pasos":                 self._paso_actual,
            # Métricas de Equidad Geográfica
            "ataques_por_zona":      dict(self._ep_ataques_por_zona),
            "detecciones_por_zona":  dict(self._ep_detecciones_por_zona),
            # Acumulados multi-objetivo por agente
            "obj_deteccion": {ag: float(self._ep_obj[ag][0]) for ag in AGENTES},
            "obj_precision":  {ag: float(self._ep_obj[ag][1]) for ag in AGENTES},
            "obj_gravedad":   {ag: float(self._ep_obj[ag][2]) for ag in AGENTES},
            # Métricas comparables para ranking MADRL cooperativo multiobjetivo.
            "por_eje": por_eje,
            "sistema": sistema,
        }
