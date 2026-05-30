"""Logger HARL para LimaTransporteEnv.

Extiende BaseLogger para:
  - Imprimir en consola métricas extendidas por episodio
  - Escribir timeseries.csv  (fila por llamada a episode_log, métricas agregadas)
  - Escribir trace.csv       (fila por episodio completado, datos brutos)
  - Escribir result_{algo}.json al finalizar (resumen para notebooks)
  - Generar historial_entrenamiento.csv y training_history.json compatibles
    con notebooks/03_resultados_tesis.ipynb y notebooks/demostracion_hipotesis.ipynb
"""
import csv
import json
import time as _time_module
from pathlib import Path

import numpy as np

from harl.common.base_logger import BaseLogger

AGENTES = ["buses", "combis", "mototaxis"]
_OBJ_KEYS = ["ep_obj_deteccion", "ep_obj_precision", "ep_obj_gravedad"]

# Cabeceras de los archivos CSV
_TIMESERIES_HEADER = [
    "episodio", "total_steps", "fps",
    "recompensa_media",
    "recompensa_buses_media", "recompensa_combis_media", "recompensa_mototaxis_media",
    "ataques_detectados_media", "ataques_no_detectados_media",
    "falsas_alarmas_media", "tasa_deteccion_media", "hom_prevenidos_media",
    "ataques_alertados_modelo_media", "ataques_no_alertados_modelo_media",
    "falsas_alarmas_modelo_media", "tasa_deteccion_modelo_media",
    "obj_deteccion_buses",  "obj_precision_buses",  "obj_gravedad_buses",
    "obj_deteccion_combis", "obj_precision_combis", "obj_gravedad_combis",
    "obj_deteccion_mototaxis", "obj_precision_mototaxis", "obj_gravedad_mototaxis",
    "obj_deteccion_total", "obj_precision_total", "obj_gravedad_total",
    "policy_loss_buses",   "dist_entropy_buses",
    "policy_loss_combis",  "dist_entropy_combis",
    "policy_loss_mototaxis", "dist_entropy_mototaxis",
    "value_loss", "avg_step_reward",
]

_TRACE_HEADER = [
    "episodio",
    "recompensa_total", "recompensa_buses", "recompensa_combis", "recompensa_mototaxis",
    "ataques_detectados", "ataques_no_detectados", "falsas_alarmas",
    "hom_prevenidos", "hom_no_detectados",
    "tasa_deteccion",
    "ataques_alertados_modelo", "ataques_no_alertados_modelo",
    "falsas_alarmas_modelo", "tasa_deteccion_modelo",
    "obj_deteccion_buses",  "obj_precision_buses",  "obj_gravedad_buses",
    "obj_deteccion_combis", "obj_precision_combis", "obj_gravedad_combis",
    "obj_deteccion_mototaxis", "obj_precision_mototaxis", "obj_gravedad_mototaxis",
    "obj_deteccion_total", "obj_precision_total", "obj_gravedad_total",
]


class LimaTransporteLogger(BaseLogger):

    def get_task_name(self):
        return "lima_transporte"

    # ── init ─────────────────────────────────────────────────────────────────

    def init(self, episodes):
        super().init(episodes)
        n = self.algo_args["train"]["n_rollout_threads"]
        self._ep_obj  = np.zeros((n, self.num_agents, 3), dtype=np.float32)
        self.done_ep_obj = []

        # Estadísticos de episodio por thread (se leen del info terminal)
        # Forma: lista de dicts {"rew_total", "rew_per_ag", "atk_det", ...}
        self._done_ep_stats: list[dict] = []

        # Contador global de episodios completados
        self._ep_trace_count = 0

        # Calcular rutas de salida
        # self.run_dir es el directorio del run (seed-xxxxx/); logs/ está dentro
        run_dir    = Path(self.run_dir)
        tables_dir = self._find_tables_dir(run_dir)

        # CSV vivos dentro del run directory
        self._ts_path    = run_dir / "timeseries.csv"
        self._trace_path = run_dir / "trace.csv"

        # Directorio de tablas para los notebooks
        self._tables_dir = tables_dir
        self._tables_dir.mkdir(parents=True, exist_ok=True)

        # Abrir CSVs
        self._ts_f    = open(str(self._ts_path),    "w", newline="", encoding="utf-8")
        self._trace_f = open(str(self._trace_path), "w", newline="", encoding="utf-8")
        self._ts_writer    = csv.writer(self._ts_f)
        self._trace_writer = csv.writer(self._trace_f)
        self._ts_writer.writerow(_TIMESERIES_HEADER)
        self._trace_writer.writerow(_TRACE_HEADER)
        self._ts_f.flush()
        self._trace_f.flush()

        # Imprimir hiperparámetros al inicio
        t    = self.algo_args["train"]
        m    = self.algo_args.get("model", {})
        a    = self.algo_args.get("algo", {})
        algo_key = str(self.args["algo"]).lower()
        algo = algo_key.upper()
        ep_len  = t.get("episode_length", "?")
        lr      = m.get("lr", "?")
        c_lr    = m.get("critic_lr", "?")
        gamma   = a.get("gamma", "?")
        lam     = a.get("gae_lambda", "?")
        clip    = a.get("clip_param", "?")
        epochs  = a.get("ppo_epoch", "?")
        c_epochs= a.get("critic_epoch", "?")
        mbatch  = a.get("actor_num_mini_batch", "?")
        entropy = a.get("entropy_coef", "?")
        vcoef   = a.get("value_loss_coef", "?")
        hidden  = m.get("hidden_sizes", "?")
        print(f"\n{'='*60}")
        print(f"  {algo} — Hiperparámetros de entrenamiento")
        print(f"{'='*60}")
        print(f"  lr_actor={lr}  lr_critic={c_lr}")
        if algo_key == "hatrpo":
            kl = a.get("kl_threshold", "?")
            ls_step = a.get("ls_step", "?")
            accept_ratio = a.get("accept_ratio", "?")
            backtrack = a.get("backtrack_coeff", "?")
            c_mbatch = a.get("critic_num_mini_batch", "?")
            action_aggregation = a.get("action_aggregation", "?")
            std_y = m.get("std_y_coef", "?")
            print(f"  gamma={gamma}  lambda(GAE)={lam}  kl_threshold={kl}")
            print(
                f"  clip={clip}  critic_epochs={c_epochs}  "
                f"critic_mini_batches={c_mbatch}"
            )
            print(
                f"  line_search: ls_step={ls_step}  accept_ratio={accept_ratio}  "
                f"backtrack_coeff={backtrack}"
            )
            print(f"  action_aggregation={action_aggregation}  std_y_coef={std_y}")
        else:
            print(f"  gamma={gamma}  lambda(GAE)={lam}  entropy_coef={entropy}")
            print(f"  clip={clip}  ppo_epochs={epochs}  critic_epochs={c_epochs}  mini_batches={mbatch}")
        print(f"  value_loss_coef={vcoef}  hidden_sizes={hidden}  episode_length={ep_len}")
        print(f"  Agentes: {AGENTES[:self.num_agents]}")
        print(f"{'='*60}\n")
        print(f"  [CSV] timeseries -> {self._ts_path}")
        print(f"  [CSV] trace      -> {self._trace_path}")

    @staticmethod
    def _find_tables_dir(run_dir: Path) -> Path:
        """Sube por los padres de run_dir hasta encontrar results/, devuelve results/tables/."""
        for p in [run_dir, *run_dir.parents]:
            if p.name == "results":
                return p / "tables"
        # fallback: 5 niveles sobre run_dir -> results/harl/../../../tables
        return run_dir.parents[4] / "tables"

    # ── per_step ─────────────────────────────────────────────────────────────

    def per_step(self, data):
        """Acumula rewards y extrae estadísticos completos del info terminal."""
        (obs, share_obs, rewards, dones, infos,
         available_actions, values, actions,
         action_log_probs, rnn_states, rnn_states_critic) = data

        dones_env  = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0

                # Leer estadísticos acumulados del info terminal (ep_* keys)
                ep_obj_snap  = np.zeros((self.num_agents, 3), dtype=np.float32)
                rew_per_ag   = np.zeros(self.num_agents, dtype=np.float32)
                atk_det_tot  = atk_nodet_tot = fp_tot = hom_prev_tot = hom_nodet_tot = 0
                atk_model_tot = atk_nomodel_tot = fp_model_tot = 0

                for ag_idx in range(self.num_agents):
                    try:
                        info = infos[t][ag_idx]
                        ep_obj_snap[ag_idx, 0] = float(info.get("ep_obj_deteccion", 0))
                        ep_obj_snap[ag_idx, 1] = float(info.get("ep_obj_precision", 0))
                        ep_obj_snap[ag_idx, 2] = float(info.get("ep_obj_gravedad",  0))
                        rew_per_ag[ag_idx]  = float(info.get("ep_reward", 0))
                        atk_det_tot  += int(info.get("ep_atk_detectados", 0))
                        atk_nodet_tot+= int(info.get("ep_atk_no_detec",   0))
                        atk_model_tot += int(info.get("ep_atk_alertados_modelo", 0))
                        atk_nomodel_tot += int(info.get("ep_atk_no_alertados_modelo", 0))
                        fp_tot       += int(info.get("ep_falsas",          0))
                        fp_model_tot += int(info.get("ep_falsas_modelo",   0))
                        hom_prev_tot += int(info.get("ep_hom_detectados",  0))
                        hom_nodet_tot+= int(info.get("ep_hom_no_detec",    0))
                    except (IndexError, TypeError, AttributeError):
                        pass

                self.done_ep_obj.append(ep_obj_snap)
                self._ep_obj[t] = 0

                total_atk = atk_det_tot + atk_nodet_tot
                tasa = atk_det_tot / total_atk if total_atk > 0 else 0.0
                total_atk_modelo = atk_model_tot + atk_nomodel_tot
                tasa_modelo = (
                    atk_model_tot / total_atk_modelo if total_atk_modelo > 0 else 0.0
                )

                self._done_ep_stats.append({
                    "rew_total":   float(self.done_episodes_rewards[-1]),
                    "rew_per_ag":  rew_per_ag.copy(),
                    "atk_det":     atk_det_tot,
                    "atk_nodet":   atk_nodet_tot,
                    "fp":          fp_tot,
                    "hom_prev":    hom_prev_tot,
                    "hom_nodet":   hom_nodet_tot,
                    "tasa":        tasa,
                    "atk_model":   atk_model_tot,
                    "atk_nomodel": atk_nomodel_tot,
                    "fp_model":    fp_model_tot,
                    "tasa_modelo": tasa_modelo,
                    "ep_obj":      ep_obj_snap.copy(),
                })

    # ── episode_log ──────────────────────────────────────────────────────────

    def episode_log(self, actor_train_infos, critic_train_info,
                    _actor_buffer, critic_buffer):
        """Imprime métricas extendidas y escribe CSV."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = _time_module.time()
        fps = int(self.total_num_steps / (self.end - self.start))

        algo = self.args["algo"].upper()
        print(
            f"Env {self.args['env']} | {algo} | "
            f"ep {self.episode}/{self.episodes} | "
            f"steps {self.total_num_steps:,}/{self.algo_args['train']['num_env_steps']:,} | "
            f"FPS {fps}"
        )

        # ── Actor ────────────────────────────────────────────────────────────
        actor_losses: list[dict] = []
        for ag_idx, ag_name in enumerate(AGENTES[:self.num_agents]):
            info   = actor_train_infos[ag_idx] if ag_idx < len(actor_train_infos) else {}
            loss   = info.get("policy_loss",    float("nan"))
            ent    = info.get("dist_entropy",   float("nan"))
            ratio  = info.get("ratio",          float("nan"))
            gnorm  = info.get("actor_grad_norm",float("nan"))
            print(
                f"  [{ag_name:8s}]  loss={loss:.4f}  entropy={ent:.4f}"
                f"  ratio={ratio:.4f}  grad={gnorm:.4f}"
            )
            for k, v in info.items():
                self.writter.add_scalars(
                    f"agent{ag_idx}/{k}", {f"agent{ag_idx}/{k}": v},
                    self.total_num_steps,
                )
            actor_losses.append({"loss": loss, "ent": ent})

        # ── Crítico ───────────────────────────────────────────────────────────
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        v_loss    = critic_train_info.get("value_loss",       float("nan"))
        c_gnorm   = critic_train_info.get("critic_grad_norm", float("nan"))
        step_rew  = critic_train_info["average_step_rewards"]
        print(f"  [crítico  ]  value_loss={v_loss:.4f}  grad={c_gnorm:.4f}"
              f"  avg_step_reward={step_rew:.4f}")
        for k, v in critic_train_info.items():
            self.writter.add_scalars(f"critic/{k}", {f"critic/{k}": v},
                                     self.total_num_steps)

        # ── Recompensa de episodio ────────────────────────────────────────────
        if self.done_episodes_rewards:
            aver_rew = np.mean(self.done_episodes_rewards)
            print(f"  ep_reward_avg={aver_rew:.2f}  (n={len(self.done_episodes_rewards)} eps)")
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_rew},
                self.total_num_steps,
            )

        # ── Multi-objetivos por agente ────────────────────────────────────────
        if self.done_ep_obj:
            obj_mean = np.mean(self.done_ep_obj, axis=0)  # (num_agents, 3)
            print("  Multi-objetivos (media episodios):")
            for ag_idx, ag_name in enumerate(AGENTES[:self.num_agents]):
                d, p, g = obj_mean[ag_idx]
                print(f"    [{ag_name:8s}]  detec={d:.3f}  prec={p:.3f}  grav={g:.3f}")
                for j, oname in enumerate(["obj_deteccion", "obj_precision", "obj_gravedad"]):
                    self.writter.add_scalars(
                        f"agent{ag_idx}/{oname}",
                        {f"agent{ag_idx}/{oname}": float(obj_mean[ag_idx, j])},
                        self.total_num_steps,
                    )
        else:
            obj_mean = np.zeros((self.num_agents, 3))

        # ── Escribir trace.csv (una fila por episodio completado) ─────────────
        for stat in self._done_ep_stats:
            self._ep_trace_count += 1
            obj = stat["ep_obj"]
            self._trace_writer.writerow([
                self._ep_trace_count,
                round(stat["rew_total"],   4),
                round(float(stat["rew_per_ag"][0]), 4),
                round(float(stat["rew_per_ag"][1]) if self.num_agents > 1 else 0.0, 4),
                round(float(stat["rew_per_ag"][2]) if self.num_agents > 2 else 0.0, 4),
                stat["atk_det"], stat["atk_nodet"], stat["fp"],
                stat["hom_prev"], stat["hom_nodet"],
                round(stat["tasa"], 4),
                stat["atk_model"], stat["atk_nomodel"], stat["fp_model"],
                round(stat["tasa_modelo"], 4),
                round(float(obj[0, 0]), 3), round(float(obj[0, 1]), 3), round(float(obj[0, 2]), 3),
                round(float(obj[1, 0]) if self.num_agents > 1 else 0.0, 3),
                round(float(obj[1, 1]) if self.num_agents > 1 else 0.0, 3),
                round(float(obj[1, 2]) if self.num_agents > 1 else 0.0, 3),
                round(float(obj[2, 0]) if self.num_agents > 2 else 0.0, 3),
                round(float(obj[2, 1]) if self.num_agents > 2 else 0.0, 3),
                round(float(obj[2, 2]) if self.num_agents > 2 else 0.0, 3),
                round(float(obj[:, 0].sum()), 3),
                round(float(obj[:, 1].sum()), 3),
                round(float(obj[:, 2].sum()), 3),
            ])
        self._trace_f.flush()

        # ── Escribir timeseries.csv (una fila por episode_log) ────────────────
        if self._done_ep_stats:
            rew_arr    = np.array([s["rew_total"] for s in self._done_ep_stats])
            ag_rew_arr = np.array([s["rew_per_ag"] for s in self._done_ep_stats])
            ev_arr     = np.array([[s["atk_det"], s["atk_nodet"], s["fp"],
                                    s["hom_prev"], s["atk_model"], s["atk_nomodel"],
                                    s["fp_model"]] for s in self._done_ep_stats])
            tasa_arr   = np.array([s["tasa"] for s in self._done_ep_stats])
            tasa_modelo_arr = np.array([s["tasa_modelo"] for s in self._done_ep_stats])

            ev_mean = ev_arr.mean(axis=0)
            ag_rew_mean = ag_rew_arr.mean(axis=0)

            def _al(info_list: list[dict], key: str, i: int) -> float:
                return float(info_list[i].get(key, float("nan"))) if i < len(info_list) else float("nan")

            ts_row = [
                self.episode, self.total_num_steps, fps,
                round(float(rew_arr.mean()), 4),
                round(float(ag_rew_mean[0]), 4),
                round(float(ag_rew_mean[1]) if self.num_agents > 1 else 0.0, 4),
                round(float(ag_rew_mean[2]) if self.num_agents > 2 else 0.0, 4),
                round(float(ev_mean[0]), 2), round(float(ev_mean[1]), 2),
                round(float(ev_mean[2]), 2), round(float(tasa_arr.mean()), 4),
                round(float(ev_mean[3]), 2),
                round(float(ev_mean[4]), 2), round(float(ev_mean[5]), 2),
                round(float(ev_mean[6]), 2), round(float(tasa_modelo_arr.mean()), 4),
                round(float(obj_mean[0, 0]), 3), round(float(obj_mean[0, 1]), 3),
                round(float(obj_mean[0, 2]), 3),
                round(float(obj_mean[1, 0]) if self.num_agents > 1 else 0.0, 3),
                round(float(obj_mean[1, 1]) if self.num_agents > 1 else 0.0, 3),
                round(float(obj_mean[1, 2]) if self.num_agents > 1 else 0.0, 3),
                round(float(obj_mean[2, 0]) if self.num_agents > 2 else 0.0, 3),
                round(float(obj_mean[2, 1]) if self.num_agents > 2 else 0.0, 3),
                round(float(obj_mean[2, 2]) if self.num_agents > 2 else 0.0, 3),
                round(float(obj_mean[:, 0].sum()), 3),
                round(float(obj_mean[:, 1].sum()), 3),
                round(float(obj_mean[:, 2].sum()), 3),
                round(_al(actor_train_infos, "policy_loss",  0), 6),
                round(_al(actor_train_infos, "dist_entropy", 0), 6),
                round(_al(actor_train_infos, "policy_loss",  1), 6),
                round(_al(actor_train_infos, "dist_entropy", 1), 6),
                round(_al(actor_train_infos, "policy_loss",  2), 6),
                round(_al(actor_train_infos, "dist_entropy", 2), 6),
                round(float(v_loss),   6),
                round(float(step_rew), 6),
            ]
            self._ts_writer.writerow(ts_row)
            self._ts_f.flush()

        # ── Limpiar listas ────────────────────────────────────────────────────
        self.done_episodes_rewards = []
        self.done_ep_obj = []
        self._done_ep_stats = []

        print()

    # ── finalize_training ─────────────────────────────────────────────────────

    def finalize_training(self) -> None:
        """Cierra CSVs y genera los archivos para los notebooks."""
        try:
            self._ts_f.close()
            self._trace_f.close()
        except Exception:
            pass

        algo = self.args["algo"].upper()
        print(f"\n[Logger] Generando archivos de resultados para {algo}…")

        trace_path = self._trace_path
        tables_dir = self._tables_dir
        tables_dir.mkdir(parents=True, exist_ok=True)

        if not trace_path.exists() or trace_path.stat().st_size < 50:
            print(f"[Logger] trace.csv vacío — omitiendo exportación.")
            return

        import pandas as pd  # importación local para no forzar dependencia en HARL

        df_trace = pd.read_csv(str(trace_path))
        n_ep = len(df_trace)
        if n_ep == 0:
            return

        algo_key = self.args["algo"].lower()

        # ── 1. result_{algo}.json ─────────────────────────────────────────────
        last50 = df_trace.tail(50)
        media_r  = float(df_trace["recompensa_total"].mean())
        std_r    = float(df_trace["recompensa_total"].std(ddof=1)) if n_ep > 1 else 0.0
        media_r50= float(last50["recompensa_total"].mean())
        std_r50  = float(last50["recompensa_total"].std(ddof=1)) if len(last50) > 1 else 0.0
        det_final= float(last50["tasa_deteccion"].mean())
        det_modelo_final = (
            float(last50["tasa_deteccion_modelo"].mean())
            if "tasa_deteccion_modelo" in last50.columns
            else det_final
        )
        fp_modelo_final = (
            float(last50["falsas_alarmas_modelo"].mean())
            if "falsas_alarmas_modelo" in last50.columns
            else float(last50["falsas_alarmas"].mean())
        )

        result = {
            "algoritmo": algo_key,
            "n_episodios": n_ep,
            "metricas_finales": {
                "n_episodios":         n_ep,
                "media_r_total":       round(media_r,   4),
                "std_r_total":         round(std_r,     4),
                "media_r_ultimos50":   round(media_r50, 4),
                "std_r_ultimos50":     round(std_r50,   4),
                "tasa_deteccion_final":round(det_final, 4),
                "tasa_deteccion_modelo_final": round(det_modelo_final, 4),
                "falsas_alarmas_modelo_media_ultimos50": round(fp_modelo_final, 3),
                "obj_deteccion_total": round(float(df_trace["obj_deteccion_total"].mean()), 3),
                "obj_precision_total": round(float(df_trace["obj_precision_total"].mean()), 3),
                "obj_gravedad_total":  round(float(df_trace["obj_gravedad_total"].mean()),  3),
            },
            "distribucion": {
                "min_r":   round(float(df_trace["recompensa_total"].min()), 4),
                "max_r":   round(float(df_trace["recompensa_total"].max()), 4),
                "p25_r":   round(float(df_trace["recompensa_total"].quantile(0.25)), 4),
                "p50_r":   round(float(df_trace["recompensa_total"].quantile(0.50)), 4),
                "p75_r":   round(float(df_trace["recompensa_total"].quantile(0.75)), 4),
            },
        }
        result_path = tables_dir / f"result_{algo_key}.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  -> result_{algo_key}.json")

        # ── 2. training_history.json (para demostracion_hipotesis.ipynb) ─────
        hist_records = []
        for _, row in df_trace.iterrows():
            hist_records.append({
                "episode":        int(row["episodio"]),
                "reward_total":   round(float(row["recompensa_total"]), 4),
                "reward_buses":   round(float(row["recompensa_buses"]),   4),
                "reward_combis":  round(float(row["recompensa_combis"]),  4),
                "reward_mototaxis": round(float(row["recompensa_mototaxis"]), 4),
                "tasa_deteccion": round(float(row["tasa_deteccion"]), 4),
                "tasa_deteccion_modelo": round(
                    float(row.get("tasa_deteccion_modelo", row["tasa_deteccion"])), 4
                ),
            })
        hist_path = tables_dir / "training_history.json"
        hist_path.write_text(
            json.dumps(hist_records, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  -> training_history.json ({n_ep} episodios)")

        # ── 3. historial_entrenamiento.csv (para 03_resultados_tesis.ipynb) ───
        df_hist = pd.DataFrame({
            "episodio":                  df_trace["episodio"],
            "recompensa_total":          df_trace["recompensa_total"],
            "recompensa_buses":          df_trace["recompensa_buses"],
            "recompensa_combis":         df_trace["recompensa_combis"],
            "recompensa_mototaxis":      df_trace["recompensa_mototaxis"],
            "ataques_detectados_total":  df_trace["ataques_detectados"],
            "ataques_no_detectados_total": df_trace["ataques_no_detectados"],
            "falsas_alarmas_total":      df_trace["falsas_alarmas"],
            "tasa_deteccion":            df_trace["tasa_deteccion"],
            "ataques_alertados_modelo_total": df_trace.get(
                "ataques_alertados_modelo", df_trace["ataques_detectados"]
            ),
            "ataques_no_alertados_modelo_total": df_trace.get(
                "ataques_no_alertados_modelo", df_trace["ataques_no_detectados"]
            ),
            "falsas_alarmas_modelo_total": df_trace.get(
                "falsas_alarmas_modelo", df_trace["falsas_alarmas"]
            ),
            "tasa_deteccion_modelo": df_trace.get(
                "tasa_deteccion_modelo", df_trace["tasa_deteccion"]
            ),
            "loss_critico":              0.0,    # se rellena desde timeseries si disponible
            "tiempo_ep_seg":             0.0,
            "obj_deteccion_total":       df_trace["obj_deteccion_total"],
            "obj_precision_total":       df_trace["obj_precision_total"],
            "obj_gravedad_total":        df_trace["obj_gravedad_total"],
            "obj_deteccion_buses":       df_trace["obj_deteccion_buses"],
            "obj_precision_buses":       df_trace["obj_precision_buses"],
            "obj_gravedad_buses":        df_trace["obj_gravedad_buses"],
            "obj_deteccion_combis":      df_trace["obj_deteccion_combis"],
            "obj_precision_combis":      df_trace["obj_precision_combis"],
            "obj_gravedad_combis":       df_trace["obj_gravedad_combis"],
            "obj_deteccion_motos":       df_trace["obj_deteccion_mototaxis"],
            "obj_precision_motos":       df_trace["obj_precision_mototaxis"],
            "obj_gravedad_motos":        df_trace["obj_gravedad_mototaxis"],
            "n_updates_ep":              2,
            "pasos_ep":                  int(self.algo_args["train"].get("episode_length", 672)),
        })

        # Rellenar loss_critico desde timeseries.csv si existe
        ts_path = self._ts_path
        if ts_path.exists() and ts_path.stat().st_size > 50:
            df_ts = pd.read_csv(str(ts_path))
            if "value_loss" in df_ts.columns and len(df_ts) >= 2:
                from scipy.interpolate import interp1d  # type: ignore[import]
                try:
                    f_loss = interp1d(
                        df_ts["episodio"], df_ts["value_loss"],
                        kind="linear", fill_value="extrapolate",
                    )
                    df_hist["loss_critico"] = f_loss(df_hist["episodio"]).round(6)
                except Exception:
                    df_hist["loss_critico"] = float(df_ts["value_loss"].mean())
            elif "value_loss" in df_ts.columns and len(df_ts) == 1:
                df_hist["loss_critico"] = float(df_ts["value_loss"].iloc[0])

        df_hist["media_movil_20"] = (
            df_hist["recompensa_total"].rolling(20, min_periods=1).mean().round(4)
        )

        hist_csv_path = tables_dir / "historial_entrenamiento.csv"
        df_hist.to_csv(str(hist_csv_path), index=False, encoding="utf-8-sig")
        print(f"  -> historial_entrenamiento.csv ({n_ep} episodios, {len(df_hist.columns)} cols)")

        # ── 4. Copiar timeseries.csv al tables_dir ────────────────────────────
        import shutil
        ts_dest = tables_dir / f"timeseries_{algo_key}.csv"
        try:
            shutil.copy2(str(ts_path), str(ts_dest))
            print(f"  -> timeseries_{algo_key}.csv")
        except Exception:
            pass

        print(f"[Logger] Archivos en: {tables_dir}\n")

    # ── off-policy hook (estático) ────────────────────────────────────────────

    @staticmethod
    def log_off_policy_episode(infos_t, num_agents, writter, total_steps):
        """Loguea multi-objetivos al fin de episodio para algoritmos off-policy."""
        if not len(infos_t) or not isinstance(infos_t[0], dict):
            return
        if "ep_obj_deteccion" not in infos_t[0]:
            return

        agentes_nombres = AGENTES[:num_agents]
        print("  Multi-objetivos (episodio off-policy):")
        for ag_idx, ag_name in enumerate(agentes_nombres):
            try:
                info = infos_t[ag_idx]
                d = float(info.get("ep_obj_deteccion", 0.0))
                p = float(info.get("ep_obj_precision", 0.0))
                g = float(info.get("ep_obj_gravedad",  0.0))
                print(f"    [{ag_name:8s}]  detec={d:.3f}  prec={p:.3f}  grav={g:.3f}")
                for oname, val in [("obj_deteccion", d), ("obj_precision", p),
                                    ("obj_gravedad", g)]:
                    writter.add_scalars(
                        f"agent{ag_idx}/{oname}",
                        {f"agent{ag_idx}/{oname}": val},
                        total_steps,
                    )
            except (IndexError, TypeError, AttributeError):
                pass
