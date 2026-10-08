"""
netra_gym_env.py -- gymnasium environment for adaptive NETRA edge decisions
===========================================================================

Built on data/netra_simulator.py (paper-calibrated events, vision confusion, power, LoRa).
The agent replaces the fixed rules of arXiv:2605.08246v1 (Algorithms 1 + 2, fixed SF) and
jointly decides camera use, alert transmission, LoRa spreading factor, gateway and a new
sleep / duty-cycle action, as proposed in PhD.pdf (Table 2).

Time model (event-driven semi-MDP)
----------------------------------
Decision epochs are
  * EVENT epochs    - a PIR trigger while the node is awake (ultrasonic reading is taken
                      automatically, as in Algorithm 1),
  * HEARTBEAT epochs - every `heartbeat_min` minutes without events, and on every wake-up.
An EVENT has up to two sub-steps:
  1. "sense" stage: the agent sees PIR + distance features. camera=on -> the camera and the
     vision model run (energy charged), the tx/sleep parts of this action are ignored and the
     SAME event is presented again with the vision confidence + label ("classified" stage).
     camera=off -> the tx/sleep parts are applied immediately (alert on sensor evidence only).
  2. "classified" stage: tx (wait or SF7..SF12) + gateway + sleep are applied.
On HEARTBEAT epochs only the sleep part matters (camera/tx are ignored; no patrol images).
Sleeping for S minutes power-gates the Pi (sleep_power_w instead of idle power); every
event during the sleep window + boot time is missed (intrusions -> missed penalty).

Observation  (Box[-1, 1], float32)                         index
    pir                      0/1                                  0
    p_dist                   min(dd/1.5, 1), 0 if reading invalid 1
    dist_valid               1 if 4<=D<=15 m and dd>0             2
    delta_d                  clip(dd / 5 m, -1, 1)                3
    vision_conf              0 unless camera used on this event   4
    camera_used              0/1                                  5
    label one-hot            human, elephant, cow, obstacle       6-9  (0 unless camera used)
    battery_soc              0..1                                 10
    rssi[g]                  clip((RSSI + 140) / 60, 0, 1)        11 .. 10+n_gateways
    [time_features] sin(2pi h/24), cos(2pi h/24), solar_frac      next 3
  Deviation from PhD.pdf's s in R^5 = [PIR, distance, YOLO conf, battery, RSSI]: all five
  are present; added: dist_valid + raw delta_d (Algorithm 1 is range-aware), camera_used +
  label one-hot (conf is meaningless without the label and only exists after the camera
  ran), optional time/solar features (battery planning), one RSSI per gateway.

Action  (Discrete, flattened)   a = ((gw * n_sleep + sleep) * n_tx + tx) * 2 + camera
    camera   {off, on}
    tx       {wait, SF7, SF8, SF9, SF10, SF11, SF12}  ("wait" = do not send; no deferred queue)
    sleep    sleep_options_min, default (0, 15, 60) minutes
    gateway  n_gateways, default 1 (the paper has one receiver: ESP32 on the train / Pi Pico)
  Default size 2 x 7 x 3 x 1 = 42 actions.

Reward (per step; weights configurable)                     PhD.pdf Sec. 3 + additions
    +10   intrusion with a delivered alert             (r_correct)
    -5    delivered alert for a non-intrusion          (r_false_alarm)
    -8    intrusion without a delivered alert, incl. events missed while asleep and all
          remaining intrusions after a brownout        (r_missed)
     0    non-intrusion correctly not alerted          (r_correct_reject)
    -w_energy * E_action / energy_ref_j                (w_energy=2, energy_ref_j=20 J ~ one
          Pi Zero camera+inference cycle). E_action = camera + inference + LoRa TX/RX
          (+ boot energy if energy_include_boot). energy_tx_only=True restricts it to TX/RX
          energy (literal PhD "-2*E_transmit"). Boot energy always drains the battery; it is
          excluded from the reward by default because idle savings from sleeping are not
          rewarded either (w_idle=0) -- the value of sleep comes via battery (SoC shaping,
          brownout). Set w_idle>0 (per idle_energy_ref_j = 1 Wh) together with
          energy_include_boot=True for an explicit energy-accounting variant.
    -w_soc_low * max(0, soc_low - soc)/soc_low per epoch (battery-low shaping, default 1, 0.2)
    -50   brownout (battery empty -> episode terminates)  (r_brownout)
  Cows: cow_is_intrusion=True treats cattle as intrusions (Table X counts them as dangerous
  alerts); False makes them neutral (0 reward either way; eq. 7 "logged only").
  reward_mode:
    "standard"      sum of the terms above.
    "lagrangian"    standard - lambda * n_missed. lambda is a dual variable updated at every
                    reset: lambda <- clip(lambda + lagrange_lr*(miss_rate - miss_rate_target),
                    0, lagrange_max). Primal-dual CMDP heuristic, so the reward is intentionally
                    non-stationary. Current lambda is in info["lagrange_lambda"].
    "lexicographic" lexi_primary_scale * (miss + brownout terms) + lexi_secondary_scale * (all
                    other terms): scale separation that approximates "minimise misses first,
                    then false alarms / energy".
  info["cost"] = number of missed intrusions in the step (for external constrained-RL code).

Energy model (see data/netra_simulator.py for provenance)
    camera 10 J, inference (P_inf - P_idle) * t_inf, LoRa Semtech airtime x 120 mA x 3.3 V
    (+ ACK RX), idle power continuously, solar from the simulator's 1-minute grid.
    Sleep power 0.05 W (Pi Zero) / 0.10 W (Pi 4) and boot 25 s @ 1.0 W / 20 s @ 4.0 W assume
    an external power-gating controller (RTC/MCU)              [SYNTHETIC -- not in paper]
RSSI: per epoch and gateway, log-distance model of the simulator (n = 3.5, shadowing 4 dB +
    weather), link distance per gateway ~ U(link_distance_m) per episode, default 1.5-3 km so
    SF choice matters (at 1 km every SF is ~100%).                 [SYNTHETIC]
Events per episode: Poisson per-day rates. Default "mixed_113" = 2.86 intrusions/day
    (human 1.56, elephant 0.30, cow 0.56, obstacle 0.45) + 13.3 false PIR triggers/day, i.e.
    the paper's 113-event ratio with the Table X class mix. Simulator scenarios can be used
    too (event_profile="field_97" -> 11 intrusions + 2.86 false/day; with that intrusion-heavy
    mix "alert on every PIR trigger, never use the camera" is near-optimal, so it is a poor
    training default).

Usage
    import sys; sys.path.insert(0, "env")
    from netra_gym_env import NetraEnv, EnvConfig
    env = NetraEnv(EnvConfig(platform="pi_zero", episode_days=1))
    obs, info = env.reset(seed=0)
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "data"))
from netra_simulator import (  # noqa: E402
    CLASSES, PAPER, PLATFORMS, SCENARIOS, WEATHER_EXTRA_SHADOW_DB, WEATHER_LOSS_DB,
    WEATHER_SOLAR, SimConfig, lora_airtime_s, lora_delivery_prob, lora_rx_energy_j,
    lora_tx_energy_j, path_loss_db, simulate, solar_power_w,
)

SF_OPTIONS = (7, 8, 9, 10, 11, 12)

# Env-level event profiles (per-day Poisson rates). Simulator scenarios are also accepted
# (their counts / 7 days). "mixed_113" [SYNTHETIC combination]: the paper's 113-event
# intrusion/false-trigger ratio (20 intrusions + 93 false triggers per week) with the
# Table X class mix of intrusions (42 human : 8 elephant : 15 cow : 12 obstacle).
ENV_PROFILES = {
    "mixed_113": dict(intrusions={"human": 20 / 7 * 42 / 77, "elephant": 20 / 7 * 8 / 77,
                                  "cow": 20 / 7 * 15 / 77, "obstacle": 20 / 7 * 12 / 77},
                      false_per_day=93 / 7),
}


@dataclass
class EnvConfig:
    platform: str = "pi_zero"
    event_profile: str = "mixed_113"           # ENV_PROFILES key or simulator scenario (counts / 7 days)
    rate_scale: float = 1.0
    episode_days: float = 1.0                   # 1 = day, 7 = week
    link_distance_m: Tuple[float, float] = (1500.0, 3000.0)
    n_gateways: int = 1
    heartbeat_min: float = 30.0
    sleep_options_min: Tuple[float, ...] = (0.0, 15.0, 60.0)
    tx_power_dbm: float = 20.0
    max_attempts: int = 3
    sigma_fast_db: float = 3.0
    battery_init_range: Tuple[float, float] = (0.5, 1.0)
    sleep_power_w: Optional[float] = None       # default per platform
    boot_s: Optional[float] = None
    boot_power_w: Optional[float] = None
    time_features: bool = True
    # reward
    r_correct: float = 10.0
    r_false_alarm: float = -5.0
    r_missed: float = -8.0
    r_correct_reject: float = 0.0
    w_energy: float = 2.0
    energy_ref_j: float = 20.0
    energy_tx_only: bool = False
    energy_include_boot: bool = False           # boot energy always drains the battery; reward only if True
    w_idle: float = 0.0                         # optional penalty on idle/sleep consumption
    idle_energy_ref_j: float = 3600.0           # 1 Wh
    w_soc_low: float = 1.0
    soc_low: float = 0.2
    r_brownout: float = -50.0
    cow_is_intrusion: bool = True
    reward_mode: str = "standard"               # standard | lagrangian | lexicographic
    miss_rate_target: float = 0.05
    lagrange_lr: float = 5.0
    lagrange_init: float = 0.0
    lagrange_max: float = 100.0
    lexi_primary_scale: float = 10.0
    lexi_secondary_scale: float = 0.1
    sim_overrides: Dict = field(default_factory=dict)


PLATFORM_SLEEP = {"pi_zero": dict(sleep_power_w=0.05, boot_s=25.0, boot_power_w=1.0),
                  "pi4": dict(sleep_power_w=0.10, boot_s=20.0, boot_power_w=4.0)}


class NetraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg = replace(config) if config is not None else EnvConfig()
        if cfg.platform not in PLATFORMS:
            raise ValueError(cfg.platform)
        if cfg.reward_mode not in ("standard", "lagrangian", "lexicographic"):
            raise ValueError(cfg.reward_mode)
        ps = PLATFORM_SLEEP[cfg.platform]
        self.sleep_w = cfg.sleep_power_w if cfg.sleep_power_w is not None else ps["sleep_power_w"]
        self.boot_s = cfg.boot_s if cfg.boot_s is not None else ps["boot_s"]
        self.boot_w = cfg.boot_power_w if cfg.boot_power_w is not None else ps["boot_power_w"]
        self.plat = PLATFORMS[cfg.platform]
        self.n_tx = 1 + len(SF_OPTIONS)
        self.n_sleep = len(cfg.sleep_options_min)
        self.action_space = spaces.Discrete(2 * self.n_tx * self.n_sleep * cfg.n_gateways)
        self.obs_dim = 11 + cfg.n_gateways + (3 if cfg.time_features else 0)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        if cfg.event_profile in ENV_PROFILES:
            pr = ENV_PROFILES[cfg.event_profile]
            self.rates = {c: r * cfg.rate_scale for c, r in pr["intrusions"].items()}
            self.false_rate = pr["false_per_day"] * cfg.rate_scale
            self.sim_scenario = "field_97"
        else:
            sc = SCENARIOS[cfg.event_profile]
            self.rates = {c: n / 7.0 * cfg.rate_scale for c, n in sc["intrusions"].items()}
            self.false_rate = sc["n_false"] / 7.0 * cfg.rate_scale
            self.sim_scenario = cfg.event_profile
        self.camera_j = PAPER["camera_w"] * PAPER["camera_on_s"]
        self.infer_j = (self.plat["infer_w"] - self.plat["idle_w"]) * self.plat["infer_s"]
        self.lagrange_lambda = cfg.lagrange_init
        self._last_ep_stats = None
        self._sim = None

    # ------------------------------------------------------------------ actions
    def encode_action(self, camera: int, tx: int, sleep: int = 0, gateway: int = 0) -> int:
        return ((gateway * self.n_sleep + sleep) * self.n_tx + tx) * 2 + camera

    def decode_action(self, a: int) -> Tuple[int, int, int, int]:
        a = int(a)
        camera = a % 2; a //= 2
        tx = a % self.n_tx; a //= self.n_tx
        sleep = a % self.n_sleep; a //= self.n_sleep
        return camera, tx, sleep, a

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        cfg = self.cfg
        if cfg.reward_mode == "lagrangian" and self._last_ep_stats is not None:
            s = self._last_ep_stats
            mr = s["missed"] / s["intrusions"] if s["intrusions"] else 0.0
            self.lagrange_lambda = float(np.clip(self.lagrange_lambda + cfg.lagrange_lr * (mr - cfg.miss_rate_target),
                                                 0.0, cfg.lagrange_max))
        rng = self.np_random
        days = cfg.episode_days
        intr = {c: int(rng.poisson(r * days)) for c, r in self.rates.items()}
        n_false = int(rng.poisson(self.false_rate * days))
        init = float(rng.uniform(*cfg.battery_init_range))
        sim_cfg = SimConfig(scenario=self.sim_scenario, platform=cfg.platform,
                            seed=int(rng.integers(0, 2**31 - 1)), intrusions=intr, n_false=n_false,
                            deployment_days=days, link_distance_m=cfg.link_distance_m,
                            tx_power_dbm=cfg.tx_power_dbm, battery_init_frac=init, **cfg.sim_overrides)
        sim = simulate(sim_cfg)
        self._sim = sim
        self.simcfg = sim.config
        self.dt = sim.config.grid_step_s
        self.horizon_s = days * 86400.0
        self.solar_wh = sim.grid_net_wh + self.plat["idle_w"] * self.dt / 3600.0
        self.cap = float(sim.config.battery_wh)
        self.b = self.cap * init
        self.b_min = self.b
        # infer daily weather from the solar grid (consistent with the simulator's draw)
        n_days = int(math.ceil(days))
        steps_per_day = int(round(86400.0 / self.dt))
        hours = ((np.arange(len(self.solar_wh)) + 0.5) * self.dt % 86400.0) / 3600.0
        clear = np.array([solar_power_w(h, self.plat["panel_w"], "clear", sim.config) for h in hours]) * self.dt / 3600.0
        self.day_weather = []
        for d in range(n_days):
            sl = slice(d * steps_per_day, (d + 1) * steps_per_day)
            ratio = self.solar_wh[sl].sum() / max(clear[sl].sum(), 1e-9)
            self.day_weather.append(min(WEATHER_SOLAR, key=lambda w: abs(WEATHER_SOLAR[w] - ratio)))
        self.gw_dist = [sim.link_distance_m] + [float(rng.uniform(*cfg.link_distance_m)) for _ in range(cfg.n_gateways - 1)]
        ev = sim.events
        self.events = [r._asdict() for r in ev.itertuples(index=False)] if len(ev) else []
        self.ev_idx = 0
        self.t = 0.0
        self.stats = dict(intrusions=sum(1 for e in self.events if self._category(e) == "intrusion"),
                          non_intrusions=sum(1 for e in self.events if self._category(e) == "none"),
                          detected=0, missed=0, missed_asleep=0, false_alarms=0, alerts_sent=0,
                          alerts_delivered=0, camera_uses=0, energy_action_j=0.0, energy_tx_j=0.0,
                          sleep_s=0.0, brownout=0, steps=0, reward=0.0, event_epochs=0)
        self._new_epoch_at(0.0, None)
        return self._obs(), self._info()

    # ------------------------------------------------------------------ helpers
    def _category(self, e) -> str:
        c = e["true_class"]
        if c == "none":
            return "none"
        if c == "cow" and not self.cfg.cow_is_intrusion:
            return "neutral"
        return "intrusion"

    def _weather(self, t: float) -> str:
        return self.day_weather[min(int(t // 86400), len(self.day_weather) - 1)]

    def _sample_rssi(self, t: float) -> List[float]:
        w = self._weather(t)
        out = []
        for d in self.gw_dist:
            sh = float(self.np_random.normal(0, self.simcfg.shadow_sd_db + WEATHER_EXTRA_SHADOW_DB[w]))
            pl = path_loss_db(d, self.simcfg.path_loss_exp) + WEATHER_LOSS_DB[w] + sh
            out.append(self.cfg.tx_power_dbm + 2 * self.simcfg.antenna_gain_dbi - pl)
        return out

    def _new_epoch_at(self, t: float, event: Optional[dict]):
        self.t = t
        self.epoch = dict(type="event" if event is not None else "heartbeat", event=event,
                          stage="sense", camera_used=False, rssi=self._sample_rssi(t))
        if event is not None:
            self.stats["event_epochs"] += 1

    def _obs(self) -> np.ndarray:
        ep = self.epoch
        o = np.zeros(self.obs_dim, dtype=np.float32)
        if ep["type"] == "event":
            e = ep["event"]
            dd = float(e["d_bg_est_m"]) - float(e["distance_m"])
            D = float(e["distance_m"])
            valid = dd > 0 and PAPER["d_valid_min_m"] <= D <= PAPER["d_valid_max_m"]
            o[0] = float(e["pir"])
            o[1] = min(dd / PAPER["d_max_norm_m"], 1.0) if valid else 0.0
            o[2] = 1.0 if valid else 0.0
            o[3] = float(np.clip(dd / 5.0, -1, 1))
            if ep["camera_used"]:
                o[5] = 1.0
                if e["pred_label"] != "background":
                    o[4] = float(e["pred_conf"])
                    o[6 + CLASSES.index(e["pred_label"])] = 1.0
        o[10] = self.b / self.cap
        for g, r in enumerate(ep["rssi"]):
            o[11 + g] = float(np.clip((r + 140.0) / 60.0, 0.0, 1.0))
        if self.cfg.time_features:
            h = (self.t % 86400.0) / 3600.0
            k = 11 + self.cfg.n_gateways
            o[k] = math.sin(2 * math.pi * h / 24); o[k + 1] = math.cos(2 * math.pi * h / 24)
            i = min(int(self.t // self.dt), len(self.solar_wh) - 1)
            o[k + 2] = float(np.clip(self.solar_wh[i] * 3600.0 / self.dt / self.plat["panel_w"], 0, 1))
        return o

    def _info(self, **kw) -> dict:
        d = dict(t_s=self.t, epoch_type=self.epoch["type"], stage=self.epoch["stage"],
                 battery_soc=self.b / self.cap, lagrange_lambda=self.lagrange_lambda)
        d.update(kw)
        return d

    def _advance(self, to_t: float, asleep_until: float) -> bool:
        """Integrate battery on the 1-minute grid from self.t to to_t. Returns True on brownout."""
        i0 = int(self.t // self.dt); i1 = min(int(to_t // self.dt), len(self.solar_wh))
        idle = self.plat["idle_w"] * self.dt / 3600.0
        slp = self.sleep_w * self.dt / 3600.0
        b = self.b
        for i in range(i0, i1):
            use = slp if i * self.dt < asleep_until else idle
            self._consumed_wh += use
            b = b + self.solar_wh[i] - use
            if b > self.cap:
                b = self.cap
            if b <= 0.0:
                self.b = 0.0; self.b_min = 0.0; self.t = i * self.dt
                return True
        self.b = b; self.b_min = min(self.b_min, b); self.t = to_t
        return False

    def _spend(self, joules: float) -> bool:
        self.b -= joules / 3600.0
        self.stats["energy_action_j"] += joules
        if self.b <= 0.0:
            self.b = 0.0; self.b_min = 0.0
            return True
        self.b_min = min(self.b_min, self.b)
        return False

    def _send(self, sf: int, gw: int) -> Tuple[bool, float]:
        rssi = self.epoch["rssi"][gw]
        p = lora_delivery_prob(rssi, sf, self.cfg.sigma_fast_db)
        e1 = lora_tx_energy_j(sf, self.cfg.tx_power_dbm) + lora_rx_energy_j(sf)
        delivered, energy = False, 0.0
        for _ in range(self.cfg.max_attempts):
            energy += e1
            ok = self.np_random.random() < p
            delivered = delivered or ok
            if ok and self.np_random.random() < p:   # ACK received
                break
        return delivered, energy

    # ------------------------------------------------------------------ step
    def step(self, action):
        cfg = self.cfg
        cam, tx, sl, gw = self.decode_action(action)
        self._consumed_wh = 0.0
        comp = dict(correct=0.0, false_alarm=0.0, missed=0.0, reject=0.0, energy=0.0, idle_energy=0.0,
                    soc=0.0, brownout=0.0)
        n_missed = 0
        terminated = truncated = False
        ep = self.epoch
        e = ep["event"]
        energy_pen_j = 0.0

        def brownout():
            nonlocal terminated, n_missed
            terminated = True
            comp["brownout"] += cfg.r_brownout
            self.stats["brownout"] = 1
            # every intrusion not yet resolved is missed (node is down), incl. the current one
            pend = ([e] if (e is not None and not resolved) else []) + self.events[self.ev_idx:]
            m = sum(1 for x in pend if self._category(x) == "intrusion")
            self.ev_idx = len(self.events)
            n_missed += m
            comp["missed"] += cfg.r_missed * m

        resolved = False
        # ---- camera sub-step
        if ep["type"] == "event" and ep["stage"] == "sense" and cam == 1:
            j = self.camera_j + self.infer_j
            self.stats["camera_uses"] += 1
            if not cfg.energy_tx_only:
                energy_pen_j += j
            ep["camera_used"] = True; ep["stage"] = "classified"
            if self._spend(j):
                brownout()
            return self._finish(comp, n_missed, energy_pen_j, terminated, False)

        # ---- resolve event
        if ep["type"] == "event":
            resolved = True
            cat = self._category(e)
            delivered = False
            if tx > 0:
                delivered, j = self._send(SF_OPTIONS[tx - 1], gw)
                self.stats["alerts_sent"] += 1; self.stats["alerts_delivered"] += int(delivered)
                self.stats["energy_tx_j"] += j
                energy_pen_j += j
                if self._spend(j):
                    brownout()
            if cat == "intrusion":
                if delivered:
                    comp["correct"] += cfg.r_correct; self.stats["detected"] += 1
                else:
                    comp["missed"] += cfg.r_missed; n_missed += 1
            elif cat == "none":
                if delivered:
                    comp["false_alarm"] += cfg.r_false_alarm; self.stats["false_alarms"] += 1
                else:
                    comp["reject"] += cfg.r_correct_reject
        if terminated:
            return self._finish(comp, n_missed, energy_pen_j, terminated, False)

        # ---- sleep + advance to next epoch
        sleep_s = cfg.sleep_options_min[sl] * 60.0
        t_now = self.t
        if sleep_s > 0:
            wake = t_now + sleep_s + self.boot_s
            self.stats["sleep_s"] += sleep_s + self.boot_s
            # events during sleep/boot are missed
            while self.ev_idx < len(self.events) and self.events[self.ev_idx]["t_s"] < min(wake, self.horizon_s):
                x = self.events[self.ev_idx]; self.ev_idx += 1
                if self._category(x) == "intrusion":
                    comp["missed"] += cfg.r_missed; n_missed += 1; self.stats["missed_asleep"] += 1
            if self._advance(min(wake, self.horizon_s), asleep_until=wake):
                brownout()
            elif wake < self.horizon_s:
                if cfg.energy_include_boot and not cfg.energy_tx_only:
                    energy_pen_j += self.boot_w * self.boot_s
                if self._spend(self.boot_w * self.boot_s):
                    brownout()
                else:
                    self._new_epoch_at(wake, None)
            else:
                truncated = True
        else:
            # skip PIR=0 intrusions (node never wakes for them -> missed)
            nxt_ev = None
            while self.ev_idx < len(self.events):
                x = self.events[self.ev_idx]
                if int(x["pir"]) == 0 or float(x["t_s"]) < t_now:
                    # PIR never fired (or event overlapped processing) -> cannot be alerted
                    self.ev_idx += 1
                    if self._category(x) == "intrusion":
                        comp["missed"] += cfg.r_missed; n_missed += 1
                    continue
                nxt_ev = x; break
            t_hb = t_now + cfg.heartbeat_min * 60.0
            if nxt_ev is not None and nxt_ev["t_s"] <= t_hb:
                t_next, ev_next = float(nxt_ev["t_s"]), nxt_ev
            else:
                t_next, ev_next = t_hb, None
            if t_next >= self.horizon_s:
                if self._advance(self.horizon_s, asleep_until=-1):
                    brownout()
                else:
                    truncated = True
            elif self._advance(t_next, asleep_until=-1):
                brownout()
            else:
                if ev_next is not None:
                    self.ev_idx += 1
                self._new_epoch_at(t_next, ev_next)
        if not terminated and not truncated:
            soc = self.b / self.cap
            comp["soc"] = -cfg.w_soc_low * max(0.0, cfg.soc_low - soc) / cfg.soc_low
        return self._finish(comp, n_missed, energy_pen_j, terminated, truncated)

    def _finish(self, comp, n_missed, energy_pen_j, terminated, truncated):
        cfg = self.cfg
        comp["energy"] = -cfg.w_energy * energy_pen_j / cfg.energy_ref_j
        comp["idle_energy"] = -cfg.w_idle * getattr(self, "_consumed_wh", 0.0) * 3600.0 / cfg.idle_energy_ref_j
        if cfg.reward_mode == "lexicographic":
            primary = comp["missed"] + comp["brownout"]
            secondary = sum(v for k, v in comp.items() if k not in ("missed", "brownout"))
            r = cfg.lexi_primary_scale * primary + cfg.lexi_secondary_scale * secondary
        else:
            r = sum(comp.values())
            if cfg.reward_mode == "lagrangian":
                r -= self.lagrange_lambda * n_missed
        self.stats["missed"] += n_missed
        self.stats["steps"] += 1
        self.stats["reward"] += r
        info = self._info(reward_components=comp, cost=n_missed)
        if terminated or truncated:
            self.stats["battery_end_soc"] = self.b / self.cap
            self.stats["battery_min_soc"] = self.b_min / self.cap
            self._last_ep_stats = dict(self.stats)
            info["episode_stats"] = dict(self.stats)
        return self._obs(), float(r), bool(terminated), bool(truncated), info


def make_env(**kwargs) -> NetraEnv:
    return NetraEnv(EnvConfig(**kwargs))
