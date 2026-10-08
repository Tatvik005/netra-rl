"""
netra_simulator.py -- paper-accurate NETRA event simulator (numpy + pandas only)
===============================================================================

Generates seeded, per-event records for the NETRA railway intrusion-detection node
described in:

    [P]  P. Kumar, M. S. Peelam, R. Kumar, S. Kumar, V. Chamola,
         "Smart Railway Obstruction Detection System using IoT and Computer Vision",
         arXiv:2605.08246v1 [cs.CV], 7 May 2026.
    [R]  PhD.pdf research summary ("Adaptive Safety-Critical IoT Detection via DRL").

Every record is ONE PIR-triggered motion event (optionally plus "quiet" PIR=0 samples)
with everything a fixed-rule baseline or an RL agent needs: PIR, ultrasonic reading,
calibrated background distance, the counterfactual vision-model output *if* the camera
were switched on, weather/temperature, synthetic battery state and synthetic LoRa RSSI.

Provenance tags used below
--------------------------
    [PAPER]      value taken directly from arXiv:2605.08246v1 (section/table cited)
    [DERIVED]    computed exactly from paper numbers (e.g. confusion matrices from P/R tables)
    [CALIBRATED] synthetic distribution tuned so the simulator reproduces a paper table
    [DATASHEET]  Semtech SX1276/77/78 datasheet typical value (verify for your revision)
    [SYNTHETIC]  assumption not in the paper -- change freely, results depend on it

Parameter table
---------------
Sensing / fusion inputs
  * Background distance D_bg ~ U(12, 15) m, calibrated as the mean of 5 empty-track
    readings                                                              [PAPER III-A.2]
  * Valid ultrasonic window 4-15 m; Algorithm 1 rejects D<4, D>15, dd<=0 [PAPER Alg.1]
  * D_MAX = 1.5 m normaliser for P_dist                                   [PAPER eq.3]
  * Ultrasonic reading noise sd = 0.03 m (paper quotes HC-SR04 +/-3 mm; outdoor
    12-15 m echoes are far noisier than the spec)                          [SYNTHETIC]
  * Optional speed-of-sound temperature drift (device assumes 343 m/s, real
    v = 331.3 + 0.606*T; calibration temperature = first-event temperature). OFF by
    default because it would create many extra small-dd false triggers the paper does
    not report.                                                            [SYNTHETIC]
  * Intrusion distance-change modes                                         [CALIBRATED]
        clear   p=0.85 : object at U(4.5, D_bg-1.2) m  -> dd >= 1.2 m
        partial p=0.10 : grazing / far-edge echo, dd ~ U(0.65, 0.95) m
        miss    p=0.05 : echo misses the target (background returned) or target
                         closer than 4 m (rejected)
    Tuned so that on the "fusion_79" scenario: prob. fusion (tau=0.65) ~95% (38/40),
    binary fusion (dd >= 1.0 m) ~85% (34/40), tau=0.45 recovers no extra intrusions
    (Tables III-V). The paper gives NO dd distribution; these are fit, not measured.
  * False PIR triggers ("vegetation, birds, passing vehicles")             [PAPER IV-A.1]
    mix vegetation 0.50 / bird 0.25 / vehicle 0.25                          [SYNTHETIC]
        vegetation: 10% leaf clutter in beam, dd ~ U(0.15, 0.55) m, else background
        bird      : 30% in beam but closer than 4 m (rejected), else background
        vehicle   : 50% beyond the background (dd<0 / D>15, rejected), else background
    -> ~5% of false triggers have 0.125 <= dd < 0.625, i.e. pass tau=0.45 but not
       tau=0.65, reproducing "2 false activations at tau=0.45, 0 at 0.65" (Table V)
                                                                           [CALIBRATED]
  * PIR: every scenario event is a PIR trigger (PIR-only detected 40/40 = 100%,
    Table III), so pir_miss_prob = 0 by default. n_quiet adds PIR=0 samples.  [PAPER]

Scenarios (counts)                                                          [PAPER]
  * fusion_79 : 40 human intrusions + 39 false triggers   (Tables III-V; 79 events)
  * events_113: 20 human intrusions + 93 false triggers   (Sec. IV-A.1; 113 events)
  * field_97  : 42 human, 15 cow ("cattle/horse"), 8 elephant, 12 obstacle,
                20 safe/false events                        (Table X; 97 events)
  NOTE: the paper's own counts are inconsistent (20 vs 40 intrusions, 93 vs 39 false
  triggers, 113 vs 79 events). Both variants are provided.

Vision model (counterfactual: what the classifier WOULD output if the camera ran)
  * Pi Zero  : MobileNet-SSD, 5.2 s/frame, confusion matrix           [DERIVED Table VI]
      exact from P/R + stated "28 elephant->cow, 26 cow->elephant"; the split of the
      remaining 22 off-diagonal errors (elephant->human 3, elephant->obstacle 12,
      cow->human 3, human->obstacle 3, obstacle->human 1) is one of the few
      allocations consistent with all four precisions/recalls           [SYNTHETIC]
  * Pi 4     : YOLOv5s ONNX, 0.8 s/frame (Table II; text also says 1.13 s),
      confusion matrix uniquely determined by Table VII + "92% cows -> human" [DERIVED]
  * Confidence: correct label ~ Beta(mean 0.82, sd 0.07); wrong label ~ Beta(0.62,
    0.10); spurious label on a no-object frame ~ Beta(0.35, 0.10). The correct-label
    mean was tuned (0.80 -> 0.82) so that ~39% of field_97 events exceed 0.8
    (Table X: 38/97 = 39.2% "high confidence >0.8"); ~95% of correct detections clear
    the 0.7 critical threshold. Shape (Beta, sd) is not in the paper.  [CALIBRATED]
  * No-object frames return "background" with p = 0.85 (Pi Zero) / 0.90 (Pi 4) [SYNTHETIC]
  * Optional night confidence penalty (camera has no IR, Sec. V-C); OFF by default
                                                                           [SYNTHETIC]

Time / environment
  * Deployment 7 days (Table X), events uniform over days; hour-of-day weights per
    source (elephant mostly night, cow/vehicle day, bird dawn/dusk, rest uniform)
                                                                           [SYNTHETIC]
  * Temperature 18-32 C diurnal sine, min 03:00 max 15:00        [PAPER range, SYNTHETIC shape]
  * Daily weather clear/light_rain/moderate_wind = 0.6/0.2/0.2   [PAPER categories, SYNTHETIC probs]

Energy / battery                                               (all battery values SYNTHETIC)
  * Idle power 0.5 W Pi Zero / 2.7 W Pi 4; inference 2.5 W / 7.5 W        [PAPER Table II]
  * Camera 2 W x 5 s = 10 J per activation                           [PAPER Sec. IV-C]
  * Solar panel 10 W (Pi Zero) / 25 W (Pi 4)                              [PAPER Table II]
  * Solar output = panel * 0.6 derate * 0.6 canopy/tilt * weather * sin(pi*(h-6)/12)
    for 06-18 h; weather factor clear 1.0 / rain 0.35 / wind 0.95        [SYNTHETIC]
  * Battery 40 Wh (Pi Zero) / 100 Wh (Pi 4), start 80% SoC               [SYNTHETIC]
    Expected daily balance with the default weather mix: Pi Zero ~ +24 Wh harvest vs
    12 Wh idle (surplus, battery mostly full); Pi 4 ~ 59 Wh vs 65 Wh idle (deficit
    ~6 Wh/day -> ~14 days from 80% SoC, matching the paper's "14 days (solar-assisted)"
    estimate, Table IX). The canopy factor 0.6 was picked for that match. [CALIBRATED]
  * The paper test rig actually ran from a 5 V 2.5 A supply "simulating" solar/battery;
    no real battery or harvest was measured.                                  [PAPER]

LoRa (all RSSI values SYNTHETIC; the paper reports no RSSI, SF rule or airtime)
  * SX1278, 868 MHz, SF7-SF12, BW 125 kHz, LoRaWAN, ACK + retry          [PAPER]
    (caveat: SX1278 is specified for <=525 MHz; the 868 MHz part is SX1276)
  * Airtime: Semtech formula, 20 B payload, CR 4/5, 8-symbol preamble, explicit
    header, CRC on, LDRO on for SF11-12                        [DATASHEET formula]
  * Sensitivity @125 kHz: SF7 -123, SF8 -126, SF9 -129, SF10 -132, SF11 -133,
    SF12 -136 dBm (SX1276 HF band)                                        [DATASHEET]
  * TX current 120 mA @ +20 dBm (PA_BOOST), 87 mA @ +17, 29 mA @ +13 (RFO); RX 11.5 mA;
    supply 3.3 V; default TX power +20 dBm                                [DATASHEET]
  * Path loss: FSPL(1 m, 868 MHz)=31.2 dB + 10*n*log10(d), n = 3.5 (vegetation),
    +2.15 dBi antennas each side, shadowing sd 4 dB (+1 rain, +2 wind), light-rain
    excess loss 1.5 dB, link distance 1000 m (field test, Fig. 4)          [SYNTHETIC]
    -> at 1 km every SF delivers ~100% (matches paper), at 2 km SF7 becomes lossy.
  * Delivery per attempt: P = Phi((RSSI - sens_SF)/sigma_fast), sigma_fast = 3 dB;
    ACK assumed symmetric                                                  [SYNTHETIC]
  * Noise floor -174 + 10log10(125e3) + 6 dB NF = -117 dBm (for SNR column) [SYNTHETIC]

Latency components
  * Fusion 0.18 s (Pi Zero, Sec. III-C) [PAPER]; inference 5.2 / 0.8 s [PAPER Table II];
    camera warm-up+capture 1.0 s, alert processing 0.2 s, retry back-off 1.0 s
    [SYNTHETIC]. Paper end-to-end: 6.5 s Pi Zero / 2.4 s Pi 4 (Table II).

Output columns (one row per event) -- see COLUMNS_DOC at the bottom of this file.

Usage
-----
    python data/netra_simulator.py                   # writes data/netra_sim_events.csv
    python data/netra_simulator.py --seed 7 --scenarios fusion_79 --platforms pi4

    from netra_simulator import SimConfig, simulate
    res = simulate(SimConfig(scenario="fusion_79", platform="pi_zero", seed=42))
    res.events  # pandas DataFrame
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Paper constants
# ----------------------------------------------------------------------------------
PAPER = {
    "w_pir": 0.4, "w_dist": 0.6, "tau_c": 0.65, "tau_c_alt": 0.45,   # eq.4-5, Table V
    "d_max_norm_m": 1.5, "d_valid_min_m": 4.0, "d_valid_max_m": 15.0,  # eq.3, Alg.1
    "d_bg_range_m": (12.0, 15.0), "n_calib_readings": 5,               # Sec. III-A.2
    "tau_ai": 0.50, "tau_alert": 0.60,                                 # Sec. III-D
    "prio_critical": 0.7, "prio_high": 0.6, "prio_medium": 0.5,        # eq.7
    "camera_w": 2.0, "camera_on_s": 5.0,                               # Sec. IV-C
    "fusion_runtime_s": 0.18,                                          # Sec. III-C
    "temp_range_c": (18.0, 32.0),                                      # Sec. IV-A.1
    "speed_sound_ms": 343.0,                                           # eq.1
}

CLASSES: List[str] = ["human", "elephant", "cow", "obstacle"]
PRED_LABELS: List[str] = CLASSES + ["background"]
DANGEROUS = {"human", "elephant", "obstacle"}      # eq.7 critical/high classes

# Confusion counts: rows = true class, columns = predicted (CLASSES order).
# Pi Zero (MobileNet-SSD, Table VI, 165 images). Diagonal/28/26 exact; rest see docstring.
CONF_COUNTS_PI_ZERO = {
    "human":    {"human": 47, "elephant": 0,  "cow": 0,  "obstacle": 3},
    "elephant": {"human": 3,  "elephant": 6,  "cow": 28, "obstacle": 12},
    "cow":      {"human": 3,  "elephant": 26, "cow": 18, "obstacle": 0},
    "obstacle": {"human": 1,  "elephant": 0,  "cow": 0,  "obstacle": 18},
}
# Pi 4 (YOLOv5s ONNX, Table VII, 170 images) -- uniquely determined by the table.
CONF_COUNTS_PI4 = {
    "human":    {"human": 49, "elephant": 1,  "cow": 0, "obstacle": 0},
    "elephant": {"human": 12, "elephant": 38, "cow": 0, "obstacle": 0},
    "cow":      {"human": 46, "elephant": 2,  "cow": 2, "obstacle": 0},
    "obstacle": {"human": 0,  "elephant": 0,  "cow": 0, "obstacle": 20},
}
PAPER_CLASSIFICATION = {   # (precision, recall, F1) %, overall accuracy %
    "pi_zero": {"elephant": (18.8, 12.2, 14.8), "cow": (39.1, 38.3, 38.7),
                "human": (87.0, 94.0, 90.4), "obstacle": (54.5, 94.7, 69.2), "overall": 53.9},
    "pi4": {"elephant": (92.7, 76.0, 83.5), "cow": (100.0, 4.0, 7.7),
            "human": (45.8, 98.0, 62.4), "obstacle": (100.0, 100.0, 100.0), "overall": 64.1},
}

PLATFORMS: Dict[str, dict] = {
    "pi_zero": dict(
        label="Raspberry Pi Zero W + MobileNet-SSD (OpenCV DNN)",
        idle_w=0.5, infer_w=2.5, infer_s=5.2, panel_w=10.0,       # [PAPER Table II]
        battery_wh=40.0,                                          # [SYNTHETIC]
        p_background_on_empty=0.85,                               # [SYNTHETIC]
        confusion=CONF_COUNTS_PI_ZERO, paper_latency_s=6.5, cost_usd=22,
    ),
    "pi4": dict(
        label="Raspberry Pi 4 (4 GB) + YOLOv5s ONNX INT8",
        idle_w=2.7, infer_w=7.5, infer_s=0.8, panel_w=25.0,       # [PAPER Table II]
        battery_wh=100.0,                                         # [SYNTHETIC]
        p_background_on_empty=0.90,                               # [SYNTHETIC]
        confusion=CONF_COUNTS_PI4, paper_latency_s=2.4, cost_usd=55,
    ),
}

# ----------------------------------------------------------------------------------
# LoRa physics
# ----------------------------------------------------------------------------------
SX127X_SENSITIVITY_DBM = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -133.0, 12: -136.0}
SX127X_TX_CURRENT_A = {20: 0.120, 17: 0.087, 13: 0.029}   # PA_BOOST +20/+17, RFO +13
SX127X_RX_CURRENT_A = 0.0115
SX127X_SUPPLY_V = 3.3
NOISE_FLOOR_DBM = -174.0 + 10 * math.log10(125e3) + 6.0      # ~ -117 dBm


def lora_airtime_s(sf: int, payload_bytes: int = 20, bw_hz: float = 125e3, cr: int = 1,
                   preamble_symbols: int = 8, crc: bool = True, explicit_header: bool = True,
                   ldro: Optional[bool] = None) -> float:
    """Semtech SX127x time-on-air (AN1200.13 / datasheet sec. 4.1.1.7). cr=1 -> 4/5."""
    if ldro is None:
        ldro = sf >= 11                       # mandated for symbol time > 16 ms at 125 kHz
    t_sym = (2 ** sf) / bw_hz
    t_preamble = (preamble_symbols + 4.25) * t_sym
    ih = 0 if explicit_header else 1
    de = 1 if ldro else 0
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return t_preamble + n_payload * t_sym


def lora_tx_current_a(tx_power_dbm: float) -> float:
    """Nearest datasheet point at or above the requested power."""
    for p in sorted(SX127X_TX_CURRENT_A):
        if tx_power_dbm <= p:
            return SX127X_TX_CURRENT_A[p]
    return SX127X_TX_CURRENT_A[20]


def lora_tx_energy_j(sf: int, tx_power_dbm: float = 20.0, payload_bytes: int = 20) -> float:
    return lora_airtime_s(sf, payload_bytes) * lora_tx_current_a(tx_power_dbm) * SX127X_SUPPLY_V


def lora_rx_energy_j(sf: int, ack_payload_bytes: int = 4) -> float:
    """Energy of listening for an ACK frame of the same SF."""
    return lora_airtime_s(sf, ack_payload_bytes) * SX127X_RX_CURRENT_A * SX127X_SUPPLY_V


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lora_delivery_prob(rssi_dbm: float, sf: int, sigma_fast_db: float = 3.0) -> float:
    """Per-attempt packet success probability for a given mean RSSI and SF [SYNTHETIC]."""
    return _phi((rssi_dbm - SX127X_SENSITIVITY_DBM[sf]) / sigma_fast_db)


def path_loss_db(distance_m: float, n: float = 3.5, freq_hz: float = 868e6) -> float:
    fspl_1m = 20 * math.log10(freq_hz) + 20 * math.log10(1.0) - 147.55   # ~31.2 dB
    return fspl_1m + 10 * n * math.log10(max(distance_m, 1.0))


# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
SCENARIOS: Dict[str, dict] = {
    "fusion_79": dict(intrusions={"human": 40}, n_false=39,
                      note="Tables III-V: 40 intrusions + 39 false PIR triggers"),
    "events_113": dict(intrusions={"human": 20}, n_false=93,
                       note="Sec. IV-A.1: 20 human intrusions + 93 environmental triggers"),
    "field_97": dict(intrusions={"human": 42, "cow": 15, "elephant": 8, "obstacle": 12},
                     n_false=20, note="Table X field deployment (97 events)"),
}

HOUR_PROFILES = {
    "uniform":   np.ones(24),
    "day":       np.array([0.2] * 6 + [1.0] * 13 + [0.2] * 5),
    "night":     np.array([1.0] * 6 + [0.3] * 12 + [1.0] * 6),
    "dawn_dusk": np.array([0.3] * 5 + [1.0] * 4 + [0.3] * 7 + [1.0] * 4 + [0.3] * 4),
}
SOURCE_HOURS = {"human": "uniform", "elephant": "night", "cow": "day", "obstacle": "uniform",
                "vegetation": "uniform", "bird": "dawn_dusk", "vehicle": "day", "quiet": "uniform"}
SOURCE_SPEED_KMH = {"human": (4, 12), "elephant": (2, 6), "cow": (2, 5), "obstacle": (0, 0)}


@dataclass
class SimConfig:
    scenario: str = "fusion_79"
    platform: str = "pi_zero"
    seed: int = 42
    intrusions: Optional[Dict[str, int]] = None     # overrides scenario counts
    n_false: Optional[int] = None
    n_quiet: int = 0                                 # PIR=0 background samples (for RL)
    deployment_days: float = 7.0
    grid_step_s: float = 60.0
    # ultrasonic
    d_bg_range_m: Tuple[float, float] = PAPER["d_bg_range_m"]
    n_calib: int = PAPER["n_calib_readings"]
    us_noise_sd_m: float = 0.03
    temperature_drift: bool = False
    # intrusion dd modes [CALIBRATED]
    p_intr_clear: float = 0.85
    p_intr_partial: float = 0.10                     # remainder = miss
    clear_min_obj_m: float = 4.5
    clear_min_dd_m: float = 1.2
    partial_dd_m: Tuple[float, float] = (0.65, 0.95)
    miss_too_close_frac: float = 0.5
    pir_miss_prob: float = 0.0
    # false triggers
    false_mix: Dict[str, float] = field(default_factory=lambda: {"vegetation": 0.5, "bird": 0.25, "vehicle": 0.25})
    veg_clutter_p: float = 0.10
    veg_clutter_dd_m: Tuple[float, float] = (0.15, 0.55)
    bird_in_beam_p: float = 0.30
    vehicle_beyond_p: float = 0.50
    # vision confidences [CALIBRATED]
    conf_correct: Tuple[float, float] = (0.82, 0.07)
    conf_wrong: Tuple[float, float] = (0.62, 0.10)
    conf_spurious: Tuple[float, float] = (0.35, 0.10)
    night_conf_penalty: float = 0.0                  # e.g. 0.3 -> conf *= 0.7 at night
    # environment
    weather_probs: Dict[str, float] = field(default_factory=lambda: {"clear": 0.6, "light_rain": 0.2, "moderate_wind": 0.2})
    # energy
    solar_derate: float = 0.6
    canopy_factor: float = 0.6                       # chosen so Pi 4 autonomy ~14 days
    battery_wh: Optional[float] = None               # default from platform
    battery_init_frac: float = 0.8
    # LoRa
    link_distance_m: Tuple[float, float] = (1000.0, 1000.0)
    path_loss_exp: float = 3.5
    antenna_gain_dbi: float = 2.15
    shadow_sd_db: float = 4.0
    tx_power_dbm: float = 20.0

    def resolved(self) -> "SimConfig":
        sc = SCENARIOS.get(self.scenario, {})
        c = replace(self)
        if c.intrusions is None:
            c.intrusions = dict(sc.get("intrusions", {"human": 40}))
        if c.n_false is None:
            c.n_false = int(sc.get("n_false", 39))
        if c.battery_wh is None:
            c.battery_wh = PLATFORMS[c.platform]["battery_wh"]
        return c


WEATHER_SOLAR = {"clear": 1.0, "light_rain": 0.35, "moderate_wind": 0.95}
WEATHER_LOSS_DB = {"clear": 0.0, "light_rain": 1.5, "moderate_wind": 0.0}
WEATHER_EXTRA_SHADOW_DB = {"clear": 0.0, "light_rain": 1.0, "moderate_wind": 2.0}


@dataclass
class SimResult:
    events: pd.DataFrame
    grid_t_s: np.ndarray            # grid start times
    grid_net_wh: np.ndarray         # solar harvest - idle consumption per grid step
    config: SimConfig
    d_bg_true_m: float
    d_bg_est_m: float
    link_distance_m: float


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def beta_from_mean_sd(rng: np.random.Generator, mean: float, sd: float) -> float:
    var = sd ** 2
    k = mean * (1 - mean) / var - 1
    return float(rng.beta(mean * k, (1 - mean) * k))


def temperature_c(hour: float) -> float:
    lo, hi = PAPER["temp_range_c"]
    return (lo + hi) / 2 + (hi - lo) / 2 * math.sin(2 * math.pi * (hour - 9.0) / 24.0)


def solar_power_w(hour: float, panel_w: float, weather: str, cfg: SimConfig) -> float:
    if hour < 6.0 or hour > 18.0:
        return 0.0
    return panel_w * cfg.solar_derate * cfg.canopy_factor * WEATHER_SOLAR[weather] * \
        math.sin(math.pi * (hour - 6.0) / 12.0)


def speed_of_sound(temp_c: float) -> float:
    return 331.3 + 0.606 * temp_c


def confusion_probs(platform: str, true_cls: str) -> np.ndarray:
    row = PLATFORMS[platform]["confusion"][true_cls]
    v = np.array([row[c] for c in CLASSES], dtype=float)
    return v / v.sum()


def run_battery(grid_net_wh: np.ndarray, cap_wh: float, b0_wh: float,
                event_steps: Optional[np.ndarray] = None,
                event_costs_wh: Optional[np.ndarray] = None) -> np.ndarray:
    """Clamped battery trajectory on the fine grid; returns battery at START of each step
    after any event cost charged in that step (length = n_steps + 1)."""
    n = len(grid_net_wh)
    costs = np.zeros(n)
    if event_steps is not None and event_costs_wh is not None:
        np.add.at(costs, event_steps, event_costs_wh)
    out = np.empty(n + 1)
    b = b0_wh
    out[0] = b
    for i in range(n):
        b = b + grid_net_wh[i] - costs[i]
        if b > cap_wh:
            b = cap_wh
        elif b < 0.0:
            b = 0.0
        out[i + 1] = b
    return out


# ----------------------------------------------------------------------------------
# Main generator
# ----------------------------------------------------------------------------------
def simulate(cfg: SimConfig) -> SimResult:
    cfg = cfg.resolved()
    if cfg.platform not in PLATFORMS:
        raise ValueError(f"unknown platform {cfg.platform}")
    plat = PLATFORMS[cfg.platform]
    rng = np.random.default_rng(cfg.seed)

    # --- deployment-level draws -----------------------------------------------------
    n_days = int(math.ceil(cfg.deployment_days))
    wnames = list(cfg.weather_probs)
    wp = np.array([cfg.weather_probs[w] for w in wnames]); wp /= wp.sum()
    day_weather = [wnames[i] for i in rng.choice(len(wnames), size=n_days, p=wp)]
    d_bg_true = float(rng.uniform(*cfg.d_bg_range_m))
    link_d = float(rng.uniform(*cfg.link_distance_m))

    # --- event list -------------------------------------------------------------------
    sources: List[Tuple[str, str]] = []           # (source, true_class)
    for cls, n in cfg.intrusions.items():
        if cls not in CLASSES:
            raise ValueError(f"unknown intrusion class {cls}")
        sources += [(cls, cls)] * int(n)
    fnames = list(cfg.false_mix)
    fp = np.array([cfg.false_mix[f] for f in fnames]); fp /= fp.sum()
    for i in rng.choice(len(fnames), size=cfg.n_false, p=fp):
        sources.append((fnames[i], "none"))
    sources += [("quiet", "none")] * cfg.n_quiet

    total_s = cfg.deployment_days * 86400.0
    times = []
    for src, _ in sources:
        w = HOUR_PROFILES[SOURCE_HOURS.get(src, "uniform")]
        while True:
            day = int(rng.integers(0, n_days))
            hr = int(rng.choice(24, p=w / w.sum()))
            t = day * 86400.0 + hr * 3600.0 + float(rng.uniform(0, 3600))
            if t < total_s:
                break
        times.append(t)
    order = np.argsort(times)
    sources = [sources[i] for i in order]
    times = [times[i] for i in order]

    # --- calibration (at deployment start, temperature of hour 0 of day 0) -------------
    t_calib = temperature_c(0.0)
    def measure(true_d: float, temp: float) -> float:
        d = true_d
        if cfg.temperature_drift:           # device assumes 343 m/s
            d = true_d * PAPER["speed_sound_ms"] / speed_of_sound(temp)
        return d + float(rng.normal(0, cfg.us_noise_sd_m))
    d_bg_est = float(np.mean([measure(d_bg_true, t_calib) for _ in range(cfg.n_calib)]))

    # --- power grid -------------------------------------------------------------------
    n_steps = int(math.ceil(total_s / cfg.grid_step_s))
    grid_t = np.arange(n_steps) * cfg.grid_step_s
    mid_h = ((grid_t + cfg.grid_step_s / 2) % 86400.0) / 3600.0
    day_idx = np.minimum((grid_t // 86400).astype(int), n_days - 1)
    solar_w = np.array([solar_power_w(h, plat["panel_w"], day_weather[d], cfg)
                        for h, d in zip(mid_h, day_idx)])
    grid_net_wh = (solar_w - plat["idle_w"]) * cfg.grid_step_s / 3600.0
    cap = float(cfg.battery_wh)
    b0 = cap * cfg.battery_init_frac
    batt_idle = run_battery(grid_net_wh, cap, b0)
    cum_solar = np.concatenate([[0.0], np.cumsum(solar_w * cfg.grid_step_s / 3600.0)])

    rows = []
    prev_step = 0
    for k, ((src, cls), t) in enumerate(zip(sources, times)):
        step = min(int(t // cfg.grid_step_s), n_steps - 1)
        hour = (t % 86400.0) / 3600.0
        day = int(t // 86400.0)
        weather = day_weather[min(day, n_days - 1)]
        temp = temperature_c(hour)
        is_intr = cls != "none"
        night = hour < 6.0 or hour >= 18.0

        # PIR
        if src == "quiet":
            pir = 0
        elif is_intr and rng.random() < cfg.pir_miss_prob:
            pir = 0
        else:
            pir = 1

        # true target distance (None -> background echo)
        mode = "background"
        target = d_bg_true
        if is_intr:
            u = rng.random()
            if u < cfg.p_intr_clear:
                mode = "clear"
                target = float(rng.uniform(cfg.clear_min_obj_m, d_bg_true - cfg.clear_min_dd_m))
            elif u < cfg.p_intr_clear + cfg.p_intr_partial:
                mode = "partial"
                target = d_bg_true - float(rng.uniform(*cfg.partial_dd_m))
            else:
                if rng.random() < cfg.miss_too_close_frac:
                    mode = "miss_too_close"
                    target = float(rng.uniform(1.0, 3.9))
                else:
                    mode = "miss_beam"
        elif src == "vegetation" and rng.random() < cfg.veg_clutter_p:
            mode = "veg_clutter"
            target = d_bg_true - float(rng.uniform(*cfg.veg_clutter_dd_m))
        elif src == "bird" and rng.random() < cfg.bird_in_beam_p:
            mode = "bird_close"
            target = float(rng.uniform(0.3, 3.9))
        elif src == "vehicle" and rng.random() < cfg.vehicle_beyond_p:
            mode = "vehicle_beyond"
            target = d_bg_true + float(rng.uniform(0.5, 5.0))
        distance = measure(target, temp)
        delta_d = d_bg_est - distance

        # vision (counterfactual if camera switched on)
        if is_intr:
            probs = confusion_probs(cfg.platform, cls)
            pred = CLASSES[int(rng.choice(len(CLASSES), p=probs))]
            conf = beta_from_mean_sd(rng, *(cfg.conf_correct if pred == cls else cfg.conf_wrong))
        else:
            if rng.random() < plat["p_background_on_empty"]:
                pred, conf = "background", float(rng.uniform(0.0, 0.3))
            else:
                pred = CLASSES[int(rng.integers(0, len(CLASSES)))]
                conf = beta_from_mean_sd(rng, *cfg.conf_spurious)
        if night and cfg.night_conf_penalty > 0:
            conf *= (1.0 - cfg.night_conf_penalty)

        # LoRa link [SYNTHETIC]
        shadow = float(rng.normal(0, cfg.shadow_sd_db + WEATHER_EXTRA_SHADOW_DB[weather]))
        pl = path_loss_db(link_d, cfg.path_loss_exp) + WEATHER_LOSS_DB[weather] + shadow
        rssi = cfg.tx_power_dbm + 2 * cfg.antenna_gain_dbi - pl

        rows.append(dict(
            event_id=k, scenario=cfg.scenario, platform=cfg.platform, seed=cfg.seed,
            t_s=round(t, 1), day=day, hour=round(hour, 3), hour_int=int(hour),
            is_night=int(night), weather=weather, temp_c=round(temp, 2),
            source=src, true_class=cls, is_intrusion=int(is_intr),
            is_dangerous=int(cls in DANGEROUS), speed_kmh=round(float(rng.uniform(*SOURCE_SPEED_KMH.get(cls, (0, 0)))), 1) if is_intr else 0.0,
            us_mode=mode, pir=pir,
            d_bg_true_m=round(d_bg_true, 4), d_bg_est_m=round(d_bg_est, 4),
            distance_m=round(distance, 4), delta_d_m=round(delta_d, 4),
            pred_label=pred, pred_conf=round(conf, 4),
            link_distance_m=round(link_d, 1), path_loss_db=round(pl, 2),
            tx_power_dbm=cfg.tx_power_dbm, rssi_dbm=round(rssi, 2),
            snr_db=round(rssi - NOISE_FLOOR_DBM, 2),
            grid_step=step,
            solar_wh_since_prev=round(float(cum_solar[step] - cum_solar[prev_step]), 5),
            idle_wh_since_prev=round(plat["idle_w"] * (step - prev_step) * cfg.grid_step_s / 3600.0, 5),
            battery_wh_idle_only=round(float(batt_idle[step]), 4),
            battery_frac_idle_only=round(float(batt_idle[step]) / cap, 4),
            battery_capacity_wh=cap,
        ))
        prev_step = step

    df = pd.DataFrame(rows)
    return SimResult(events=df, grid_t_s=grid_t, grid_net_wh=grid_net_wh, config=cfg,
                     d_bg_true_m=d_bg_true, d_bg_est_m=d_bg_est, link_distance_m=link_d)


COLUMNS_DOC = {
    "event_id": "index in time order within one (scenario, platform, seed) run",
    "t_s / day / hour": "time since deployment start (00:00 day 0)",
    "source": "human/elephant/cow/obstacle (intrusion) or vegetation/bird/vehicle/quiet",
    "true_class": "ground-truth class; 'none' for non-intrusions (hide from agents)",
    "us_mode": "how the ultrasonic echo was generated (clear/partial/miss_*/background/...)",
    "pir": "PIR output 0/1",
    "d_bg_est_m": "calibrated background distance (mean of 5 readings)",
    "distance_m / delta_d_m": "ultrasonic reading and d_bg_est - reading",
    "pred_label / pred_conf": "vision output IF the camera were activated (counterfactual)",
    "rssi_dbm / snr_db": "SYNTHETIC LoRa link quality at tx_power_dbm",
    "path_loss_db": "SYNTHETIC path loss incl. shadowing (rssi = tx + 2*gain - path_loss)",
    "solar_wh_since_prev / idle_wh_since_prev": "SYNTHETIC energy flows since previous event",
    "battery_wh_idle_only": "SYNTHETIC battery if the node only idled (no actions)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--platforms", nargs="+", default=list(PLATFORMS))
    ap.add_argument("--n-quiet", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV path (default data/netra_sim_events.csv)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "netra_sim_events.csv")
    frames = []
    for sc in args.scenarios:
        for pf in args.platforms:
            res = simulate(SimConfig(scenario=sc, platform=pf, seed=args.seed, n_quiet=args.n_quiet))
            frames.append(res.events)
            e = res.events
            print(f"{sc:<11} {pf:<8} events={len(e):4d} intrusions={int(e.is_intrusion.sum()):3d} "
                  f"D_bg={res.d_bg_true_m:.2f} m (est {res.d_bg_est_m:.3f}) "
                  f"RSSI mean={e.rssi_dbm.mean():.1f} dBm  batt_end={e.battery_frac_idle_only.iloc[-1]:.2f}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print("LoRa airtime (20 B, 125 kHz, CR4/5): " +
          ", ".join(f"SF{sf}={lora_airtime_s(sf)*1000:.1f} ms" for sf in range(7, 13)))


if __name__ == "__main__":
    main()
