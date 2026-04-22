"""pdm/config.py
================
Project-wide hyperparameter configuration.

Loads values from the ``.env`` file at the project root via *python-dotenv*.
The hardcoded defaults here act as a safe fallback so the project works even
without a ``.env`` file present.  Values in ``.env`` override the defaults;
real OS environment variables override both.

Usage
-----
    from pdm import config as CFG

    env = TurboFanEnv(
        ...,
        scheduled_maintenance_cost=CFG.ENV_SCHEDULED_MAINTENANCE_COST,
    )
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Locate the .env at the project root (one level above the pdm/ package).
_ENV_FILE = Path(__file__).parent.parent / ".env"
# override=False: real OS environment variables take priority over .env values.
load_dotenv(_ENV_FILE, override=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _floatlist(key: str, default: list[float]) -> list[float]:
    raw = os.getenv(key)
    if raw is None:
        return list(default)
    return [float(x.strip()) for x in raw.split(",")]


# ── Environment / reward shaping ─────────────────────────────────────────────
ENV_SCHEDULED_MAINTENANCE_COST: float = _float("ENV_SCHEDULED_MAINTENANCE_COST", -20.0)
ENV_SAFE_CYCLE_REWARD:           float = _float("ENV_SAFE_CYCLE_REWARD",           1.0)
ENV_CATASTROPHIC_FAILURE_PENALTY: float = _float("ENV_CATASTROPHIC_FAILURE_PENALTY", -100.0)
ENV_PROXIMITY_PENALTY_THRESHOLD:  int   = _int("ENV_PROXIMITY_PENALTY_THRESHOLD",   20)
ENV_PROXIMITY_PENALTY_SCALE:      float = _float("ENV_PROXIMITY_PENALTY_SCALE",     2.0)

# ── Economic costs (shared by evaluation and sensitivity scripts) ─────────────
COST_MAINTENANCE_UNIT: float = _float("COST_MAINTENANCE_UNIT", 20.0)
COST_FAILURE_UNIT:     float = _float("COST_FAILURE_UNIT",     100.0)

# ── Data preprocessing ────────────────────────────────────────────────────────
DATA_SUBSET:      str = _str("DATA_SUBSET",      "FD001")
DATA_WINDOW_SIZE: int = _int("DATA_WINDOW_SIZE", 30)

# ── Training pipeline ─────────────────────────────────────────────────────────
TRAIN_NUM_EPISODES:    int = _int("TRAIN_NUM_EPISODES",    500)
TRAIN_SEED:            int = _int("TRAIN_SEED",            42)
TRAIN_FIXED_INTERVAL:  int = _int("TRAIN_FIXED_INTERVAL",  150)

# ── Double DQN ────────────────────────────────────────────────────────────────
DDQN_GAMMA:          float = _float("DDQN_GAMMA",          0.99)
DDQN_LEARNING_RATE:  float = _float("DDQN_LEARNING_RATE",  3e-4)
DDQN_LR_END_FACTOR:  float = _float("DDQN_LR_END_FACTOR",  0.05)
DDQN_LR_DECAY_STEPS: int   = _int("DDQN_LR_DECAY_STEPS",   50_000)
DDQN_EPSILON_START:  float = _float("DDQN_EPSILON_START",   1.0)
DDQN_EPSILON_END:    float = _float("DDQN_EPSILON_END",     0.02)
DDQN_EPSILON_DECAY:  float = _float("DDQN_EPSILON_DECAY",   0.99995)
DDQN_BATCH_SIZE:     int   = _int("DDQN_BATCH_SIZE",        256)
DDQN_BUFFER_CAPACITY: int  = _int("DDQN_BUFFER_CAPACITY",   100_000)
DDQN_TAU:            float = _float("DDQN_TAU",             0.005)
DDQN_HIDDEN_DIM:     int   = _int("DDQN_HIDDEN_DIM",        256)
DDQN_MAX_GRAD_NORM:  float = _float("DDQN_MAX_GRAD_NORM",   5.0)

# ── Prioritised Experience Replay (PER) ───────────────────────────────────────
PER_ALPHA:              float = _float("PER_ALPHA",              0.6)
PER_BETA_START:         float = _float("PER_BETA_START",         0.4)
PER_BETA_END:           float = _float("PER_BETA_END",           1.0)
PER_BETA_ANNEAL_STEPS:  int   = _int("PER_BETA_ANNEAL_STEPS",    200_000)
PER_FAILURE_BOOST:      float = _float("PER_FAILURE_BOOST",      5.0)
PER_FAILURE_RUL_THRESHOLD: int = _int("PER_FAILURE_RUL_THRESHOLD", 30)

# ── Dyna-Q ────────────────────────────────────────────────────────────────────
DYNAQ_PLANNING_STEPS:  int   = _int("DYNAQ_PLANNING_STEPS",   100)
DYNAQ_WORLD_MODEL_LR:  float = _float("DYNAQ_WORLD_MODEL_LR", 3e-4)

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_FDR_THRESHOLD:   int = _int("EVAL_FDR_THRESHOLD",   50)
EVAL_SEED:            int = _int("EVAL_SEED",            42)
EVAL_FIXED_INTERVAL:  int = _int("EVAL_FIXED_INTERVAL",  150)

# ── Sensitivity analysis ──────────────────────────────────────────────────────
SENS_FAILURE_COST_SWEEP: list[float] = _floatlist(
    "SENS_FAILURE_COST_SWEEP", [50.0, 100.0, 200.0, 300.0, 500.0, 1_000.0]
)
SENS_SEED:           int = _int("SENS_SEED",           42)
SENS_FIXED_INTERVAL: int = _int("SENS_FIXED_INTERVAL", 150)
