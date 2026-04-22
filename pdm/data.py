from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


RAW_COLUMNS = [
    "unit_nr",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]
RAW_COLUMNS += [f"sensor_{index}" for index in range(1, 22)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]

# Sensors with zero variance in all FD001 operating conditions — they carry no
# information and add spurious input dimensions to the Q-network.
# Dropping them reduces the state dimension from 22 → 16 and accelerates convergence.
DEAD_SENSOR_COLUMNS: list[str] = [
    "sensor_1",   # Tf  — constant across FD001
    "sensor_5",   # Nf  — constant across FD001
    "sensor_10",  # NRf — constant across FD001
    "sensor_16",  # PCNfRdmd — constant across FD001
    "sensor_18",  # W31 — constant across FD001
    "sensor_19",  # W32 — constant across FD001
]
ACTIVE_SENSOR_COLUMNS: list[str] = [
    col for col in SENSOR_COLUMNS if col not in set(DEAD_SENSOR_COLUMNS)
]


@dataclass(slots=True)
class EngineEpisode:
    engine_id: int
    states: np.ndarray
    rul: np.ndarray
    max_cycle: int


class CMAPSSPreprocessor:
    def __init__(self, data_dir: str | Path, subset: str = "FD001", window_size: int = 30) -> None:
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.window_size = window_size
        self.scaler = MinMaxScaler()

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.data_dir / f"train_{self.subset}.txt"
        test_path = self.data_dir / f"test_{self.subset}.txt"
        if not train_path.exists() or not test_path.exists():
            missing = [str(path) for path in (train_path, test_path) if not path.exists()]
            raise FileNotFoundError(f"Missing CMAPSS files: {missing}")

        train_df = self._read_split(train_path)
        test_df = self._read_split(test_path)
        train_df = self._add_train_rul(train_df)
        train_df, test_df = self._scale_sensors(train_df, test_df)
        return train_df, test_df

    def _read_split(self, path: Path) -> pd.DataFrame:
        dataframe = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=RAW_COLUMNS,
            engine="python",
        )
        return dataframe

    def _add_train_rul(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        max_cycles = dataframe.groupby("unit_nr")["time_cycles"].transform("max")
        dataframe = dataframe.copy()
        dataframe["RUL"] = max_cycles - dataframe["time_cycles"]
        return dataframe

    def _scale_sensors(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.copy()
        test_df = test_df.copy()
        # Scaler is fitted on training data only — prevents data leakage.
        # Dead-sensor columns are excluded so the scaler never sees zero-variance features.
        self.scaler.fit(train_df[ACTIVE_SENSOR_COLUMNS])
        train_df.loc[:, ACTIVE_SENSOR_COLUMNS] = self.scaler.transform(train_df[ACTIVE_SENSOR_COLUMNS])
        test_df.loc[:, ACTIVE_SENSOR_COLUMNS] = self.scaler.transform(test_df[ACTIVE_SENSOR_COLUMNS])
        return train_df, test_df

    def build_episodes(self, train_df: pd.DataFrame) -> Dict[int, EngineEpisode]:
        episodes: Dict[int, EngineEpisode] = {}
        for engine_id, engine_df in train_df.groupby("unit_nr"):
            engine_df = engine_df.sort_values("time_cycles").reset_index(drop=True)
            if len(engine_df) < self.window_size:
                continue

            sensor_roll = (
                engine_df[ACTIVE_SENSOR_COLUMNS]
                .rolling(window=self.window_size, min_periods=self.window_size)
                .mean()
                .iloc[self.window_size - 1 :]
                .to_numpy(dtype=np.float32)
            )
            aligned = engine_df.iloc[self.window_size - 1 :].reset_index(drop=True)
            cycle_feature = aligned[["time_cycles"]].to_numpy(dtype=np.float32)
            states = np.concatenate([sensor_roll, cycle_feature], axis=1)
            episodes[int(engine_id)] = EngineEpisode(
                engine_id=int(engine_id),
                states=states,
                rul=aligned["RUL"].to_numpy(dtype=np.int64),
                max_cycle=int(engine_df["time_cycles"].max()),
            )
        return episodes

    def build_raw_windows(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        windows: list[np.ndarray] = []
        labels: list[int] = []
        for _, engine_df in train_df.groupby("unit_nr"):
            engine_df = engine_df.sort_values("time_cycles").reset_index(drop=True)
            if len(engine_df) < self.window_size:
                continue

            sensor_values = engine_df[ACTIVE_SENSOR_COLUMNS].to_numpy(dtype=np.float32)
            rul_values = engine_df["RUL"].to_numpy(dtype=np.int64)
            for start_index in range(len(engine_df) - self.window_size + 1):
                end_index = start_index + self.window_size
                windows.append(sensor_values[start_index:end_index])
                labels.append(int(rul_values[end_index - 1]))

        return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def split_episodes(
    episodes: Dict[int, EngineEpisode],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[list[EngineEpisode], list[EngineEpisode]]:
    engine_ids = np.array(sorted(episodes.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(engine_ids)
    split_index = max(1, int(len(engine_ids) * (1.0 - validation_fraction)))
    train_ids = engine_ids[:split_index]
    eval_ids = engine_ids[split_index:]
    train_episodes = [episodes[int(engine_id)] for engine_id in train_ids]
    eval_episodes = [episodes[int(engine_id)] for engine_id in eval_ids]
    return train_episodes, eval_episodes
