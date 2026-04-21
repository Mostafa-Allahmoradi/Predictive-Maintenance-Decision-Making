"""
Turbofan Predictive Maintenance — Streamlit Demo Dashboard
===========================================================
Visualises step-by-step decisions made by Double DQN, Dyna-Q, Fixed-Interval,
and Random policies on NASA CMAPSS FD001 engines.

Run with:
    streamlit run app.py

Keyboard shortcuts during demo
    ▶  Play button   — starts auto-advance at the chosen speed
    ⏸  Pause button  — freezes the simulation at the current step
    ↺  Reset button  — resets the selected engine and clears history
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from pdm.agents import DoubleDQNAgent
from pdm.data import CMAPSSPreprocessor, EngineEpisode

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

# CMAPSS sensor index → physical name (state vector layout: sensor_1..21, cycle)
CRITICAL_SENSORS: dict[str, int] = {
    "T24 · LPC outlet temp": 1,
    "T30 · HPC outlet temp": 2,
    "T50 · LPT outlet temp": 3,
    "P30 · HPC pressure":    6,
}
SENSOR_COLORS = ["#00b4d8", "#f77f00", "#2dc653", "#e8c400"]
SENSOR_TAIL = 60  # cycles to show in the live sensor window

SCHEDULED_MAINTENANCE_COST: float = -20.0
SAFE_CYCLE_REWARD: float = 1.0
CATASTROPHIC_FAILURE_PENALTY: float = -100.0
FAILURE_RUL_WARNING: int = 50   # highlight threshold on the RUL gauge

CHECKPOINTS: dict[str, Path] = {
    "Double DQN": ARTIFACTS_DIR / "checkpoint_DoubleDQN.pt",
    "Dyna-Q":     ARTIFACTS_DIR / "checkpoint_DynaQ.pt",
}

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PdM RL Dashboard — CMAPSS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS — dark theme, status badge animations, metric card polish
st.markdown(
    """
    <style>
    /* Force dark background for sidebar */
    section[data-testid="stSidebar"] { background: #0d1117; }

    /* Status badge variants */
    .status-safe {
        background: #155724; color: #d4edda;
        padding: 10px 20px; border-radius: 8px;
        font-size: 1.1rem; font-weight: bold; text-align: center;
        border: 1px solid #1e7e34;
    }
    .status-warning {
        background: #856404; color: #fff3cd;
        padding: 10px 20px; border-radius: 8px;
        font-size: 1.1rem; font-weight: bold; text-align: center;
        border: 1px solid #d4a017;
        animation: pulse 1.2s ease-in-out infinite;
    }
    .status-emergency {
        background: #721c24; color: #f8d7da;
        padding: 10px 20px; border-radius: 8px;
        font-size: 1.1rem; font-weight: bold; text-align: center;
        border: 1px solid #c0392b;
        animation: pulse 0.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%   { opacity: 1.0; }
        50%  { opacity: 0.55; }
        100% { opacity: 1.0; }
    }

    /* Plotly chart padding fix */
    div[data-testid="stPlotlyChart"] { padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CMAPSS FD001 dataset…")
def _load_all_episodes() -> dict[int, EngineEpisode]:
    preprocessor = CMAPSSPreprocessor(DATA_DIR, subset="FD001", window_size=30)
    train_df, _ = preprocessor.load()
    return preprocessor.build_episodes(train_df)


@st.cache_resource(show_spinner="Loading trained checkpoints…")
def _load_agents() -> dict[str, DoubleDQNAgent | None]:
    result: dict[str, DoubleDQNAgent | None] = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if ckpt_path.exists():
            result[name] = DoubleDQNAgent.from_checkpoint(str(ckpt_path), device="cpu")
        else:
            result[name] = None
    return result


# ── Policy helpers ────────────────────────────────────────────────────────────

def _compute_action(
    policy: str,
    state: np.ndarray,
    agent: DoubleDQNAgent | None,
    fixed_interval: int,
) -> int:
    if policy == "Fixed-Interval":
        return 1 if int(round(float(state[-1]))) >= fixed_interval else 0
    if policy == "Random":
        return int(np.random.randint(2))
    # RL policies
    if agent is None:
        return int(np.random.randint(2))
    return agent.act(state, explore=False)


def _get_q_values(agent: DoubleDQNAgent | None, state: np.ndarray) -> tuple[float, float]:
    """Return (q_stay, q_maintain). Returns (0, 0) when agent is unavailable."""
    if agent is None:
        return 0.0, 0.0
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = agent.online_network(state_t).squeeze().numpy()
    return float(q[0]), float(q[1])


# ── Environment step (mirrors TurboFanEnv.step exactly) ──────────────────────

def _env_step(episode: EngineEpisode, step_idx: int, action: int) -> dict:
    state = episode.states[step_idx]
    current_rul = int(episode.rul[step_idx])

    if action == 1:
        reward = SCHEDULED_MAINTENANCE_COST
        terminated = True
        next_idx = step_idx
        event = "scheduled_maintenance"
    elif current_rul <= 0:
        reward = CATASTROPHIC_FAILURE_PENALTY
        terminated = True
        next_idx = step_idx
        event = "catastrophic_failure"
    else:
        reward = SAFE_CYCLE_REWARD
        next_idx = step_idx + 1
        if next_idx >= len(episode.states):
            terminated = True
            event = "completed"
        else:
            terminated = False
            event = "safe_operation"

    return {
        "next_idx": next_idx,
        "reward": reward,
        "terminated": terminated,
        "event": event,
        "state": state,
        "rul": current_rul,
        "cycle": int(round(float(state[-1]))),
    }


# ── Session-state management ──────────────────────────────────────────────────

def _blank_history() -> dict:
    return {
        "cycle": [],
        "rul_actual": [],
        **{f"sensor_{name}": [] for name in CRITICAL_SENSORS},
        "q_stay": [],
        "q_maintain": [],
        "reward": [],
        "event": [],
        "action": [],
    }


def _init_simulation(episode: EngineEpisode) -> None:
    ss = st.session_state
    ss.step_idx = 0
    ss.cumulative_reward = 0.0
    ss.done = False
    ss.episode = episode
    ss.history = _blank_history()
    ss.playing = ss.get("playing", False) and False  # always pause on reset


# ── Chart factories ───────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,22,30,0.9)",
    font=dict(color="#c9d1d9"),
    margin=dict(l=4, r=4, t=36, b=4),
)


def _sensor_chart(history: dict) -> go.Figure:
    fig = go.Figure()
    cycles = history["cycle"][-SENSOR_TAIL:]

    for color, (name, _) in zip(SENSOR_COLORS, CRITICAL_SENSORS.items()):
        vals = history[f"sensor_{name}"][-SENSOR_TAIL:]
        fig.add_trace(go.Scatter(
            x=cycles, y=vals, mode="lines",
            name=name.split("·")[0].strip(),
            line=dict(color=color, width=2),
        ))

    # Mark terminal events with vertical lines
    events = history["event"][-SENSOR_TAIL:]
    for i, ev in enumerate(events):
        if ev == "scheduled_maintenance":
            fig.add_vline(x=cycles[i], line=dict(color="#f77f00", dash="dash", width=1.5),
                          annotation_text="Maint.", annotation_position="top left",
                          annotation_font=dict(color="#f77f00", size=10))
        elif ev == "catastrophic_failure":
            fig.add_vline(x=cycles[i], line=dict(color="#e63946", dash="solid", width=2),
                          annotation_text="FAIL", annotation_position="top left",
                          annotation_font=dict(color="#e63946", size=10))

    fig.update_layout(
        **_CHART_LAYOUT,
        height=300,
        title=dict(text="Live Sensor Feed  (last 60 cycles)", x=0, font=dict(size=14)),
        xaxis=dict(gridcolor="#21262d", title="Cycle"),
        yaxis=dict(gridcolor="#21262d", title="Normalised value", range=[-0.05, 1.08]),
        legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
    )
    return fig


def _qvalue_chart(q_stay: float, q_maintain: float, policy: str) -> go.Figure:
    consider_maint = q_maintain > q_stay

    if policy in ("Double DQN", "Dyna-Q"):
        labels = ["Continue (a=0)", "Maintain (a=1)"]
        values = [q_stay, q_maintain]
        colors = [
            "#2dc653" if not consider_maint else "#3d4f43",
            "#e63946" if consider_maint else "#4f3d3d",
        ]
        text_vals = [f"{q_stay:.4f}", f"{q_maintain:.4f}"]
        x_title = "Q-value"
        title = "🧠 Agent Q-Values  [Stay vs. Maintain]"
    else:
        # Fixed-interval: rule coverage; Random: 50/50
        labels = ["Continue", "Maintain"]
        values = [0.5, 0.5]
        colors = ["#4c78a8", "#4c78a8"]
        text_vals = ["50%", "50%"]
        x_title = "Probability (uniform)"
        title = "🎲 Random Policy — No Q-Values"

    fig = go.Figure(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=text_vals,
        textposition="outside",
        textfont=dict(size=13, color="#c9d1d9"),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=185,
        title=dict(text=title, x=0, font=dict(size=13)),
        xaxis=dict(gridcolor="#21262d", title=x_title),
        showlegend=False,
    )
    return fig


def _fi_coverage_chart(current_cycle: int, fixed_interval: int) -> go.Figure:
    pct = min(current_cycle / max(fixed_interval, 1), 1.0)
    fig = go.Figure(go.Bar(
        y=["Continue", "Maintain"],
        x=[1.0 - pct, pct],
        orientation="h",
        marker=dict(
            color=["#2dc653" if pct < 0.8 else "#4f3d3d", "#e63946" if pct >= 1.0 else "#856404"],
            line=dict(width=0),
        ),
        text=[f"{(1-pct)*100:.0f}%", f"{pct*100:.0f}%"],
        textposition="outside",
        textfont=dict(size=13, color="#c9d1d9"),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=185,
        title=dict(text=f"📏 Rule Coverage  (cycle {current_cycle} / interval {fixed_interval})", x=0, font=dict(size=13)),
        xaxis=dict(range=[0, 1.25], gridcolor="#21262d", tickformat=".0%"),
        showlegend=False,
    )
    return fig


def _rul_gauge(rul_actual: int, max_rul: int) -> go.Figure:
    pct = rul_actual / max(max_rul, 1)
    bar_color = "#2dc653" if pct > 0.5 else ("#f39c12" if pct > 0.2 else "#e63946")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul_actual,
        number=dict(suffix=" cycles", font=dict(size=32, color="#c9d1d9")),
        gauge=dict(
            axis=dict(
                range=[0, max_rul],
                tickwidth=1,
                tickcolor="#c9d1d9",
                tickfont=dict(color="#c9d1d9"),
            ),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="rgba(18,22,30,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, max_rul * 0.20], color="rgba(230,57,70,0.18)"),
                dict(range=[max_rul * 0.20, max_rul * 0.50], color="rgba(243,156,18,0.12)"),
                dict(range=[max_rul * 0.50, max_rul], color="rgba(45,198,83,0.08)"),
            ],
            threshold=dict(
                line=dict(color="#c9d1d9", width=3),
                thickness=0.8,
                value=FAILURE_RUL_WARNING,
            ),
        ),
        title=dict(text="Actual RUL", font=dict(color="#c9d1d9", size=14)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _cumulative_cost_chart(history: dict) -> go.Figure:
    if not history["reward"]:
        return go.Figure()
    cumulative = np.cumsum(history["reward"]).tolist()
    cycles = history["cycle"]
    last = cumulative[-1]
    line_color = "#2dc653" if last >= 0 else "#e63946"
    fill_color = "rgba(45,198,83,0.12)" if last >= 0 else "rgba(230,57,70,0.12)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=cumulative,
        mode="lines", fill="tozeroy",
        line=dict(color=line_color, width=2.5),
        fillcolor=fill_color,
        name="Cumulative reward",
        hovertemplate="Cycle %{x}<br>Reward: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#555", dash="dot", width=1))

    # Annotate maintenance and failure events
    for i, ev in enumerate(history["event"]):
        if ev == "scheduled_maintenance":
            fig.add_annotation(
                x=cycles[i], y=cumulative[i],
                text="M", showarrow=False,
                font=dict(color="#f77f00", size=10),
                bgcolor="rgba(0,0,0,0.5)",
            )
        elif ev == "catastrophic_failure":
            fig.add_annotation(
                x=cycles[i], y=cumulative[i],
                text="F", showarrow=False,
                font=dict(color="#e63946", size=10),
                bgcolor="rgba(0,0,0,0.5)",
            )

    fig.update_layout(
        **_CHART_LAYOUT,
        height=220,
        title=dict(text="Cumulative Operational Reward", x=0, font=dict(size=14)),
        xaxis=dict(gridcolor="#21262d", title="Cycle"),
        yaxis=dict(gridcolor="#21262d", title="Total reward"),
        showlegend=False,
    )
    return fig


# ── Status badge ──────────────────────────────────────────────────────────────

def _status_html(event: str, q_maintain: float, q_stay: float, rul: int) -> str:
    if event == "catastrophic_failure":
        return '<div class="status-emergency">🔴 EMERGENCY SHUTDOWN — Catastrophic Failure Occurred</div>'
    if event == "scheduled_maintenance":
        return '<div class="status-emergency">🟠 MAINTENANCE TRIGGERED — Engine Taken Offline</div>'
    if q_maintain > q_stay or rul < FAILURE_RUL_WARNING:
        return '<div class="status-warning">🟡 CONSIDERING MAINTENANCE — Agent Detects Critical Zone</div>'
    return '<div class="status-safe">🟢 SAFE OPERATION — Continuing Normal Run</div>'


# ═══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ PdM Control Panel")
    st.markdown("---")

    # ── Data check ──────────────────────────────────────────────────────────
    if not (DATA_DIR.exists() and any(DATA_DIR.glob("train_*.txt"))):
        st.error(
            f"CMAPSS data not found in `{DATA_DIR}/`.\n\n"
            "Place `train_FD001.txt` and `test_FD001.txt` there, then refresh."
        )
        st.stop()

    all_episodes = _load_all_episodes()
    agents = _load_agents()

    # ── Engine selector ──────────────────────────────────────────────────────
    engine_ids = sorted(all_episodes.keys())
    selected_engine_id = st.selectbox(
        "🔩 Test Engine ID",
        options=engine_ids,
        index=0,
        help="Engine unit number from CMAPSS FD001 (1–100).",
    )
    selected_episode = all_episodes[selected_engine_id]

    st.markdown(
        f"<small>📋 Max cycles recorded: **{selected_episode.max_cycle}** &nbsp;|&nbsp; "
        f"Initial RUL: **{selected_episode.rul[0]}** cycles &nbsp;|&nbsp; "
        f"Sim steps: **{len(selected_episode.states)}**</small>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Policy selector ──────────────────────────────────────────────────────
    policy_options = ["Fixed-Interval", "Double DQN", "Dyna-Q", "Random"]
    selected_policy = st.selectbox(
        "🤖 Decision Policy",
        options=policy_options,
        index=0,
        help="Choose which policy drives maintenance decisions.",
    )

    fixed_interval = 150
    if selected_policy == "Fixed-Interval":
        fixed_interval = st.slider(
            "Maintenance Interval (cycles)",
            min_value=30, max_value=300, value=150, step=10,
            help="Trigger maintenance when cycle count reaches this value.",
        )

    # Checkpoint status indicators
    if selected_policy in ("Double DQN", "Dyna-Q"):
        rl_key = selected_policy  # matches CHECKPOINTS keys
        if agents.get(rl_key) is None:
            st.warning(
                f"⚠️ No checkpoint for **{selected_policy}**.\n\n"
                "Run `python train_compare.py` first.\n"
                "Falling back to random actions until trained."
            )
        else:
            st.success(f"✅ {selected_policy} checkpoint loaded")

    st.markdown("---")

    # ── Playback controls ────────────────────────────────────────────────────
    speed = st.slider(
        "▶️ Simulation Speed",
        min_value=0.05, max_value=2.0, value=0.4, step=0.05,
        format="%.2f s/step",
        help="Delay between simulation steps. Lower = faster.",
    )

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        play_btn = st.button("▶ Play", use_container_width=True)
    with btn_col2:
        pause_btn = st.button("⏸ Pause", use_container_width=True)
    with btn_col3:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    st.markdown("---")

    # ── Cost reference ───────────────────────────────────────────────────────
    st.markdown("**💰 Reward Structure**")
    st.markdown(
        "🟢 Safe operation &nbsp;&nbsp;`+1 / cycle`  \n"
        "🟠 Scheduled maint. &nbsp;`−20`  \n"
        "🔴 Catastrophic fail. &nbsp;`−100`",
    )

    st.markdown("---")
    st.caption("NASA CMAPSS FD001 · Double DQN + Dyna-Q · PER")


# ═══════════════════════════════════════════════════════════════════════════════
#   SESSION STATE — manage resets and button events
# ═══════════════════════════════════════════════════════════════════════════════

_reset_needed = (
    reset_btn
    or "episode" not in st.session_state
    or st.session_state.get("_engine_id") != selected_engine_id
    or st.session_state.get("_policy") != selected_policy
)

if _reset_needed:
    _init_simulation(selected_episode)
    st.session_state["_engine_id"] = selected_engine_id
    st.session_state["_policy"] = selected_policy
    st.session_state.playing = False

if play_btn:
    st.session_state.playing = True
if pause_btn:
    st.session_state.playing = False

ss = st.session_state

# ── Resolve active agent ──────────────────────────────────────────────────────
if selected_policy == "Double DQN":
    active_agent: DoubleDQNAgent | None = agents.get("Double DQN")
elif selected_policy == "Dyna-Q":
    active_agent = agents.get("Dyna-Q")
else:
    active_agent = None

# ── Step ──────────────────────────────────────────────────────────────────────
if ss.playing and not ss.done:
    _state = ss.episode.states[ss.step_idx]
    _action = _compute_action(selected_policy, _state, active_agent, fixed_interval)
    _result = _env_step(ss.episode, ss.step_idx, _action)

    h = ss.history
    h["cycle"].append(_result["cycle"])
    h["rul_actual"].append(_result["rul"])
    for _name, _idx in CRITICAL_SENSORS.items():
        h[f"sensor_{_name}"].append(float(_state[_idx]))
    _qs, _qm = _get_q_values(active_agent, _state)
    h["q_stay"].append(_qs)
    h["q_maintain"].append(_qm)
    h["reward"].append(_result["reward"])
    h["event"].append(_result["event"])
    h["action"].append(_action)

    ss.cumulative_reward += _result["reward"]
    ss.step_idx = _result["next_idx"]
    ss.done = _result["terminated"]
    if ss.done:
        ss.playing = False

# ── Current-step snapshot for rendering ──────────────────────────────────────
h = ss.history
if h["cycle"]:
    cur_cycle = h["cycle"][-1]
    cur_rul = h["rul_actual"][-1]
    cur_qs = h["q_stay"][-1]
    cur_qm = h["q_maintain"][-1]
    cur_event = h["event"][-1]
    cur_action = h["action"][-1]
else:
    _s0 = ss.episode.states[0]
    cur_cycle = int(round(float(_s0[-1])))
    cur_rul = int(ss.episode.rul[0])
    cur_qs, cur_qm = _get_q_values(active_agent, _s0)
    cur_event = "safe_operation"
    cur_action = 0

max_rul = int(ss.episode.rul[0])


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("⚙️ Turbofan Engine — Predictive Maintenance RL Dashboard")
st.markdown(
    f"Engine **#{selected_engine_id}** &nbsp;|&nbsp; "
    f"Policy: `{selected_policy}` &nbsp;|&nbsp; "
    f"Dataset: NASA CMAPSS FD001"
)

# Status badge
st.markdown(_status_html(cur_event, cur_qm, cur_qs, cur_rul), unsafe_allow_html=True)
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

# ── Top metrics row ───────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("🔄 Current Cycle", f"{cur_cycle}")
mc2.metric(
    "⏱️ Actual RUL",
    f"{cur_rul}",
    delta=f"{cur_rul - max_rul} vs start",
    delta_color="normal",
)
mc3.metric("💰 Cumulative Reward", f"{ss.cumulative_reward:+.0f}")
mc4.metric("📋 Steps Taken", f"{len(h['cycle'])}")
event_display = cur_event.replace("_", " ").title()
mc5.metric("📌 Last Event", event_display)

st.markdown("---")

# ── Row 1: sensor feed (left) + Q-value bar (right) ──────────────────────────
r1_left, r1_right = st.columns([2, 1])

with r1_left:
    if h["cycle"]:
        st.plotly_chart(_sensor_chart(h), use_container_width=True, key="sensor_chart")
    else:
        st.info("▶ Press **Play** in the sidebar to start the simulation.")

with r1_right:
    if selected_policy in ("Double DQN", "Dyna-Q"):
        st.plotly_chart(
            _qvalue_chart(cur_qs, cur_qm, selected_policy),
            use_container_width=True,
            key="q_chart",
        )
        if active_agent is None:
            st.caption("⚠️ Untrained agent — actions are random.")
        else:
            action_label = "**MAINTAIN**" if cur_action == 1 else "**Continue**"
            confidence = abs(cur_qm - cur_qs)
            st.caption(
                f"Decision this step: {action_label} &nbsp;|&nbsp; "
                f"Confidence margin: `{confidence:.4f}`"
            )
    elif selected_policy == "Fixed-Interval":
        st.plotly_chart(
            _fi_coverage_chart(cur_cycle, fixed_interval),
            use_container_width=True,
            key="fi_chart",
        )
        cycles_left = max(0, fixed_interval - cur_cycle)
        st.caption(f"Next scheduled maintenance in **{cycles_left}** cycles.")
    else:
        st.markdown("### 🎲 Random Policy")
        st.info("This policy selects Stay / Maintain with equal probability. No Q-values available.")

# ── Row 2: RUL gauge (left) + cumulative cost chart (right) ──────────────────
r2_left, r2_right = st.columns([1, 1])

with r2_left:
    st.plotly_chart(_rul_gauge(cur_rul, max_rul), use_container_width=True, key="rul_gauge")
    # Predicted RUL: simple linear extrapolation from current cycle
    _predicted_rul = max(0, max_rul - (cur_cycle - int(round(float(ss.episode.states[0][-1])))))
    _delta = _predicted_rul - cur_rul
    _delta_str = f"{_delta:+d} cycles vs. linear prediction"
    st.caption(f"📐 Linear-trend prediction: **{_predicted_rul}** cycles &nbsp;|&nbsp; {_delta_str}")

with r2_right:
    if h["reward"]:
        st.plotly_chart(_cumulative_cost_chart(h), use_container_width=True, key="cost_chart")

        # Running cost breakdown
        n_maint = h["event"].count("scheduled_maintenance")
        n_fail = h["event"].count("catastrophic_failure")
        n_safe = h["event"].count("safe_operation")
        total_maint_cost = n_maint * abs(SCHEDULED_MAINTENANCE_COST)
        total_fail_cost = n_fail * abs(CATASTROPHIC_FAILURE_PENALTY)
        total_safe_gain = n_safe * SAFE_CYCLE_REWARD

        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("🟢 Op. Gain", f"+{total_safe_gain:.0f}")
        cb2.metric("🟠 Maint. Cost", f"−{total_maint_cost:.0f}")
        cb3.metric("🔴 Failure Cost", f"−{total_fail_cost:.0f}")
    else:
        st.info("Cost history will appear after the first simulation step.")

# ── Done message ──────────────────────────────────────────────────────────────
if ss.done and h["event"]:
    final_event = h["event"][-1]
    if final_event == "catastrophic_failure":
        st.error(
            "🔴 **Simulation complete: Catastrophic Failure.**  \n"
            f"Engine #{selected_engine_id} was run to destruction at cycle {cur_cycle}.  \n"
            f"Total reward: **{ss.cumulative_reward:+.0f}**"
        )
    elif final_event == "scheduled_maintenance":
        st.success(
            "🟠 **Simulation complete: Scheduled Maintenance.**  \n"
            f"Engine #{selected_engine_id} was retired at cycle {cur_cycle} with RUL = {cur_rul}.  \n"
            f"Total reward: **{ss.cumulative_reward:+.0f}**"
        )
    elif final_event == "completed":
        st.info(
            "✅ **Simulation complete: Full run recorded.**  \n"
            f"Engine #{selected_engine_id} completed all {len(h['cycle'])} recorded cycles.  \n"
            f"Total reward: **{ss.cumulative_reward:+.0f}**"
        )
    st.caption("Press **↺ Reset** in the sidebar to run again.")


# ═══════════════════════════════════════════════════════════════════════════════
#   TECHNICAL DEEP-DIVE — expandable
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📐 Technical Deep-Dive — Bellman Equations & Architecture", expanded=False):
    t_ddqn, t_dyna, t_per, t_env = st.tabs(
        ["Double DQN", "Dyna-Q Planning", "Prioritised Replay (PER)", "Environment & State"]
    )

    # ── Double DQN ──────────────────────────────────────────────────────────

    with t_ddqn:
        st.markdown("### Double DQN — Decoupled Action Selection & Evaluation")
        st.markdown(
            "Standard DQN suffers from **maximisation bias**: the same network selects "
            "_and_ evaluates the next best action, systematically overestimating Q-values. "
            "Double DQN uses the **online network** ($\\theta$) to choose the action, "
            "and the frozen **target network** ($\\theta'$) to score it:"
        )
        st.latex(r"""
            Y_t^{\text{DoubleDQN}} = R_{t+1} + \gamma \;
            \underbrace{
                Q\!\left(S_{t+1},\;
                    \argmax_{a}\, Q(S_{t+1}, a;\;\theta_{\text{online}});
                \;\theta'_{\text{target}}\right)
            }_{\text{target network evaluates online-selected action}}
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            | Symbol | Value | Meaning |
            |--------|-------|---------|
            | $\\gamma$ | 0.99 | Discount factor |
            | Batch size | 256 | RTX 3060 Ti optimised |
            | LR | 3×10⁻⁴ → 1.5×10⁻⁵ | Linear decay |
            | Target sync | every 100 steps | Hard update |
            | $\\text{max\\_grad\\_norm}$ | 10.0 | Prevents −100 gradient explosion |
            """)
        with col2:
            st.markdown("""
            **Network architecture (both agents):**
            ```
            Input  →  Linear(22, 128)  →  ReLU
                   →  Linear(128, 128) →  ReLU
                   →  Linear(128, 2)   →  Q[Stay], Q[Maintain]
            ```
            **Loss (IS-weighted MSE):**
            """)
            st.latex(r"\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} w_i \cdot (Y_i - Q(s_i, a_i;\,\theta))^2")

    # ── Dyna-Q ──────────────────────────────────────────────────────────────

    with t_dyna:
        st.markdown("### Dyna-Q — Model-Based Planning Inside Model-Free Learning")
        st.markdown(
            "Dyna-Q augments every real step with **N = 50 simulated planning steps** "
            "using a learned _World Model_ — a small MLP that predicts state transitions."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**World Model — Residual Prediction**")
            st.latex(r"""
                \hat{\Delta s} = f_\phi(s_t, a_t), \qquad
                \hat{s}_{t+1} = s_t + \hat{\Delta s}
            """)
            st.latex(r"""
                \mathcal{L}_{\text{world}} = \text{MSE}(\hat{\Delta s},\; s' - s)
            """)
            st.markdown(
                "Predicting the **residual** $\\Delta s$ equalises gradient magnitudes: "
                "sensors are scaled $[0,1]$ while the raw cycle count runs 1–300+. "
                "Absolute-state MSE would let the cycle term dominate by $10^4\\times$."
            )
        with c2:
            st.markdown("**Per-step Dyna-Q loop**")
            st.code(
                "for each real interaction:\n"
                "  1. Buffer.add(s, a, r, s')           ← store real experience\n"
                "  2. World model update  (MSE on Δs)   ← improve predictions\n"
                "  3. Q-network update    (real batch)   ← model-free update\n"
                "  4. repeat N=50 times:\n"
                "       sample s from buffer\n"
                "       sample a ~ Uniform{0,1}\n"
                "       ŝ' = s + f_φ(s, a)             ← hallucinate next state\n"
                "       simulate r, done from ŝ'\n"
                "       Q-network update (sim. batch)   ← planning update",
                language="text",
            )

    # ── PER ─────────────────────────────────────────────────────────────────

    with t_per:
        st.markdown("### Prioritised Experience Replay (Schaul et al., 2015)")
        st.markdown(
            "Rare **failure transitions** (RUL < 20 or reward ≤ −50) are 5× more likely "
            "to be sampled, preventing them from being drowned out by safe-operation cycles "
            "which make up > 95% of all transitions."
        )
        st.latex(r"""
            p_i = |\delta_i| + \varepsilon, \qquad
            P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad
            w_i = \left( N \cdot P(i) \right)^{-\beta}
        """)
        st.markdown("""
        | Parameter | Value | Role |
        |-----------|-------|------|
        | $\\alpha = 0.6$ | Prioritisation strength | 0 = uniform, 1 = fully greedy priority |
        | $\\beta$: 0.4 → 1.0 | IS bias correction | Annealed over 100 k steps |
        | Failure boost | ×5 | Initial priority multiplier for near-failure transitions |
        | SumTree | O(log N) | Efficient stratified priority sampling |

        IS weights $w_i$ multiply the per-sample loss before the mean, correcting for
        the sampling bias introduced by non-uniform priority.
        """)

    # ── Environment ─────────────────────────────────────────────────────────

    with t_env:
        st.markdown("### TurboFanEnv — Custom `gymnasium.Env`")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **State space** $s_t \\in \\mathbb{R}^{22}$:
            - 21 sensor readings (30-cycle trailing mean, Min-Max scaled)
            - Current cycle count (raw)

            **Action space** $A = \\{0, 1\\}$:
            - 0 = Continue operation
            - 1 = Trigger scheduled maintenance

            **Reward function $R(s_t, a_t)$:**
            """)
            st.latex(r"""
                R =
                \begin{cases}
                    +1    & \text{if } a=0 \text{ and } \text{RUL} > 0 \\
                    -20   & \text{if } a=1 \text{ (scheduled maintenance)} \\
                    -100  & \text{if } a=0 \text{ and } \text{RUL} = 0 \text{ (failure)}
                \end{cases}
            """)
        with c2:
            st.markdown("""
            **Sliding-window pre-processing:**
            - Window size: 30 cycles
            - Aggregation: trailing mean (compresses temporal context into a single vector)
            - MinMaxScaler fitted on training split only (no data leakage)

            **Episode lifecycle:**
            ```
            reset() → pick engine → return s_0
              ↓
            step(a) → compute reward, advance index
              ↓
            terminated when:
              a=1  (maintenance)
              RUL=0 (failure)
              index exhausted (natural end)
            ```
            """)


# ═══════════════════════════════════════════════════════════════════════════════
#   AUTO-ADVANCE — must be the very last block
# ═══════════════════════════════════════════════════════════════════════════════

if ss.playing and not ss.done:
    time.sleep(speed)
    st.rerun()
