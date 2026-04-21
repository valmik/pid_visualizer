import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="2nd-Order System Explorer", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e0e4ef;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.5px;
}
.stSlider > div[data-baseweb="slider"] > div {
    background: #1e2230;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stNumberInput input {
    font-family: 'IBM Plex Mono', monospace;
    background: #1a1d27 !important;
    color: #a8f0c6 !important;
    border: 1px solid #2e3347 !important;
    border-radius: 4px;
}
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    background: #1a1d27;
    color: #6ec6ff;
    border: 1px solid #2e3347;
    border-radius: 4px;
    width: 100%;
    padding: 0.35rem 0.5rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #252a3a;
    border-color: #6ec6ff;
    color: #ffffff;
}
.col-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6ec6ff;
    padding: 0.2rem 0;
    border-bottom: 1px solid #2e3347;
    margin-bottom: 0.8rem;
}
.section-divider {
    border: none;
    border-top: 1px solid #1e2230;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SecondOrderSystem class
# ─────────────────────────────────────────────
class SecondOrderSystem:
    def __init__(self, zeta=None, omega_n=None, m=None, b=None, k=None,
                 kp=0.0, ki=0.0, kd=0.0, disturbance=0.0):
        has_zw  = zeta    is not None and omega_n is not None
        has_mbk = m       is not None and b       is not None and k is not None

        self.kp = kp; self.ki = ki; self.kd = kd
        self.disturbance = disturbance

        if has_zw and not has_mbk:
            self.zeta = zeta; self.omega_n = omega_n; self.m = 1.0
            self.update_bk()
        elif has_mbk and not has_zw:
            self.m = m; self.b = b; self.k = k
            self.update_zeta_omega_n()
        else:
            self.m = 1.0; self.k = 1.0; self.b = 0.0
            self.update_zeta_omega_n()

    def gen_zeta(self):    return self.b / (2 * self.m * self.omega_n) if self.omega_n > 0 else 0.0
    def gen_omega_n(self): return np.sqrt(max(self.k / self.m, 0.0))
    def gen_b(self):       return 2 * self.zeta * self.m * self.omega_n
    def gen_k(self):       return self.m * self.omega_n**2

    def generate_system(self):
        A = np.array([[0, 1], [-self.k/self.m, -self.b/self.m]])
        B = np.array([0, 1/self.m])
        C = np.array([1, 0]); D = np.array([0])
        self.plant_sys = control.StateSpace(A, B, C, D)
        tau = 0.01*self.kd
        self.controller_sys  = control.TransferFunction([self.kd, self.kp, self.ki], [tau, 1, 0])
        self.closed_loop_sys = control.feedback(self.plant_sys, self.controller_sys)

    def update_zeta_omega_n(self):
        self.omega_n = self.gen_omega_n()
        self.zeta    = self.gen_zeta()
        self.generate_system()

    def update_bk(self):
        self.b = self.gen_b(); self.k = self.gen_k()
        self.generate_system()

    def plot_response(self, T=20):
        time = np.linspace(0, T, 2000)
        dist = self.disturbance * np.ones(len(time))
        initial_state=[0]*self.closed_loop_sys.nstates
        initial_state[0] = 1
        resp = control.forced_response(self.closed_loop_sys, T=time, U=dist, X0=initial_state)
        return resp.time, resp.outputs


def transient_metrics(t, y, settling_rel=0.02):
    """Rise time (10–90%), settling time (±2% of total change), overshoot, min peak."""
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if len(y) < 10:
        return None
    y0 = float(y[0])
    tail_n = max(len(y) // 10, 30)
    y_ss = float(np.mean(y[-tail_n:]))
    dy = y_ss - y0
    tol = 1e-9 * max(1.0, abs(y0), abs(y_ss))
    dy_mag = max(abs(dy), tol)

    band = settling_rel * dy_mag if abs(dy) >= tol else settling_rel * max(abs(y_ss), 1e-12)
    within = np.abs(y - y_ss) <= band
    settling_time = np.nan
    for k in range(len(y)):
        if np.all(within[k:]):
            settling_time = float(t[k])
            break

    rise_time = np.nan
    if abs(dy) >= tol:
        y10, y90 = y0 + 0.1 * dy, y0 + 0.9 * dy
        if dy > 0:
            m10 = np.where(y >= y10)[0]
            m90 = np.where(y >= y90)[0]
        else:
            m10 = np.where(y <= y10)[0]
            m90 = np.where(y <= y90)[0]
        if len(m10) and len(m90) and m90[0] > m10[0]:
            rise_time = float(t[m90[0]] - t[m10[0]])

    imax = int(np.argmax(y))
    imin = int(np.argmin(y))
    y_max, y_min = float(y[imax]), float(y[imin])
    t_max, t_min = float(t[imax]), float(t[imin])

    if abs(dy) >= tol:
        if dy > 0:
            overshoot_pct = 100.0 * max(0.0, y_max - y_ss) / abs(dy)
        else:
            overshoot_pct = 100.0 * max(0.0, y_ss - y_min) / abs(dy)
    else:
        overshoot_pct = 0.0

    return dict(
        y_ss=y_ss,
        y0=y0,
        rise_time=rise_time,
        settling_time=settling_time,
        overshoot_pct=overshoot_pct,
        y_max=y_max,
        y_min=y_min,
        t_max=t_max,
        t_min=t_min,
        imax=imax,
        imin=imin,
    )


def _fmt_metric(x, fmt=".3f"):
    if x is None:
        return "—"
    try:
        xf = float(x)
        if np.isnan(xf):
            return "—"
        return f"{xf:{fmt}}"
    except (TypeError, ValueError):
        return "—"


# ─────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────
DEFAULTS = dict(m=1.0, b=0.5, k=1.0, zeta=0.25, omega_n=1.0,
                kp=0.0, ki=0.0, kd=0.0, disturbance=0.0)

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Bounds shared by sliders, number inputs, and sync clipping
M_MIN, M_MAX = 0.1, 5.0
B_MIN, B_MAX = 0.0, 25.0
K_MIN, K_MAX = 0.01, 20.0
ZETA_MIN, ZETA_MAX = 0.0, 5.0
OMEGA_MIN, OMEGA_MAX = 0.01, 10.0

# Slider vs number-input column weights (2.4:1 ≈ 20% narrower slider than 3:1)
SLIDER_NB_SPLIT = [2.4, 1]
# Control column / plot column (~20% narrower control strip vs former 1:1.55)
CTRL_PLOT_SPLIT = [0.8, 1.55]

# ─────────────────────────────────────────────
# Sync helpers
# ─────────────────────────────────────────────
def _mirror_nb(*keys):
    """Keep number_input keys aligned after programmatic session_state updates."""
    for k in keys:
        st.session_state[f"_nb_{k}"] = st.session_state[k]


def sync_from_mbk():
    m = st.session_state.m
    b = st.session_state.b
    k = st.session_state.k
    omega_n = np.sqrt(max(k / m, 1e-9))
    zeta    = b / (2 * m * omega_n)
    st.session_state.omega_n = float(np.clip(omega_n, OMEGA_MIN, OMEGA_MAX))
    st.session_state.zeta    = float(np.clip(zeta,    ZETA_MIN, ZETA_MAX))
    _mirror_nb("zeta", "omega_n")


def sync_from_zeta_omega():
    m       = st.session_state.m
    omega_n = st.session_state.omega_n
    zeta    = st.session_state.zeta
    k = m * omega_n**2
    b = 2 * zeta * m * omega_n
    st.session_state.k = float(np.clip(k, K_MIN, K_MAX))
    st.session_state.b = float(np.clip(b, B_MIN, B_MAX))
    _mirror_nb("k", "b")


def sync_mbk_chain():
    """m,b,k → ζ,ωₙ (clip), then ζ,ωₙ → b,k so saturated modal params match plant sliders."""
    sync_from_mbk()
    sync_from_zeta_omega()
    _mirror_nb("m", "b", "k", "zeta", "omega_n")


def _randomize_mbk():
    st.session_state.m = float(np.random.uniform(0.2, M_MAX))
    st.session_state.b = float(np.random.uniform(0.0, min(5.0, B_MAX)))
    st.session_state.k = float(np.random.uniform(0.1, K_MAX))
    sync_mbk_chain()


def _randomize_pid():
    st.session_state.kp = float(np.random.uniform(0.0, 10.0))
    st.session_state.kd = float(np.random.uniform(0.0, 5.0))
    st.session_state.ki = float(np.random.uniform(0.0, 5.0))
    _mirror_nb("kp", "kd", "ki")


# ─────────────────────────────────────────────
# Helper: labelled slider + number_input pair
# ─────────────────────────────────────────────
def param_widget(label, key, min_val, max_val, step, fmt="%.3f", on_change=None):
    """Render a slider with an editable number box below it."""
    def _on_slider_change():
        # Keep number_input in sync when the slider moves (separate widget keys).
        st.session_state[f"_nb_{key}"] = st.session_state[key]
        if on_change:
            on_change()

    st.markdown(f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#a8f0c6'>{label}</span>",
                unsafe_allow_html=True)
    col_sl, col_nb = st.columns(SLIDER_NB_SPLIT)
    nb_key = f"_nb_{key}"
    if nb_key not in st.session_state:
        st.session_state[nb_key] = float(st.session_state[key])

    with col_sl:
        st.slider(label, min_val, max_val, key=key,
                  step=step, label_visibility="collapsed",
                  on_change=_on_slider_change)
    with col_nb:
        # Do not pass value= with key= — Streamlit treats that as conflicting with Session State.
        st.number_input(label, min_value=float(min_val), max_value=float(max_val),
                        step=float(step), format=fmt, key=nb_key,
                        label_visibility="collapsed",
                        on_change=lambda k=key, on_c=on_change: _nb_sync(k, on_c))


def _nb_sync(key, on_change_fn):
    st.session_state[key] = st.session_state[f"_nb_{key}"]
    if on_change_fn:
        on_change_fn()


# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.markdown("## 2nd-Order System Explorer")
st.markdown("<p style='color:#5a6080;font-size:0.85rem;margin-top:-0.5rem'>Closed-loop PID response of a mass-spring-damper</p>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Left: m·b·k + ζ·ωₙ  |  Right (wide): plot + disturbance
# ─────────────────────────────────────────────
col_left, col_right = st.columns(CTRL_PLOT_SPLIT, gap="large")

with col_left:
    row_plant_l, row_plant_r = st.columns([2.2, 1], vertical_alignment="center")
    with row_plant_l:
        st.markdown('<div class="col-header" style="margin-bottom:0.5rem">Plant  —  m · b · k</div>', unsafe_allow_html=True)
    with row_plant_r:
        st.button("⟳  Randomise  m · b · k", key="rand_mbk", on_click=_randomize_mbk, use_container_width=True)

    param_widget("m  (mass)",      "m",  M_MIN, M_MAX, 0.05, on_change=sync_mbk_chain)
    param_widget("b  (damping)",   "b",  B_MIN, B_MAX, 0.05, on_change=sync_mbk_chain)
    param_widget("k  (stiffness)", "k",  K_MIN, K_MAX, 0.05, on_change=sync_mbk_chain)

    st.markdown('<div class="col-header" style="margin-top:1rem">Modal  —  ζ · ωₙ</div>', unsafe_allow_html=True)
    param_widget("ζ  (damping ratio)", "zeta",    ZETA_MIN, ZETA_MAX, 0.01, on_change=sync_from_zeta_omega)
    param_widget("ωₙ (natural freq)",  "omega_n", OMEGA_MIN, OMEGA_MAX, 0.05, on_change=sync_from_zeta_omega)

with col_right:
    st.markdown('<div class="col-header">Response</div>', unsafe_allow_html=True)
    # ─────────────────────────────────────────
    # Build system & plot
    # ─────────────────────────────────────────
    try:
        sys = SecondOrderSystem(
            m=st.session_state.m,
            b=st.session_state.b,
            k=st.session_state.k,
            kp=st.session_state.kp,
            ki=st.session_state.ki,
            kd=st.session_state.kd,
            disturbance=st.session_state.disturbance,
        )
        t, y = sys.plot_response(T=20)

        plt.rcParams.update({
            "figure.facecolor":  "#0d0f14",
            "axes.facecolor":    "#12151e",
            "axes.edgecolor":    "#1e2230",
            "axes.labelcolor":   "#7a8099",
            "xtick.color":       "#3a3f55",
            "ytick.color":       "#3a3f55",
            "grid.color":        "#1a1d2a",
            "grid.linewidth":    0.8,
            "text.color":        "#e0e4ef",
            "font.family":       "monospace",
        })

        fig, ax = plt.subplots(figsize=(10, 4.2))

        ax.fill_between(t, y, 0, where=(y >= 0), alpha=0.20, color="#6ec6ff")
        ax.fill_between(t, y, 0, where=(y <  0), alpha=0.20, color="#ff6e8a")

        ax.axhline(0, color="#2e3347", linewidth=0.8, linestyle="--")

        ax.plot(t, y, color="#6ec6ff", linewidth=1.8, label="y(t)")

        m = transient_metrics(t, y) if len(y) else None
        if m and len(y) > 1:
            if m["imax"] != m["imin"]:
                ax.plot(m["t_max"], m["y_max"], "o", color="#ffb86c", markersize=5, zorder=5)
                ax.annotate(f"  max {m['y_max']:.3f}",
                            xy=(m["t_max"], m["y_max"]),
                            color="#ffb86c", fontsize=8, va="bottom")
                ax.plot(m["t_min"], m["y_min"], "o", color="#ff6e8a", markersize=5, zorder=5)
                ax.annotate(f"  min {m['y_min']:.3f}",
                            xy=(m["t_min"], m["y_min"]),
                            color="#ff6e8a", fontsize=8, va="top")
            else:
                ax.plot(m["t_max"], m["y_max"], "o", color="#a8f0c6", markersize=5, zorder=5)
                ax.annotate(f"  peak {m['y_max']:.3f}",
                            xy=(m["t_max"], m["y_max"]),
                            color="#a8f0c6", fontsize=8, va="bottom")

            stats = (
                f"t_r = {_fmt_metric(m['rise_time'])} s\n"
                f"t_s = {_fmt_metric(m['settling_time'])} s\n"
                f"overshoot = {_fmt_metric(m['overshoot_pct'], '.2f')} %\n"
                f"neg peak = {_fmt_metric(m['y_min'], '.4f')} @ {_fmt_metric(m['t_min'])} s"
            )
            ax.text(
                0.98, 0.97, stats,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                color="#c8cdd8",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#12151e",
                          edgecolor="#2e3347", alpha=0.94),
            )

        ax.set_xlabel("time  (s)", fontsize=9)
        ax.set_ylabel("output  y(t)", fontsize=9)
        ax.set_title(
            f"ζ={st.session_state.zeta:.3f}   ωₙ={st.session_state.omega_n:.3f}   "
            f"kp={st.session_state.kp:.2f}  kd={st.session_state.kd:.2f}  ki={st.session_state.ki:.2f}   "
            f"d={st.session_state.disturbance:.2f}",
            fontsize=9, color="#5a6080", pad=8
        )
        ax.grid(True, which="both")
        ax.set_xlim(t[0], t[-1])
        fig.tight_layout(pad=1.0)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"Simulation error: {e}")

    st.markdown('<div style="margin-top:0.85rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="col-header" style="margin-bottom:0.5rem">Disturbance input</div>', unsafe_allow_html=True)
    param_widget("Disturbance", "disturbance", -5.0, 5.0, 0.05)

# ─────────────────────────────────────────────
# Controller — full width below the two columns above
# ─────────────────────────────────────────────
st.markdown('<div style="margin-top:0.85rem"></div>', unsafe_allow_html=True)
row_ctl_l, row_ctl_r = st.columns([2.2, 1], vertical_alignment="center")
with row_ctl_l:
    st.markdown('<div class="col-header" style="margin-bottom:0.5rem">Controller  —  kp · kd · ki</div>', unsafe_allow_html=True)
with row_ctl_r:
    st.button("⟳  Randomise  kp · kd · ki", key="rand_pid", on_click=_randomize_pid, use_container_width=True)

col_kp, col_kd, col_ki = st.columns(3, gap="medium")
with col_kp:
    param_widget("kp  (proportional)", "kp", 0.0, 50.0, 0.1)
with col_kd:
    param_widget("kd  (derivative)",   "kd", 0.0, 20.0, 0.05)
with col_ki:
    param_widget("ki  (integral)",     "ki", 0.0, 20.0, 0.05)