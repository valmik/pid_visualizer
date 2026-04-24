import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control
import streamlit as st
from fractions import Fraction

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Higher-Order PID Explorer", layout="wide")

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
.stTextInput input {
    font-family: 'IBM Plex Mono', monospace;
    background: #1a1d27 !important;
    color: #a8f0c6 !important;
    border: 1px solid #2e3347 !important;
    border-radius: 4px;
}
.stSelectbox > div[data-baseweb="select"] > div {
    background: #1a1d27 !important;
    border: 1px solid #2e3347 !important;
    font-family: 'IBM Plex Mono', monospace;
    color: #a8f0c6 !important;
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
.eq-box {
    background: #12151e;
    border: 1px solid #2e3347;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #c8d0e8;
    margin-bottom: 0.7rem;
    line-height: 1.7;
}
.eq-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6ec6ff;
    margin-bottom: 0.3rem;
}
.eigenval-error {
    color: #ff6e8a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}
.info-badge {
    display: inline-block;
    background: #1a1d27;
    border: 1px solid #2e3347;
    border-radius: 3px;
    padding: 0.1rem 0.45rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #ffb86c;
    margin: 0.1rem 0.2rem 0.1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
SLIDER_NB_SPLIT = [2.4, 1]
CTRL_PLOT_SPLIT = [0.8, 1.55]


def parse_eigenvalue(s: str):
    """
    Parse a string into a complex number.
    Accepts: -2, -1+2j, -1-2j, 2j, -0.5+1.5j, etc.
    Returns complex or raises ValueError.
    """
    s = s.strip().replace(" ", "")
    if not s:
        raise ValueError("empty")
    # Python can eval complex literals safely if we restrict chars
    allowed = set("0123456789.+-*ej()") | set("ij")
    # replace 'i' with 'j' for convenience
    s2 = s.replace("i", "j")
    if not all(c in "0123456789.+-jJeE()" for c in s2):
        raise ValueError(f"invalid characters in '{s}'")
    try:
        val = complex(s2)
    except ValueError:
        # Try eval as last resort (still restricted)
        try:
            val = complex(eval(s2, {"__builtins__": {}}, {}))
        except Exception:
            raise ValueError(f"cannot parse '{s}'")
    return val


def eigenvalues_to_poly(eigs):
    """
    Given a list of eigenvalues (complex), return real coefficients of the
    characteristic polynomial (highest degree first), e.g. [1, a, b, c, ...].
    Complex eigenvalues must come in conjugate pairs; we enforce this by
    matching pairs automatically.
    """
    poly = np.array([1.0])
    processed = [False] * len(eigs)
    for i, e in enumerate(eigs):
        if processed[i]:
            continue
        if abs(e.imag) < 1e-10:
            # Real root: (s - e)
            poly = np.polymul(poly, [1.0, -e.real])
            processed[i] = True
        else:
            # Complex root — find conjugate
            conj_found = False
            for j in range(i + 1, len(eigs)):
                if not processed[j] and abs(eigs[j] - e.conjugate()) < 1e-8:
                    # Quadratic factor: (s-e)(s-e*) = s^2 - 2Re(e)*s + |e|^2
                    poly = np.polymul(poly, [1.0, -2 * e.real, abs(e) ** 2])
                    processed[i] = True
                    processed[j] = True
                    conj_found = True
                    break
            if not conj_found:
                raise ValueError(
                    f"Eigenvalue {e} has no conjugate pair — "
                    "complex eigenvalues must come in conjugate pairs."
                )
    return poly


def poly_to_ss(char_poly):
    """
    Given characteristic polynomial coefficients [1, a1, a2, ..., an],
    build a controllable canonical form state-space system with unit gain
    (output = first state).
    """
    n = len(char_poly) - 1  # order
    a = char_poly[1:]  # a1 … an (exclude leading 1)

    # Companion / controllable canonical form
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -a[::-1]

    B = np.zeros((n, 1))
    B[-1, 0] = 1.0

    C = np.zeros((1, n))
    C[0, 0] = 1.0

    D = np.zeros((1, 1))
    return control.StateSpace(A, B, C, D)


def build_and_simulate(plant_ss, kp, ki, kd, T=20, n_pts=2000):
    """
    Build a PID closed-loop around plant_ss and simulate the initial-condition
    response with plant output y(0)=1 and all controller states starting at 0.

    We find which closed-loop state corresponds to the plant output by reading
    the closed-loop C matrix directly (argmax of |C|). This is robust to
    whatever state ordering control.feedback() chooses internally.
    """
    tau = max(0.01 * kd, 1e-4)
    if abs(ki) < 1e-9:
        # Avoid a spurious free integrator when ki=0 by cancelling the s factor
        controller = control.TransferFunction([kd, kp], [tau, 1])
    else:
        controller = control.TransferFunction([kd, kp, ki], [tau, 1, 0])

    controller_ss = control.tf2ss(controller)
    cl = control.feedback(plant_ss * controller_ss)

    # Set x0 so that C @ x0 = 1 (plant output starts at 1).
    # Our plant has C=[1,0,...,0] with exactly one nonzero entry, so argmax(|C|)
    # finds the correct state regardless of how feedback() orders the states.
    C = np.array(cl.C).flatten()
    idx = int(np.argmax(np.abs(C)))
    x0 = np.zeros(cl.nstates)
    x0[idx] = 1.0 / C[idx]

    cl_eigs = np.linalg.eigvals(np.array(cl.A))
    time = np.linspace(0, T, n_pts)
    resp = control.forced_response(cl, T=time, U=np.zeros(n_pts), X0=x0)
    return resp.time, resp.outputs.flatten(), cl_eigs


def dominant_eigs(eigs):
    """
    Always return exactly 2 eigenvalues (a valid real or conjugate-paired set)
    for the 2nd-order dominant approximation.

    Rules (modes sorted by |Re(λ)| ascending = slowest first):
      1. Dominant mode is a complex pair           → use that pair
      2. Dominant mode is real, next is also real  → use both reals
      3. Dominant mode is real, next is complex    → use dominant real +
           the real part of the complex mode as a second real eigenvalue,
           giving a 2nd-order approximation that captures the decay rate of
           the next mode without its oscillation
      4. Only one mode (1st-order system)          → repeat the single eigenvalue
           to form a degenerate (critically damped) 2nd-order approximation
    """
    # Group into modes
    used = [False] * len(eigs)
    modes = []
    for i, e in enumerate(eigs):
        if used[i]:
            continue
        if abs(e.imag) < 1e-8:
            modes.append([e])
            used[i] = True
        else:
            conj_idx = None
            for j in range(i + 1, len(eigs)):
                if not used[j] and abs(eigs[j] - e.conjugate()) < 1e-6:
                    conj_idx = j
                    break
            if conj_idx is not None:
                modes.append([e, eigs[conj_idx]])
                used[i] = True
                used[conj_idx] = True
            else:
                modes.append([complex(e.real, 0)])
                used[i] = True

    modes.sort(key=lambda m: abs(m[0].real))

    if not modes:
        return [complex(-1, 0), complex(-1, 0)]

    d = modes[0]

    # Rule 1: dominant is complex pair → done
    if len(d) == 2:
        return d

    # Only one mode total (1st-order system) → degenerate 2nd-order
    if len(modes) == 1:
        return [d[0], d[0]]

    nxt = modes[1]
    # Rule 2: next mode is also real → use both
    if len(nxt) == 1:
        return [d[0], nxt[0]]

    # Rule 3: dominant real + next mode is complex → use real part of complex as 2nd eig
    real_approx = complex(nxt[0].real, 0)
    return [d[0], real_approx]


def poly_str(coeffs, var="s"):
    """Format polynomial coefficients into a human-readable string."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        c_r = round(c.real if isinstance(c, complex) else c, 4)
        if abs(c_r) < 1e-9:
            continue
        c_str = f"{abs(c_r):.4g}"
        if power == 0:
            term = c_str
        elif power == 1:
            term = f"{c_str}{var}"
        else:
            term = f"{c_str}{var}^{power}"
        sign = "+" if c_r >= 0 else "−"
        terms.append((sign, term))

    if not terms:
        return "0"
    result = terms[0][1] if terms[0][0] == "+" else f"−{terms[0][1]}"
    for sign, term in terms[1:]:
        result += f"  {sign}  {term}"
    return result


def ode_string_from_poly(coeffs):
    """
    Given char poly [1, a1, a2, ..., an], build ODE string:
    y^(n) + a1·y^(n-1) + ... + an·y = u
    """
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        c_r = round(c.real if isinstance(c, complex) else c, 4)
        if abs(c_r) < 1e-9:
            continue
        c_str = f"{abs(c_r):.4g}"
        if power == 0:
            deriv = "y"
        elif power == 1:
            deriv = "ẏ"
        elif power == 2:
            deriv = "ÿ"
        else:
            deriv = f"y⁽{power}⁾"

        coeff_part = "" if abs(c_r - 1.0) < 1e-9 else f"{c_str}·"
        sign = "+" if c_r >= 0 else "−"
        terms.append((sign, f"{coeff_part}{deriv}"))

    if not terms:
        return "0 = u"
    result = terms[0][1] if terms[0][0] == "+" else f"−{terms[0][1]}"
    for sign, term in terms[1:]:
        result += f"  {sign}  {term}"
    return result + "  =  u(t)"


def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}"


def transient_metrics(t, y, settling_rel=0.02):
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
    band = settling_rel * dy_mag
    within = np.abs(y - y_ss) <= band
    settling_time = np.nan
    for k in range(len(y)):
        if np.all(within[k:]):
            settling_time = float(t[k])
            break
    rise_time = np.nan
    if abs(dy) >= tol:
        y10, y90 = y0 + 0.1 * dy, y0 + 0.9 * dy
        m10 = np.where(y >= y10)[0] if dy > 0 else np.where(y <= y10)[0]
        m90 = np.where(y >= y90)[0] if dy > 0 else np.where(y <= y90)[0]
        if len(m10) and len(m90) and m90[0] > m10[0]:
            rise_time = float(t[m90[0]] - t[m10[0]])
    imax = int(np.argmax(y))
    imin = int(np.argmin(y))
    overshoot = 100.0 * max(0.0, y[imax] - y_ss) / dy_mag if dy > 0 else 0.0
    return dict(rise_time=rise_time, settling_time=settling_time,
                overshoot_pct=overshoot,
                y_max=float(y[imax]), y_min=float(y[imin]),
                t_max=float(t[imax]), t_min=float(t[imin]),
                imax=imax, imin=imin, y_ss=y_ss)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "ho_order" not in st.session_state:
    st.session_state.ho_order = 3

if "ho_eig_strings" not in st.session_state:
    st.session_state.ho_eig_strings = ["-1", "-2+1j", "-2-1j"]

if "ho_kp" not in st.session_state:
    st.session_state.ho_kp = 0.0
if "ho_ki" not in st.session_state:
    st.session_state.ho_ki = 0.0
if "ho_kd" not in st.session_state:
    st.session_state.ho_kd = 0.0


def _default_eigs_for_order(n):
    """Return sensible default eigenvalue strings for order n."""
    presets = {
        1: ["-1"],
        2: ["-1+1j", "-1-1j"],
        3: ["-1", "-2+1j", "-2-1j"],
        4: ["-1+1j", "-1-1j", "-3+0.5j", "-3-0.5j"],
        5: ["-0.5", "-1+1j", "-1-1j", "-3+0.5j", "-3-0.5j"],
    }
    return presets.get(n, ["-1"] * n)


def _randomize_eigs():
    n = st.session_state.ho_order
    eigs = []
    i = 0
    while i < n:
        remaining = n - i
        if remaining == 1:
            # Must be real
            real = -np.random.uniform(0.3, 4.0)
            eigs.append(f"{real:.3f}")
            i += 1
        else:
            # 50/50 real or complex pair
            if np.random.rand() > 0.5:
                real = -np.random.uniform(0.3, 4.0)
                eigs.append(f"{real:.3f}")
                i += 1
            else:
                real = -np.random.uniform(0.3, 4.0)
                imag = np.random.uniform(0.3, 3.0)
                eigs.append(f"{real:.3f}+{imag:.3f}j")
                eigs.append(f"{real:.3f}-{imag:.3f}j")
                i += 2
    eigs = eigs[:n]
    st.session_state.ho_eig_strings = eigs
    # Also write directly into each text_input's session-state key so the
    # widgets re-render with the new values on the next run.
    for idx, val in enumerate(eigs):
        st.session_state[f"ho_eig_{idx}"] = val


def param_widget_ho(label, key, min_val, max_val, step, fmt="%.3f"):
    """Slider + number input widget reusing the same style as page 1."""
    def _on_slider():
        st.session_state[f"_nb_{key}"] = st.session_state[key]

    st.markdown(
        f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#a8f0c6'>{label}</span>",
        unsafe_allow_html=True,
    )
    col_sl, col_nb = st.columns(SLIDER_NB_SPLIT)
    nb_key = f"_nb_{key}"
    if nb_key not in st.session_state:
        st.session_state[nb_key] = float(st.session_state[key])

    with col_sl:
        st.slider(label, min_val, max_val, key=key,
                  step=step, label_visibility="collapsed",
                  on_change=_on_slider)
    with col_nb:
        st.number_input(label, min_value=float(min_val), max_value=float(max_val),
                        step=float(step), format=fmt, key=nb_key,
                        label_visibility="collapsed",
                        on_change=lambda k=key: _nb_sync_ho(k))


def _nb_sync_ho(key):
    st.session_state[key] = st.session_state[f"_nb_{key}"]


# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.markdown("## Higher-Order PID Explorer")
st.markdown(
    "<p style='color:#5a6080;font-size:0.85rem;margin-top:-0.5rem'>"
    "Compare PID response of an n-th order system against its dominant 2nd-order approximation<br>"
    "By Valmik Prabhu (and Claude)</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

col_left, col_right = st.columns(CTRL_PLOT_SPLIT, gap="large")

# ─────────────────────────────────────────────
# LEFT COLUMN
# ─────────────────────────────────────────────
with col_left:

    # ── Order selector ──────────────────────
    st.markdown('<div class="col-header">System Order</div>', unsafe_allow_html=True)

    order_options = [1, 2, 3, 4, 5]
    prev_order = st.session_state.ho_order
    selected_order = st.selectbox(
        "System order",
        options=order_options,
        index=order_options.index(st.session_state.ho_order),
        label_visibility="collapsed",
    )

    # When order changes, reset eigenvalue strings
    if selected_order != prev_order:
        st.session_state.ho_order = selected_order
        defaults = _default_eigs_for_order(selected_order)
        st.session_state.ho_eig_strings = defaults
        for idx, val in enumerate(defaults):
            st.session_state[f"ho_eig_{idx}"] = val
        prev_order = selected_order

    # Ensure we have the right number of eig strings
    n_order = selected_order
    eigs_list = st.session_state.ho_eig_strings
    while len(eigs_list) < n_order:
        eigs_list.append("-1")
    eigs_list = eigs_list[:n_order]
    st.session_state.ho_eig_strings = eigs_list

    # ── Eigenvalue inputs ────────────────────
    st.markdown('<div class="col-header" style="margin-top:1rem">Eigenvalues  (λ₁ … λₙ)</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='font-family:IBM Plex Mono;font-size:0.72rem;color:#5a6080'>"
        "Enter real or complex values, e.g.  <b>-2</b>  or  <b>-1+2j</b><br>"
        "Complex eigenvalues must come in conjugate pairs.</span>",
        unsafe_allow_html=True,
    )

    new_eig_strings = []
    parse_errors = []
    for i in range(n_order):
        val = st.text_input(
            f"λ{i+1}",
            value=eigs_list[i],
            key=f"ho_eig_{i}",
            placeholder="-1+2j",
        )
        new_eig_strings.append(val)
        try:
            parse_eigenvalue(val)
        except ValueError as ex:
            parse_errors.append(f"λ{i+1}: {ex}")

    st.session_state.ho_eig_strings = new_eig_strings

    # Randomize button
    st.markdown('<div style="margin-top:0.4rem"></div>', unsafe_allow_html=True)
    st.button("⟳  Randomise eigenvalues", key="rand_eigs", on_click=_randomize_eigs, use_container_width=True)

    if parse_errors:
        for err in parse_errors:
            st.markdown(f'<div class="eigenval-error">⚠ {err}</div>', unsafe_allow_html=True)

    # ── PID gains ───────────────────────────
    st.markdown('<div class="col-header" style="margin-top:1rem">Controller  —  kp · kd · ki</div>', unsafe_allow_html=True)
    param_widget_ho("kp  (proportional)", "ho_kp", 0.0, 50.0, 0.1)
    param_widget_ho("kd  (derivative)",   "ho_kd", 0.0, 20.0, 0.05)
    param_widget_ho("ki  (integral)",     "ho_ki", 0.0, 20.0, 0.05)

# ─────────────────────────────────────────────
# RIGHT COLUMN
# ─────────────────────────────────────────────
with col_right:

    st.markdown('<div class="col-header">System Equations & Response</div>', unsafe_allow_html=True)

    # ── Parse eigenvalues ───────────────────
    eigs_parsed = []
    all_ok = True
    for s in st.session_state.ho_eig_strings:
        try:
            eigs_parsed.append(parse_eigenvalue(s))
        except ValueError:
            all_ok = False
            break

    # Also check conjugate pairing
    conj_error = None
    if all_ok and len(eigs_parsed) == n_order:
        try:
            char_poly_full = eigenvalues_to_poly(eigs_parsed)
        except ValueError as e:
            conj_error = str(e)
            all_ok = False

    if not all_ok or conj_error:
        err_msg = conj_error or "Fix eigenvalue inputs on the left."
        st.error(f"Cannot build system: {err_msg}")
    else:
        # ── Differential equations ──────────
        # Full system ODE
        full_ode = ode_string_from_poly(char_poly_full)

        # Dominant approximation (1st or 2nd order, always valid)
        dom_eigs = dominant_eigs(eigs_parsed)
        char_poly_2nd = eigenvalues_to_poly(dom_eigs)
        dom_order = len(char_poly_2nd) - 1

        ode_2nd = ode_string_from_poly(char_poly_2nd)

        # Format dominant eig display
        dom_eig_strs = []
        for e in dom_eigs:
            if abs(e.imag) < 1e-8:
                dom_eig_strs.append(f"{e.real:.4g}")
            else:
                sign = "+" if e.imag >= 0 else "−"
                dom_eig_strs.append(f"{e.real:.4g} {sign} {abs(e.imag):.4g}j")

        # Display equations
        st.markdown(f'<div class="eq-label">Full {n_order}-th Order System (open loop)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="eq-box">{full_ode}</div>', unsafe_allow_html=True)

        dom_label_eigs = "  \u00b7  ".join(dom_eig_strs)
        approx_order_label = "1st" if dom_order == 1 else "2nd"
        st.markdown(
            f'<div class="eq-label">Dominant {approx_order_label}-Order Approximation'
            f'  <span style="color:#5a6080;font-size:0.65rem">'
            f'dominant eigenvalue(s): {dom_label_eigs}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="eq-box">{ode_2nd}</div>', unsafe_allow_html=True)

        # ── Simulate ────────────────────────
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

        try:
            # Full system
            plant_full = poly_to_ss(char_poly_full)
            t_full, y_full, cl_eigs_full = build_and_simulate(
                plant_full,
                st.session_state.ho_kp, st.session_state.ho_ki, st.session_state.ho_kd,
            )

            # Dominant-mode approx
            plant_2nd = poly_to_ss(char_poly_2nd)
            t_2nd, y_2nd, cl_eigs_2nd = build_and_simulate(
                plant_2nd,
                st.session_state.ho_kp, st.session_state.ho_ki, st.session_state.ho_kd,
            )

            fig, ax = plt.subplots(figsize=(10, 4.6))

            # Shaded fills
            ax.fill_between(t_full, y_full, 0, alpha=0.12, color="#6ec6ff")
            ax.fill_between(t_2nd,  y_2nd,  0, alpha=0.10, color="#a8f0c6")

            ax.axhline(0, color="#2e3347", linewidth=0.8, linestyle="--")

            ax.plot(t_full, y_full, color="#6ec6ff", linewidth=1.8,
                    label=f"Full {n_order}-th order", zorder=4)
            ax.plot(t_2nd,  y_2nd,  color="#a8f0c6", linewidth=1.5,
                    linestyle="--", label=f"{approx_order_label}-order approx", zorder=3)

            # Annotate metrics for full system
            m = transient_metrics(t_full, y_full)
            m2 = transient_metrics(t_2nd, y_2nd)

            def _metric_box(metrics, color, label_prefix, x_pos):
                if metrics is None:
                    return
                stats = (
                    f"{label_prefix}\n"
                    f"t_r  = {_fmt(metrics['rise_time'])} s\n"
                    f"t_s  = {_fmt(metrics['settling_time'])} s\n"
                    f"OS   = {_fmt(metrics['overshoot_pct'])} %"
                )
                ax.text(
                    x_pos, 0.97, stats,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    verticalalignment="top",
                    horizontalalignment="right",
                    color=color,
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#12151e",
                              edgecolor="#2e3347", alpha=0.94),
                )

            _metric_box(m,  "#6ec6ff", f"n={n_order} full sys", 0.98)
            _metric_box(m2, "#a8f0c6", f"{approx_order_label}-order approx", 0.73)

            ax.set_xlabel("time  (s)", fontsize=9)
            ax.set_ylabel("output  y(t)", fontsize=9)
            ax.set_title(
                f"Order {n_order}  ·  kp={st.session_state.ho_kp:.2f}  "
                f"kd={st.session_state.ho_kd:.2f}  ki={st.session_state.ho_ki:.2f}",
                fontsize=9, color="#5a6080", pad=8,
            )
            ax.grid(True, which="both")
            ax.set_xlim(t_full[0], t_full[-1])

            # Legend
            leg = ax.legend(loc="lower right", fontsize=8,
                            facecolor="#12151e", edgecolor="#2e3347",
                            labelcolor="#e0e4ef")

            # Instability warning if any CL eigenvalue has positive real part
            unstable_full = any(e.real > 1e-6 for e in cl_eigs_full)
            unstable_2nd  = any(e.real > 1e-6 for e in cl_eigs_2nd)
            if unstable_full or unstable_2nd:
                ax.text(
                    0.5, 0.93, "that's a nono",
                    transform=ax.transAxes,
                    fontsize=11, fontweight="bold",
                    verticalalignment="top",
                    horizontalalignment="center",
                    color="#ff6e8a",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a0d10",
                              edgecolor="#ff6e8a", alpha=0.95),
                    zorder=10,
                )

            fig.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Open-loop eigenvalue badges ──
            st.markdown('<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)
            badges_html = "<span style='font-family:IBM Plex Mono;font-size:0.7rem;color:#5a6080'>Open-loop λ:  </span>"
            for e in eigs_parsed:
                if abs(e.imag) < 1e-8:
                    label = f"{e.real:.4g}"
                else:
                    sign = "+" if e.imag >= 0 else "−"
                    label = f"{e.real:.4g}{sign}{abs(e.imag):.4g}j"
                color = "#ffb86c" if abs(e.imag) > 1e-8 else "#a8f0c6"
                badges_html += f'<span class="info-badge" style="color:{color}">{label}</span>'
            st.markdown(badges_html, unsafe_allow_html=True)

            # ── Closed-loop eigenvalue badges ──
            st.markdown('<div style="margin-top:0.3rem"></div>', unsafe_allow_html=True)

            def _cl_eig_badges(eigs, sys_label):
                # Filter out the fast tau-artifact poles (|Re| > 500) — those are
                # controller implementation artefacts, not physical CL poles.
                physical = [e for e in eigs if abs(e.real) < 500]
                physical.sort(key=lambda e: e.real)
                html = f"<span style='font-family:IBM Plex Mono;font-size:0.7rem;color:#5a6080'>{sys_label}:  </span>"
                for e in physical:
                    unstable = e.real > 1e-6
                    if abs(e.imag) < 1e-4:
                        label = f"{e.real:.4g}"
                    else:
                        sign = "+" if e.imag >= 0 else "−"
                        label = f"{e.real:.4g}{sign}{abs(e.imag):.4g}j"
                    if unstable:
                        color = "#ff6e8a"
                        label = "⚠ " + label
                    elif abs(e.imag) > 1e-4:
                        color = "#ffb86c"
                    else:
                        color = "#a8f0c6"
                    html += f'<span class="info-badge" style="color:{color}">{label}</span>'
                return html

            st.markdown(_cl_eig_badges(cl_eigs_full, f"CL λ  (n={n_order} full)"), unsafe_allow_html=True)
            st.markdown('<div style="margin-top:0.2rem"></div>', unsafe_allow_html=True)
            st.markdown(_cl_eig_badges(cl_eigs_2nd,  f"CL λ  ({approx_order_label}-order approx)"), unsafe_allow_html=True)

        except Exception as sim_err:
            st.error(f"Simulation error: {sim_err}")