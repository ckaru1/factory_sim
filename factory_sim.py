# =========================
# Imports
# =========================
import streamlit as st
import matplotlib.pyplot as plt
import simpy
import random
import math


# =========================
# Page & minimal styling
# =========================
st.set_page_config(page_title="Factory Line UI", layout="wide")
st.markdown(
    """
    <style>
      .appview-container .main .block-container{
        max-width: 96vw;
        padding-top: 0.75rem;
        padding-bottom: 1.5rem;
      }

      
      section[data-testid="stSidebar"] { width: 300px; }
      @media (min-width: 1400px){
        section[data-testid="stSidebar"] { width: 320px; }
      }
      h1, h2, h3 { margin-top: .4rem; margin-bottom: .5rem; }
      .caption { opacity:.7; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Factory Line Simulation")
st.caption("Use the sidebar to configure the run.")


# =========================
# Simulation config 
# =========================
STATIONS = ["Miner", "Smelter", "Constructor", "Painter", "Packager"]

NUM_ITEMS = 100
ARRIVAL_JITTER = 0.0          # 0.0 = deterministic; 0.3 => mean inter-arrival 1.3× miner_time
IMBALANCE_FACTOR = 0.0        # spreads stations faster/slower around the center
ENABLE_JITTER = True          # enables station-time randomness (per-job variability)

# Per-station min/max process times (seconds per item)
RANGES = {
    "Miner":       {"min": 20, "max": 45},
    "Smelter":     {"min": 25, "max": 70},
    "Constructor": {"min": 15, "max": 50},
    "Painter":     {"min": 12, "max": 40},
    "Packager":    {"min": 10, "max": 35},
}

# Per-station variability as ± fraction (treated as CV for lognormal)
VARIABILITY = {
    "Smelter":     0.05,  # 5%
    "Constructor": 0.03,  # 3%
    "Painter":     0.02,  # 2%
    "Packager":    0.01,  # 1%
}

# Reproducibility (optional)
RANDOM_SEED = None  # set to an int (e.g., 42) to lock randomness, or leave as None
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# Quiet logging
VERBOSE = False
def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# State containers (populated per run)
station_stats = {
    "Smelter":     {"count": 0, "busy": 0},
    "Constructor": {"count": 0, "busy": 0},
    "Painter":     {"count": 0, "busy": 0},
    "Packager":    {"count": 0, "busy": 0},
}
last_completion_time = {"t": 0}

# Placeholders updated by set_machine_times()
miner_time = smelter_time = constructor_time = painter_time = packager_time = 0.0
chosen_times = {}
cycle_time = 0.0
efficiency = 0.0
bottlenecks = []


# =========================
# Helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def get_time(seconds):
    seconds = int(round(seconds))
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}m {seconds}s"

def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def util_color(u_pct: float) -> str:
    # <70% light blue, 70–90% blue, >90% coral/red
    if u_pct < 70: return "#9EC5FE"
    if u_pct < 90: return "#5B8DEF"
    return "#F45B69"


# =========================
# Calculator: station times
# =========================
def set_machine_times(mode="max", eta=None, slack=0.5):
    global miner_time, smelter_time, constructor_time, painter_time, packager_time
    global chosen_times, cycle_time, efficiency, bottlenecks

    min_times = [RANGES[s]["min"] for s in STATIONS]
    max_times = [RANGES[s]["max"] for s in STATIONS]
    num_stations = len(STATIONS)

    total_work_content = sum(min_times)
    fastest_feasible_cycle = max(min_times)
    slowest_feasible_cycle = min(max_times)

    if fastest_feasible_cycle > slowest_feasible_cycle:
        raise ValueError("Infeasible ranges: no overlap between min and max settings")

    if mode == "max":
        cycle = fastest_feasible_cycle
    elif mode == "eff":
        if not (eta and 0 < eta <= 1):
            raise ValueError("Need eta in (0,1] when mode='eff'")
        requested_cycle = total_work_content / (num_stations * eta)
        cycle = clamp(requested_cycle, fastest_feasible_cycle, slowest_feasible_cycle)
    else:
        raise ValueError("mode must be 'max' or 'eff'")

    # Apply slack then distribute imbalance linearly across stations around center
    chosen = {}
    for i, s in enumerate(STATIONS):
        base_time = cycle + slack
        adjusted = base_time * (1.0 + IMBALANCE_FACTOR * (i - num_stations/2) / num_stations)
        chosen[s] = clamp(adjusted, RANGES[s]["min"], RANGES[s]["max"])

    miner_time       = chosen["Miner"]
    smelter_time     = chosen["Smelter"]
    constructor_time = chosen["Constructor"]
    painter_time     = chosen["Painter"]
    packager_time    = chosen["Packager"]

    chosen_times = chosen
    cycle_time = cycle
    efficiency = total_work_content / (num_stations * cycle)
    bottlenecks = [s for s in STATIONS if RANGES[s]["min"] == fastest_feasible_cycle]

    throughput = 1.0 / cycle
    return chosen, cycle, throughput, efficiency, bottlenecks


# =========================
# SimPy processes
# =========================
def machine(name, env, process_time, input_store, output_store, total_items=None):
    finished_count = 0
    while True:
        item = yield input_store.get()

        if ENABLE_JITTER:
            cv = VARIABILITY.get(name, 0.0)
            if cv > 0:
                sigma = math.sqrt(math.log(1.0 + cv*cv))
                mu = math.log(process_time) - 0.5 * sigma * sigma
                actual_time = random.lognormvariate(mu, sigma)
            else:
                actual_time = process_time
        else:
            actual_time = process_time

        yield env.timeout(actual_time)
        log(f"{name} finished item {item} at {get_time(env.now)}")

        station_stats[name]["count"] += 1
        station_stats[name]["busy"]  += actual_time

        if output_store:
            yield output_store.put(item)
        else:
            log(f"Item {item} COMPLETED at {get_time(env.now)}")
            finished_count += 1
            last_completion_time["t"] = env.now
            if total_items and finished_count == total_items:
                log(f"=== Production completed at {get_time(env.now)} ===")

def source(env, NUM_ITEMS, output_store):
    log(f"=== Starting production ===")
    for i in range(1, NUM_ITEMS + 1):
        yield output_store.put(i)
        log(f"Item {i} Mined at {get_time(env.now)}")

        # Deterministic arrivals when ARRIVAL_JITTER == 0.0
        if ARRIVAL_JITTER > 0.0:
            mean = miner_time * (1.0 + ARRIVAL_JITTER)
            inter = random.expovariate(1.0 / mean)
        else:
            inter = miner_time
        yield env.timeout(inter)

def run_simulation():
    env = simpy.Environment()
    queue_A = simpy.Store(env)
    queue_B = simpy.Store(env)
    queue_C = simpy.Store(env)
    queue_D = simpy.Store(env)

    env.process(source(env, NUM_ITEMS, queue_A))
    env.process(machine("Smelter",     env, smelter_time,     queue_A, queue_B))
    env.process(machine("Constructor", env, constructor_time, queue_B, queue_C))
    env.process(machine("Painter",     env, painter_time,     queue_C, queue_D))
    env.process(machine("Packager",    env, packager_time,    queue_D, None, total_items=NUM_ITEMS))

    env.run()


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Run Settings")

    NUM_ITEMS = st.slider(
        "Number of items", 1, 1000, 100, 10,
        help="How many parts to simulate this run"
    )

    mode = st.selectbox(
        "Mode", ["max", "eff"], index=0,
        help="`max`: fastest feasible cycle time. `eff`: pick a cycle time to hit target line-balancing efficiency"
    )

    eta = st.slider(
        "Target efficiency η", 0.60, 1.00, 0.85,
        help="Desired design efficiency (0–1). Higher η → tighter cycle time"
    ) if mode == "eff" else None

    slack = st.slider(
        "Slack (sec)", 0.0, 2.0, 0.5, 0.1,
        help="Extra seconds added to the selected cycle time before clamping per-station. Small slack reduces starvation/blocking"
    )

    imbalance_factor = st.slider(
        "Line Imbalance Factor", 0.0, 2.0, 0.0, 0.1,
        help="Scales spread between faster/slower stations (creates persistent utilization differences)"
    )

    st.header("Randomness Settings")

    randomness_mode = st.selectbox(
        "Randomness mode",
        ["Off", "Arrivals only", "Stations only", "All randomness"],
        index=3,
        help="Choose which types of randomness to include in the simulation"
    )

    arrival_jitter = st.slider(
        "Arrival jitter (± fraction)", 0.0, 0.8, 0.0, 0.05,
        help="Randomness in mining interval. 0.0 = regular arrivals; >0 = bursty exponential arrivals"
    )

    with st.expander("Station Variability (± fraction)", expanded=False):
        st.caption("Per-job processing randomness (treated as CV). Higher → more job-to-job variation")
        for station in VARIABILITY.keys():
            VARIABILITY[station] = st.slider(
                f"{station}", 0.0, 0.50,
                value=float(VARIABILITY[station]),
                step=0.01,
                key=f"{station}_var",
                help="Coefficient of variation for this station’s processing time"
            )

    st.header("Station Settings")
    with st.expander("Per-Station Ranges (sec/item)", expanded=False):
        st.caption("Bounds for each station’s process time (sec/item). Keep min ≤ max")
        for station in STATIONS:
            c1, c2 = st.columns(2)
            with c1:
                min_val = st.number_input(
                    f"{station} min",
                    value=float(RANGES[station]["min"]),
                    step=1.0,
                    help="Lower bound on this station’s process time"
                )
            with c2:
                max_val = st.number_input(
                    f"{station} max",
                    value=float(RANGES[station]["max"]),
                    step=1.0,
                    help="Upper bound on this station’s process time"
                )
            # auto-correct if user inverted bounds
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            RANGES[station]["min"] = min_val
            RANGES[station]["max"] = max_val

run = st.button("Run Simulation", type="primary")


# =========================
# Results
# =========================
st.subheader("Results")
if run:
    # Apply sidebar settings to globals used by the sim
    NUM_ITEMS = int(NUM_ITEMS)
    IMBALANCE_FACTOR = float(imbalance_factor)

    if randomness_mode == "Off":
        ENABLE_JITTER = False
        ARRIVAL_JITTER = 0.0
    elif randomness_mode == "Arrivals only":
        ENABLE_JITTER = False
        ARRIVAL_JITTER = float(arrival_jitter)
    elif randomness_mode == "Stations only":
        ENABLE_JITTER = True
        ARRIVAL_JITTER = 0.0
    elif randomness_mode == "All randomness":
        ENABLE_JITTER = True
        ARRIVAL_JITTER = float(arrival_jitter)

    # Recalculate station times with current mode/eta/slack
    set_machine_times(mode=mode, eta=eta if mode == "eff" else None, slack=slack)

    # Reset per-run stats
    station_stats = {k: {"count": 0, "busy": 0} for k in station_stats}
    last_completion_time = {"t": 0}

    # Run the simulation
    run_simulation()

    # ---- KPIs ----
    makespan = last_completion_time["t"]
    if not makespan or makespan <= 0:
        makespan = 0.0

    sim_throughput = (NUM_ITEMS / makespan) if makespan > 0 else 0.0
    total_busy = sum(s["busy"] for s in station_stats.values())
    nstations = len(station_stats)
    runtime_eff = (total_busy / (nstations * makespan)) if makespan > 0 else 0.0

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Makespan", fmt_duration(makespan), help="Total simulated time to finish all items")
    with kpi2:
        st.metric("Throughput", f"{sim_throughput:.3f} items/s", help="Average completion rate in items per second")
    with kpi3:
        st.metric("Runtime Efficiency", f"{runtime_eff*100:.1f}%", help="Average utilization across stations during the run")

    # ---- Utilization chart ----
    plot_stations = ["Smelter", "Constructor", "Painter", "Packager"]
    utils_pct = [
        (station_stats[s]["busy"] / makespan) * 100 if makespan > 0 else 0.0
        for s in plot_stations
    ]
    colors = [util_color(u) for u in utils_pct]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(plot_stations, utils_pct, color=colors, alpha=0.95, edgecolor="#333", linewidth=0.6)
    ax.set_title("Station Utilization — Single Run", fontsize=16, weight="bold")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 111, 10))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for rect, val in zip(bars, utils_pct):
        y = min(rect.get_height() + 2, 106)
        ax.text(rect.get_x() + rect.get_width()/2, y, f"{val:.1f}%", ha="center", va="bottom", fontsize=11, weight="bold")

    for i, s in enumerate(plot_stations):
        if s in bottlenecks:
            ax.text(i, min(utils_pct[i] + 9, 106), "BN", ha="center", va="bottom", fontsize=10, color="#E11D48", weight="bold")
            bars[i].set_edgecolor("#E11D48")
            bars[i].set_linewidth(2.0)

    st.pyplot(fig, clear_figure=True)
    st.caption("Color scheme: Light blue < 70% (under-loaded), Blue 70–90% (healthy), Red > 90% (over-loaded). 'BN' marks the bottleneck station")

else:
    st.info("This space will show KPIs and the utilization chart after you click **Run Simulation**.")