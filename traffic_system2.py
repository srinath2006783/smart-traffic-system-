import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from neo4j import GraphDatabase
import time

# -----------------------------
# NEO4J CONFIG
# -----------------------------
NEO4J_URI = "neo4j+s://2ba57011.databases.neo4j.io"
NEO4J_USER = "2ba57011"
NEO4J_PASSWORD = "MPg5aMmkFJnam_F2zhCVr5WzphPcj0L7GsVFVuUDAUQ"
NEO4J_DB = "2ba57011"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🚦 Intelligent Traffic Control System")

WINDOW = 20
MAX_QUEUE = 120

SIM_INTERVAL = 1.0
DB_INTERVAL = 5.0
REFRESH_INTERVAL = st.slider("Refresh Speed", 1, 10, 4)

# -----------------------------
# CORE GRAPH (4 NODES)
# -----------------------------
nodes = ["T Nagar", "Guindy", "Velachery", "OMR Junction"]

edges = [
    ("T Nagar","Guindy"),
    ("Guindy","Velachery"),
    ("Velachery","OMR Junction"),
    ("OMR Junction","Guindy")
]

# adjacency weights (influence)
W = {
    ("T Nagar","Guindy"): 0.2,
    ("Guindy","Velachery"): 0.3,
    ("Velachery","OMR Junction"): 0.3,
    ("OMR Junction","Guindy"): 0.2
}

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

# -----------------------------
# STATE INIT
# -----------------------------
if "queues" not in st.session_state:
    st.session_state.queues = {n: np.random.randint(20, 40) for n in nodes}

if "history" not in st.session_state:
    st.session_state.history = {n: [30]*WINDOW for n in nodes}

if "signals" not in st.session_state:
    st.session_state.signals = {n: 0.5 for n in nodes}

if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

if "last_db_push" not in st.session_state:
    st.session_state.last_db_push = time.time()

# -----------------------------
# MATHEMATICAL MODEL DISPLAY
# -----------------------------
st.subheader("📐 System Model")

st.latex(r"Q_i(t+1) = Q_i(t) + A_i(t) + \sum_j w_{ji} Q_j(t) - S_i(t)")

# -----------------------------
# OPTIMIZATION (SIGNAL CONTROL)
# -----------------------------
total_queue = sum(st.session_state.queues.values())

for node in nodes:
    if total_queue > 0:
        st.session_state.signals[node] = st.session_state.queues[node] / total_queue

# -----------------------------
# SIMULATION (REAL SYSTEM)
# -----------------------------
if time.time() - st.session_state.last_update > SIM_INTERVAL:
    st.session_state.last_update = time.time()

    newQ = {}

    for node in nodes:
        Q = st.session_state.queues[node]

        # arrivals (stochastic input)
        arrivals = np.random.poisson(4)

        # influence from neighbors
        influence = sum(
            W.get((src,node),0) * st.session_state.queues[src]
            for src, tgt in edges if tgt == node
        )

        # service based on optimized signal
        service = st.session_state.signals[node] * 15

        newQ[node] = max(0, min(MAX_QUEUE, Q + arrivals + influence - service))

    st.session_state.queues = newQ

    for node in nodes:
        hist = st.session_state.history[node]
        hist.append(newQ[node])
        hist.pop(0)

# -----------------------------
# PREDICTION (TREND + MEAN)
# -----------------------------
predictions = {}

for node in nodes:
    series = st.session_state.history[node]

    trend = series[-1] - series[-2]
    avg = np.mean(series[-5:])

    predictions[node] = avg + trend

# -----------------------------
# GRIDLOCK DETECTION
# -----------------------------
if all(q > 90 for q in st.session_state.queues.values()):
    st.error("🚨 GRIDLOCK: Entire system saturated")

# -----------------------------
# PUSH TO NEO4J
# -----------------------------
def update_neo4j():
    with driver.session(database=NEO4J_DB) as session:
        for node in nodes:
            session.run("""
            MERGE (n:Intersection {id:$id})
            SET n.queue=$q
            """, id=node, q=float(st.session_state.queues[node]))

if time.time() - st.session_state.last_db_push > DB_INTERVAL:
    st.session_state.last_db_push = time.time()
    update_neo4j()

# -----------------------------
# GRAPH VISUAL
# -----------------------------
edge_traces = []

for src, tgt in edges:
    q1 = st.session_state.queues[src]
    q2 = st.session_state.queues[tgt]

    color = "red" if q1 > q2 else "gray"
    width = 4 if q1 > q2 else 1

    x0,y0 = pos[src]
    x1,y1 = pos[tgt]

    edge_traces.append(go.Scatter(
        x=[x0,x1,None],
        y=[y0,y1,None],
        mode="lines",
        line=dict(color=color,width=width)
    ))

node_trace = go.Scatter(
    x=[pos[n][0] for n in nodes],
    y=[pos[n][1] for n in nodes],
    text=[f"{n}<br>Q={st.session_state.queues[n]:.1f}" for n in nodes],
    mode="markers+text",
    textposition="top center",
    marker=dict(size=25, color=list(st.session_state.queues.values()), colorscale="Reds", showscale=True)
)

fig = go.Figure(data=edge_traces + [node_trace])
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INSIGHTS (NOT HARDCODED)
# -----------------------------
st.subheader("🚨 System Insights")

for node in nodes:
    q = st.session_state.queues[node]
    pred = predictions[node]

    if pred > 80:
        st.error(f"{node} will become critical soon")
    elif q > 60:
        st.warning(f"{node} is currently congested")
    else:
        st.success(f"{node} is stable")

# -----------------------------
# BOTTLENECK DETECTION
# -----------------------------
st.subheader("🧠 Bottlenecks")

for node in nodes:
    incoming = sum(st.session_state.queues[src] for src,tgt in edges if tgt == node)
    outgoing = sum(st.session_state.queues[tgt] for src,tgt in edges if src == node)

    if incoming > outgoing + 20:
        st.error(f"{node} is restricting flow")

# -----------------------------
# SIGNAL DECISIONS
# -----------------------------
st.subheader("🚦 Optimized Signal Allocation")

for node in nodes:
    st.write(f"{node}: Green time = {st.session_state.signals[node]*100:.1f}%")

# -----------------------------
# PREDICTION GRAPH
# -----------------------------
selected = st.selectbox("Inspect Node", nodes)

fig_ts = go.Figure()

fig_ts.add_trace(go.Scatter(
    y=st.session_state.history[selected],
    mode="lines+markers",
    name="Actual"
))

fig_ts.add_trace(go.Scatter(
    y=[predictions[selected]]*len(st.session_state.history[selected]),
    mode="lines",
    name="Predicted"
))

st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------------
# REFRESH
# -----------------------------
time.sleep(REFRESH_INTERVAL)
st.rerun()
