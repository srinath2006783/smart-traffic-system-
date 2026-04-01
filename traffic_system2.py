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
st.title("🚦 Smart Traffic Control (Core Network)")

WINDOW = 20
MAX_QUEUE = 100

SIM_INTERVAL = 1.0
DB_INTERVAL = 5.0
REFRESH_INTERVAL = st.slider("Refresh Speed", 1, 10, 4)

# -----------------------------
# NODES + EDGES
# -----------------------------
nodes = ["T Nagar", "Guindy", "Velachery", "OMR Junction"]

edges = [
    ("T Nagar","Guindy"),
    ("Guindy","Velachery"),
    ("Velachery","OMR Junction"),
    ("OMR Junction","Guindy")
]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

# -----------------------------
# STATE INIT
# -----------------------------
if "queues" not in st.session_state:
    st.session_state.queues = {n: np.random.randint(10, 30) for n in nodes}

if "history" not in st.session_state:
    st.session_state.history = {n: [20]*WINDOW for n in nodes}

if "signals" not in st.session_state:
    st.session_state.signals = {n: 0.5 for n in nodes}  # green ratio

if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

if "last_db_push" not in st.session_state:
    st.session_state.last_db_push = time.time()

# -----------------------------
# SIMULATION (WITH SIGNAL CONTROL)
# -----------------------------
if time.time() - st.session_state.last_update > SIM_INTERVAL:
    st.session_state.last_update = time.time()

    newQ = {}

    for node in nodes:
        Q = st.session_state.queues[node]

        arrivals = np.random.poisson(4)

        # incoming flow from neighbors
        incoming = sum(0.3 * st.session_state.queues[src] for src, tgt in edges if tgt == node)

        # signal-based service
        green = st.session_state.signals[node]
        service = green * 10

        newQ[node] = max(0, min(MAX_QUEUE, Q + arrivals + incoming - service))

    st.session_state.queues = newQ

    for node in nodes:
        hist = st.session_state.history[node]
        hist.append(newQ[node])
        hist.pop(0)

# -----------------------------
# GRIDLOCK DETECTION
# -----------------------------
gridlock = all(q > 80 for q in st.session_state.queues.values())

if gridlock:
    st.error("🚨 GRIDLOCK DETECTED — entire network saturated!")

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
# SIGNAL CONTROL LOGIC
# -----------------------------
for node in nodes:
    q = st.session_state.queues[node]

    if q > 70:
        st.session_state.signals[node] = 0.9
    elif q > 40:
        st.session_state.signals[node] = 0.7
    else:
        st.session_state.signals[node] = 0.4

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
# HUMAN INSIGHTS
# -----------------------------
st.subheader("🚨 Traffic Insights")

for node in nodes:
    q = st.session_state.queues[node]

    if q > 80:
        st.error(f"{node} is critically congested")
    elif q > 50:
        st.warning(f"{node} is heavily loaded")
    else:
        st.success(f"{node} is flowing")

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("🔮 Prediction")

for node in nodes:
    series = st.session_state.history[node]
    trend = series[-1] - series[-2]
    future = series[-1] + trend

    if future > 70:
        st.warning(f"{node} congestion will increase")

# -----------------------------
# BOTTLENECK
# -----------------------------
st.subheader("🧠 Bottleneck Detection")

for node in nodes:
    incoming = sum(st.session_state.queues[a] for a,b in edges if b == node)
    outgoing = sum(st.session_state.queues[b] for a,b in edges if a == node)

    if incoming > outgoing + 20:
        st.error(f"{node} is causing upstream congestion")

# -----------------------------
# SIGNAL ACTIONS
# -----------------------------
st.subheader("🚦 Signal Actions")

for node in nodes:
    green = st.session_state.signals[node]

    if green > 0.8:
        st.error(f"Maximize green at {node}")
    elif green > 0.6:
        st.warning(f"Increase green at {node}")
    else:
        st.success(f"Normal signal at {node}")

# -----------------------------
# REFRESH
# -----------------------------
time.sleep(REFRESH_INTERVAL)
st.rerun()