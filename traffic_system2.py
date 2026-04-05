import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("🚦 Optimized Semantic + Graph Traffic System")

NUM_NODES = 12
WINDOW = 20
MAX_QUEUE = 80
ARRIVAL_RATE = 3
SERVICE_RATE = 5

# Slower refresh = smoother app
st_autorefresh(interval=3000, key="refresh")

nodes = [f"N{i}" for i in range(NUM_NODES)]

# ---------------- NEO4J ----------------
NEO4J_URI = "neo4j+s://2ba57011.databases.neo4j.io"
NEO4J_USER = "2ba57011"
NEO4J_PASSWORD = "MPg5aMmkFJnam_F2zhCVr5WzphPcj0L7GsVFVuUDAUQ"
NEO4J_DB = "2ba57011"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
EMBED_DIM = 384

# ---------------- GRAPH ----------------
if "graph" not in st.session_state:
    G_temp = nx.gnm_random_graph(NUM_NODES, NUM_NODES*2, directed=True)
    mapping = {i: nodes[i] for i in range(NUM_NODES)}
    st.session_state.graph = nx.relabel_nodes(G_temp, mapping)
    st.session_state.pos = nx.spring_layout(st.session_state.graph, seed=42)

G = st.session_state.graph
pos = st.session_state.pos

# ---------------- PUSH GRAPH (ONCE) ----------------
def push_graph():
    with driver.session() as session:
        for node in nodes:
            session.run("MERGE (n:Intersection {id:$id})", id=node)

        for edge in G.edges():
            session.run("""
            MATCH (a:Intersection {id:$a}), (b:Intersection {id:$b})
            MERGE (a)-[:CONNECTED_TO]->(b)
            """, a=edge[0], b=edge[1])

if "graph_loaded" not in st.session_state:
    push_graph()
    st.session_state.graph_loaded = True

# ---------------- STATE ----------------
if "queues" not in st.session_state:
    st.session_state.queues = {n: np.random.randint(0, 10) for n in nodes}

if "history" not in st.session_state:
    st.session_state.history = {n: list(np.random.randint(0, 10, WINDOW)) for n in nodes}

# ---------------- SIMULATION ----------------
for node in nodes:
    Q = st.session_state.queues[node]

    arrivals = np.random.poisson(ARRIVAL_RATE)
    service = SERVICE_RATE if np.random.rand() > 0.5 else 0

    newQ = max(0, min(MAX_QUEUE, Q + arrivals - service))
    st.session_state.queues[node] = newQ

    hist = st.session_state.history[node]
    hist.append(newQ)
    if len(hist) > WINDOW:
        hist.pop(0)

# ---------------- FAST NEO4J UPDATE ----------------
def update_neo4j():
    data = [
        {"id": node, "q": float(st.session_state.queues[node])}
        for node in nodes
    ]

    with driver.session() as session:
        session.run("""
        UNWIND $data AS row
        MATCH (n:Intersection {id: row.id})
        SET n.queue = row.q
        """, data=data)

update_neo4j()

# ---------------- DESCRIPTORS ----------------
def generate_descriptor(node, series):
    Q = series[-1]
    trend = series[-1] - series[-2] if len(series) > 1 else 0

    if Q > 40:
        level = "heavy congestion"
    elif Q > 20:
        level = "moderate traffic"
    else:
        level = "light traffic"

    if trend > 2:
        t = "rapidly increasing"
    elif trend > 0:
        t = "increasing"
    elif trend < -2:
        t = "rapidly decreasing"
    else:
        t = "stable"

    return f"{node} has {level} and is {t}"

descriptors = [generate_descriptor(n, st.session_state.history[n]) for n in nodes]

# ---------------- CACHE EMBEDDINGS ----------------
@st.cache_data(ttl=5)
def compute_embeddings(desc):
    return model.encode(desc)

emb = compute_embeddings(tuple(descriptors))
vectors = np.array(emb).astype("float32")

# ---------------- CACHE FAISS ----------------
@st.cache_data(ttl=5)
def build_faiss(vectors):
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(vectors)
    return index

index = build_faiss(vectors)

# ---------------- GRAPH VIS ----------------
edge_x, edge_y = [], []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y, node_color = [], [], []

for node in nodes:
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(st.session_state.queues[node])

fig = go.Figure()

fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines"))

fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(size=15, color=node_color, colorscale="Reds", showscale=True),
    text=nodes
))

st.plotly_chart(fig, use_container_width=True)

# ---------------- QUERY ----------------
st.subheader("🔍 Semantic Traffic Query")

query = st.text_input("Describe traffic condition")

if query:
    qvec = model.encode([query]).astype("float32")
    D, I = index.search(qvec, 5)

    st.subheader("Matching Nodes")
    matched_nodes = []

    for idx in I[0]:
        node = nodes[idx]
        matched_nodes.append(node)
        st.write(node)
        st.write(descriptors[idx])

    # ---------------- GRAPH REASONING ----------------
    st.subheader("🔗 Impact Analysis")

    with driver.session() as session:
        for node in matched_nodes:
            result = session.run("""
            MATCH (a:Intersection {id:$id})-[:CONNECTED_TO]->(b)
            RETURN b.id AS neighbor, b.queue AS q
            """, id=node)

            for r in result:
                st.write(f"{node} → {r['neighbor']} (queue={r['q']})")

# ---------------- BOTTLENECK ----------------
st.subheader("🚨 Bottlenecks")

with driver.session() as session:
    result = session.run("""
    MATCH (a:Intersection)-[:CONNECTED_TO]->(b)
    WHERE a.queue > b.queue + 5
    RETURN a.id AS from, b.id AS to
    """)

    for r in result:
        st.error(f"{r['from']} → {r['to']} congestion")
