import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    st.error("neo4j package not installed. Run: pip install neo4j")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CityFlow — Intelligent Traffic",
    layout="wide",
    page_icon="🚦"
)

# ── Neo4j credentials ────────────────────────────────────────────────────────
NEO4J_URI      = "neo4j+s://2ba57011.databases.neo4j.io"
NEO4J_USER     = "2ba57011"
NEO4J_PASSWORD = "MPg5aMmkFJnam_F2zhCVr5WzphPcj0L7GsVFVuUDAUQ"
NEO4J_DB       = "2ba57011"

# ── Simulation constants ─────────────────────────────────────────────────────
NUM_NODES    = 12
WINDOW       = 30
MAX_QUEUE    = 100
BASE_ARRIVAL = 3
EMBED_DIM    = 384

INTERSECTION_NAMES = [
    "Anna Salai / Mount Rd", "T.Nagar Junction",    "Adyar Signal",
    "Vadapalani Circle",     "Koyambedu Terminal",  "Guindy Overpass",
    "Velachery Hub",         "Tambaram Bypass",      "Porur Signal",
    "Madhavaram Gate",       "Sholinganallur IT",    "Perambur Depot"
]
nodes = [f"N{i}" for i in range(NUM_NODES)]

# ════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0c10 !important;
    color: #e2e8f0 !important;
  }
  .main, .stApp { background: #0a0c10 !important; }
  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

  .metric-card {
    background: linear-gradient(135deg, #141820 0%, #1a1f2e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
  }
  .metric-label {
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: #64748b; font-family: 'Space Mono', monospace; margin-bottom: 6px;
  }
  .metric-value {
    font-size: 28px; font-weight: 700;
    font-family: 'Space Mono', monospace; color: #f1f5f9;
  }
  .metric-sub { font-size: 12px; color: #64748b; margin-top: 4px; }

  .status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .node-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px; background: #141820; border-radius: 8px;
    margin: 4px 0; border-left: 3px solid var(--col);
    font-family: 'Space Mono', monospace; font-size: 13px;
  }
  .tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
  }
  .tag-low  { background: #064e3b; color: #6ee7b7; }
  .tag-mid  { background: #7c2d12; color: #fdba74; }
  .tag-high { background: #7f1d1d; color: #fca5a5; }

  .section-header {
    font-family: 'Space Mono', monospace; font-size: 13px;
    letter-spacing: 2px; text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid #1e2535; padding-bottom: 8px;
    margin: 20px 0 12px 0;
  }
  .db-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 6px; font-size: 11px;
    font-family: 'Space Mono', monospace; font-weight: 700; letter-spacing: 1px;
    margin: 2px 0;
  }
  .neo4j-badge { background:#052e16; color:#4ade80; border:1px solid #166534; }
  .faiss-badge { background:#1e1b4b; color:#a5b4fc; border:1px solid #3730a3; }
  .sim-badge   { background:#1c1917; color:#a8a29e; border:1px solid #44403c; }

  .stTextInput > div > div > input {
    background: #141820 !important; border: 1px solid #2d3748 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
  }
  .stSelectbox > div > div { background: #141820 !important; border-color: #2d3748 !important; }
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 12px !important;
    letter-spacing: 1px !important; padding: 8px 20px !important;
  }
  .stButton > button:hover { background: linear-gradient(135deg,#2563eb,#4f46e5) !important; }

  .alert-critical {
    background:#450a0a; border:1px solid #991b1b; border-left:4px solid #ef4444;
    border-radius:8px; padding:12px 16px; margin:6px 0;
    font-family:'Space Mono',monospace; font-size:13px;
  }
  .alert-warn {
    background:#431407; border:1px solid #92400e; border-left:4px solid #f97316;
    border-radius:8px; padding:12px 16px; margin:6px 0;
    font-family:'Space Mono',monospace; font-size:13px;
  }
  .query-result {
    background:#0f172a; border:1px solid #1e3a5f; border-radius:10px;
    padding:16px; margin:8px 0; font-family:'Space Mono',monospace;
    font-size:12px; color:#93c5fd;
  }
  .cypher-block {
    background:#020617; border:1px solid #1e293b; border-radius:8px;
    padding:14px; font-family:'Space Mono',monospace; font-size:12px;
    color:#7dd3fc; margin:8px 0; white-space:pre-wrap;
  }
  .connected-banner {
    background: linear-gradient(90deg, #052e16, #0f2027);
    border: 1px solid #166534; border-radius:10px;
    padding:12px 18px; margin:8px 0;
    font-family:'Space Mono',monospace; font-size:12px; color:#4ade80;
  }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  NEO4J DRIVER  (cached singleton)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

driver = get_driver() if NEO4J_AVAILABLE else None

def neo4j_run(query, **params):
    with driver.session(database=NEO4J_DB) as s:
        return list(s.run(query, **params))

def push_graph_to_neo4j(G):
    neo4j_run("MATCH (n:Intersection) DETACH DELETE n")
    for i, node in enumerate(nodes):
        neo4j_run(
            "CREATE (n:Intersection {id:$id, name:$name, zone:$zone, queue:0.0, "
            "predicted_queue:0.0, congestion_level:'LOW', updated_at:datetime()})",
            id=node, name=INTERSECTION_NAMES[i], zone=f"Zone-{i//3+1}"
        )
    for a, b in G.edges():
        neo4j_run("""
            MATCH (x:Intersection {id:$a}), (y:Intersection {id:$b})
            MERGE (x)-[:ROAD {distance:$dist}]->(y)
        """, a=a, b=b, dist=round(np.random.uniform(0.5, 3.5), 2))

def sync_queues_to_neo4j(queues, preds):
    rows = [
        {
            "id":    n,
            "q":     float(queues[n]),
            "pred":  float(preds[n]),
            "level": ("HIGH" if queues[n] > 55 else "MODERATE" if queues[n] > 25 else "LOW")
        }
        for n in nodes
    ]
    neo4j_run("""
        UNWIND $rows AS r
        MATCH (n:Intersection {id: r.id})
        SET n.queue            = r.q,
            n.predicted_queue  = r.pred,
            n.congestion_level = r.level,
            n.updated_at       = datetime()
    """, rows=rows)

def get_neo4j_stats():
    n_nodes = neo4j_run("MATCH (n:Intersection) RETURN count(n) AS c")[0]["c"]
    n_edges = neo4j_run("MATCH ()-[r:ROAD]->() RETURN count(r) AS c")[0]["c"]
    n_heavy = neo4j_run("MATCH (n:Intersection) WHERE n.congestion_level='HIGH' RETURN count(n) AS c")[0]["c"]
    return int(n_nodes), int(n_edges), int(n_heavy)

def neo4j_shortest_path(src, dst):
    rows = neo4j_run("""
        MATCH p = shortestPath(
            (a:Intersection {id:$src})-[:ROAD*]-(b:Intersection {id:$dst})
        )
        RETURN [n IN nodes(p) | n.id] AS path, length(p) AS hops
    """, src=src, dst=dst)
    if rows:
        return rows[0]["path"], rows[0]["hops"]
    return [], 0

def neo4j_zone_summary():
    return neo4j_run("""
        MATCH (n:Intersection)
        RETURN n.zone AS zone, avg(n.queue) AS avg_q, count(n) AS cnt
        ORDER BY avg_q DESC
    """)

# ════════════════════════════════════════════════════════════════════════════
#  SENTENCE TRANSFORMER + FAISS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if FAISS_AVAILABLE:
        return SentenceTransformer("all-MiniLM-L6-v2")
    return None

model = load_model()

def level_label(q):
    if q < 25:   return "LOW"
    elif q < 55: return "MODERATE"
    else:        return "HIGH"

def build_descriptor(node, q, pred, G):
    name  = INTERSECTION_NAMES[nodes.index(node)]
    lvl   = level_label(q).lower()
    nbrs  = [INTERSECTION_NAMES[nodes.index(nb)] for nb in list(G.successors(node))[:2]]
    trend = ("worsening rapidly" if pred > q+8 else
             "slightly worsening" if pred > q+3 else
             "improving" if pred < q-5 else "stable")
    desc  = f"{name} ({node}) has {lvl} congestion and is {trend}."
    if nbrs:
        desc += f" Connected to {', '.join(nbrs)}."
    return desc

@st.cache_data(ttl=5)
def build_faiss_index(desc_tuple):
    descs = list(desc_tuple)
    vecs  = model.encode(descs).astype("float32")
    idx   = faiss.IndexFlatL2(EMBED_DIM)
    idx.add(vecs)
    return idx, descs

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════════════
if "graph" not in st.session_state:
    # No fixed seed — every run is genuinely different
    G_tmp   = nx.gnm_random_graph(NUM_NODES, NUM_NODES * 2, directed=True)
    mapping = {i: nodes[i] for i in range(NUM_NODES)}
    st.session_state.graph       = nx.relabel_nodes(G_tmp, mapping)
    st.session_state.pos         = nx.spring_layout(st.session_state.graph)
    st.session_state.tick        = 0
    # Spread initial queues across a wider range so not all start low
    st.session_state.queues      = {n: float(np.random.randint(2, 45)) for n in nodes}
    st.session_state.history     = {n: list(np.random.randint(2, 40, WINDOW).astype(float)) for n in nodes}
    st.session_state.event_log   = []
    st.session_state.neo4j_ready = False
    # Per-node traffic weight: some intersections are naturally busier
    st.session_state.node_weight = {n: np.random.uniform(0.6, 1.6) for n in nodes}

G   = st.session_state.graph
pos = st.session_state.pos

# Push graph to Neo4j once per session
if driver and not st.session_state.neo4j_ready:
    try:
        push_graph_to_neo4j(G)
        st.session_state.neo4j_ready = True
    except Exception as e:
        st.warning(f"Neo4j init warning: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  SIMULATION TICK
# ════════════════════════════════════════════════════════════════════════════
def simulate_tick():
    new_q = {}
    node_weight = st.session_state.get("node_weight", {n: 1.0 for n in nodes})
    tick = st.session_state.tick

    for node in nodes:
        Q    = st.session_state.queues[node]
        nbrs = list(G.successors(node))
        avg_nb = np.mean([st.session_state.queues[nb] for nb in nbrs]) if nbrs else 0

        # ── Arrivals ──────────────────────────────────────────────────────
        # Base poisson + neighbor spillover + node-specific weight
        # When a node is at capacity, upstream nodes also slow arrivals (backpressure)
        backpressure = max(0.0, 1.0 - Q / MAX_QUEUE)   # 1.0 when empty, 0.0 when full
        raw_arrival  = np.random.poisson(BASE_ARRIVAL * node_weight[node])
        spillover    = 0.2 * avg_nb * backpressure
        arrivals     = raw_arrival * backpressure + spillover

        # ── Service (green signal duration scales with queue) ─────────────
        # Critical fix: at Q>=80 we deploy emergency service (>arrivals)
        # so the queue MUST drain, never get permanently stuck
        if Q >= 80:
            service = float(np.random.poisson(18))   # emergency: way more than arrivals
        elif Q >= 60:
            service = float(np.random.poisson(13))
        elif Q >= 35:
            service = float(np.random.poisson(8))
        elif Q >= 15:
            service = float(np.random.poisson(5))
        else:
            service = float(np.random.poisson(3))

        # ── Random incident: occasional surge (accident, event) ───────────
        # ~8% chance of a spike at any node per tick
        if np.random.random() < 0.08:
            arrivals += np.random.randint(5, 15)

        # ── Random clearance: occasional rapid drain (signal optimisation) ─
        # ~6% chance of a clearance event
        if np.random.random() < 0.06:
            service += np.random.randint(8, 18)

        new_q[node] = float(max(0.0, min(MAX_QUEUE, Q + arrivals - service)))

    for node in nodes:
        st.session_state.queues[node] = new_q[node]
        hist = st.session_state.history[node]
        hist.append(new_q[node])
        if len(hist) > WINDOW:
            hist.pop(0)

    st.session_state.tick += 1
    heavy = [n for n in nodes if new_q[n] > 70]
    if heavy:
        st.session_state.event_log.append({
            "tick": st.session_state.tick,
            "time": datetime.now().strftime("%H:%M:%S"),
            "nodes": heavy, "type": "CONGESTION"
        })
        if len(st.session_state.event_log) > 50:
            st.session_state.event_log.pop(0)

def predict(node):
    hist = st.session_state.history[node]
    if len(hist) < 5: return hist[-1]
    weights = np.array([1, 1.5, 2, 2.5, 3])
    return float(np.average(hist[-5:], weights=weights))

# ── Run one tick ──────────────────────────────────────────────────────────────
simulate_tick()
predictions = {n: predict(n) for n in nodes}
descs       = [build_descriptor(n, st.session_state.queues[n], predictions[n], G) for n in nodes]

# Sync to Neo4j
if driver and st.session_state.neo4j_ready:
    try:
        sync_queues_to_neo4j(st.session_state.queues, predictions)
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ System Status")

    if driver and st.session_state.neo4j_ready:
        st.markdown('<div class="db-badge neo4j-badge">● NEO4J LIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="db-badge" style="background:#1c0a0a;color:#f87171;border:1px solid #7f1d1d">✕ NEO4J OFFLINE</div>', unsafe_allow_html=True)

    if FAISS_AVAILABLE:
        st.markdown('<div class="db-badge faiss-badge">● FAISS ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="db-badge sim-badge">✕ FAISS NOT INSTALLED</div>', unsafe_allow_html=True)

    st.markdown('<div class="db-badge sim-badge">● SIMULATION RUNNING</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Database**")
    st.code(f"URI : {NEO4J_URI}\nDB  : {NEO4J_DB}\nUser: {NEO4J_USER}", language="text")

    st.markdown("---")
    st.markdown(f"**Tick:** `{st.session_state.tick}`")
    st.markdown(f"**Nodes:** `{NUM_NODES}` &nbsp; **Window:** `{WINDOW}`")

    try:
        from streamlit_autorefresh import st_autorefresh
        auto = st.toggle("Auto-refresh (3s)", value=False)
        if auto:
            st_autorefresh(interval=3000, key="autorefresh")
    except ImportError:
        st.caption("Install streamlit-autorefresh for auto mode")

    st.markdown("---")
    st.markdown("**Schema**")
    st.markdown("""
```
(:Intersection {
  id, name, zone,
  queue,
  predicted_queue,
  congestion_level,
  updated_at
})-[:ROAD {
  distance
}]->(:Intersection)
```""")

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:4px">
  <div style="font-family:'Space Mono',monospace;font-size:26px;font-weight:700;
              background:linear-gradient(90deg,#38bdf8,#818cf8);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    🚦 CITYFLOW
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:11px;
              color:#64748b;letter-spacing:2px;padding-top:6px">
    INTELLIGENT TRAFFIC MANAGEMENT · CHENNAI
  </div>
</div>
<div style="font-size:12px;color:#475569;font-family:'Space Mono',monospace;margin-bottom:20px">
  Neo4j AuraDB &nbsp;·&nbsp; FAISS Semantic Search &nbsp;·&nbsp; Adaptive Signal Control
  &nbsp;|&nbsp; Tick #{st.session_state.tick} &nbsp;·&nbsp; {datetime.now().strftime("%H:%M:%S")}
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TOP METRICS
# ════════════════════════════════════════════════════════════════════════════
avg_q    = np.mean(list(st.session_state.queues.values()))
heavy_n  = sum(1 for q in st.session_state.queues.values() if q > 55)
warn_n   = sum(1 for n in nodes if predictions[n] > st.session_state.queues[n] + 8)
max_node = max(nodes, key=lambda n: st.session_state.queues[n])

def metric_card(col, label, value, sub, accent):
    col.markdown(f"""
    <div class="metric-card" style="--accent:{accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
metric_card(c1, "AVG QUEUE LENGTH",   f"{avg_q:.1f}",      "vehicles / intersection",        "#38bdf8")
metric_card(c2, "HEAVY CONGESTION",   f"{heavy_n} nodes",  "queue > 55",                     "#ef4444")
metric_card(c3, "SURGE WARNINGS",     f"{warn_n} alerts",  "predicted to worsen next tick",  "#f97316")
metric_card(c4, "WORST INTERSECTION", max_node,
            f"{INTERSECTION_NAMES[nodes.index(max_node)]} · {st.session_state.queues[max_node]:.0f} vehicles",
            "#a78bfa")

# ════════════════════════════════════════════════════════════════════════════
#  MAIN: Graph  |  Node Table
# ════════════════════════════════════════════════════════════════════════════
col_graph, col_table = st.columns([2, 1])

with col_graph:
    st.markdown('<div class="section-header">INTERSECTION NETWORK GRAPH</div>', unsafe_allow_html=True)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]; x1, y1 = pos[b]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    nx_v, ny_v, nc_v, nt_v, ns_v = [], [], [], [], []
    for node in nodes:
        x, y = pos[node]; q = st.session_state.queues[node]
        nx_v.append(x); ny_v.append(y); nc_v.append(q)
        nt_v.append(f"<b>{node}</b><br>{INTERSECTION_NAMES[nodes.index(node)]}<br>"
                    f"Queue: {q:.0f} | Pred: {predictions[node]:.0f}<br>"
                    f"Status: {level_label(q)}")
        ns_v.append(max(14, min(30, q * 0.35)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="#1e293b", width=1.5), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=nx_v, y=ny_v, mode="markers+text",
        marker=dict(
            size=ns_v, color=nc_v,
            colorscale=[[0,"#064e3b"],[0.35,"#065f46"],[0.55,"#78350f"],[0.75,"#92400e"],[1,"#7f1d1d"]],
            cmin=0, cmax=MAX_QUEUE, showscale=True,
            colorbar=dict(
                title=dict(text="Queue", font=dict(color="#64748b", size=11)),
                tickfont=dict(color="#64748b"),
                bgcolor="#0a0c10",
                bordercolor="#2d3748"
            ),
            line=dict(color="#0a0c10", width=1.5)
        ),
        text=nodes, textposition="middle center",
        textfont=dict(color="white", size=10, family="Space Mono"),
        hovertext=nt_v, hoverinfo="text"
    ))
    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
        margin=dict(l=0,r=0,t=0,b=0), height=420, showlegend=False,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)

    sel = st.selectbox("📈 Queue history for:",
                       nodes, format_func=lambda n: f"{n} — {INTERSECTION_NAMES[nodes.index(n)]}")
    hist_data = st.session_state.history[sel]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=hist_data, mode="lines+markers",
                              line=dict(color="#38bdf8",width=2),
                              marker=dict(size=4,color="#38bdf8"),
                              fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"))
    fig2.add_hline(y=predictions[sel], line=dict(color="#f97316",dash="dot",width=1.5),
                   annotation_text=f"Pred: {predictions[sel]:.0f}",
                   annotation_font_color="#f97316")
    fig2.update_layout(paper_bgcolor="#0a0c10", plot_bgcolor="#141820",
                       height=160, margin=dict(l=0,r=0,t=10,b=0),
                       xaxis=dict(showgrid=False,color="#475569"),
                       yaxis=dict(gridcolor="#1e293b",color="#475569"),
                       font=dict(color="#94a3b8"), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_table:
    st.markdown('<div class="section-header">LIVE NODE STATUS</div>', unsafe_allow_html=True)

    for node in sorted(nodes, key=lambda n: st.session_state.queues[n], reverse=True):
        q    = st.session_state.queues[node]
        pred = predictions[node]
        lvl  = level_label(q)
        tag  = {"LOW":"tag-low","MODERATE":"tag-mid","HIGH":"tag-high"}[lvl]
        col_ = "#ef4444" if lvl=="HIGH" else ("#f97316" if lvl=="MODERATE" else "#4ade80")
        tclr = "#ef4444" if pred > q+3 else ("#4ade80" if pred < q-3 else "#94a3b8")
        tico = "↑" if pred > q+3 else ("↓" if pred < q-3 else "→")
        st.markdown(f"""
        <div class="node-row" style="--col:{col_}">
          <span><b>{node}</b>
            <span style="color:#64748b;font-size:11px"> {INTERSECTION_NAMES[nodes.index(node)][:16]}</span>
          </span>
          <span>
            <b>{q:.0f}</b>
            <span style="color:{tclr};margin-left:4px">{tico}{pred:.0f}</span>
            &nbsp;<span class="tag {tag}">{lvl}</span>
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:20px">🚦 SIGNAL CONTROL</div>',
                unsafe_allow_html=True)
    for n in sorted(nodes, key=lambda n: st.session_state.queues[n], reverse=True)[:3]:
        q     = st.session_state.queues[n]
        green = min(90, int(30 + q * 0.6))
        st.markdown(f"""
        <div class="alert-warn">
          <b>{n}</b> · {INTERSECTION_NAMES[nodes.index(n)][:22]}<br>
          <span style="color:#94a3b8;font-size:12px">
            Green phase → <b style="color:#fbbf24">{green}s</b> &nbsp;|&nbsp; Queue: {q:.0f}
          </span>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECOND ROW : Alerts | Smart Query | Neo4j Live Explorer
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
col_alert, col_query, col_neo = st.columns([1, 1.2, 1])

# ── ALERTS ────────────────────────────────────────────────────────────────────
with col_alert:
    st.markdown('<div class="section-header">🚨 CONGESTION ALERTS</div>', unsafe_allow_html=True)
    alert_count = 0

    for node in nodes:
        q    = st.session_state.queues[node]
        pred = predictions[node]
        if q > 70:
            st.markdown(f"""
            <div class="alert-critical">
              <span class="status-dot" style="background:#ef4444"></span>
              <b>CRITICAL</b> · {node}<br>
              <span style="color:#94a3b8;font-size:12px">
                {INTERSECTION_NAMES[nodes.index(node)]}<br>
                Queue: {q:.0f} | Pred: {pred:.0f}
              </span>
            </div>""", unsafe_allow_html=True)
            alert_count += 1
        elif pred > q + 10:
            st.markdown(f"""
            <div class="alert-warn">
              <span class="status-dot" style="background:#f97316"></span>
              <b>WARNING</b> · {node}<br>
              <span style="color:#94a3b8;font-size:12px">
                {INTERSECTION_NAMES[nodes.index(node)]}<br>
                Rising: {q:.0f} → {pred:.0f}
              </span>
            </div>""", unsafe_allow_html=True)
            alert_count += 1

    if alert_count == 0:
        st.markdown("""
        <div style="padding:24px;text-align:center;color:#4ade80;
                    font-family:'Space Mono',monospace;font-size:13px">
          ✓ All intersections nominal
        </div>""", unsafe_allow_html=True)

    if st.session_state.event_log:
        st.markdown('<div class="section-header" style="margin-top:16px">EVENT LOG</div>',
                    unsafe_allow_html=True)
        for ev in reversed(st.session_state.event_log[-6:]):
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:11px;
                        color:#64748b;padding:4px 0;border-bottom:1px solid #1e293b">
              [{ev['time']}] T{ev['tick']} · {ev['type']} · {', '.join(ev['nodes'])}
            </div>""", unsafe_allow_html=True)

# ── SMART QUERY ────────────────────────────────────────────────────────────────
with col_query:
    st.markdown('<div class="section-header">🔍 SMART QUERY ENGINE</div>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="e.g. 'heavy congestion near IT hub' or 'worsening nodes'")

    if query:
        ql = query.lower()

        if any(k in ql for k in ["least","low","clear","free"]):
            st.markdown("**🟢 Least Congested**")
            for n in sorted(nodes, key=lambda n: st.session_state.queues[n])[:4]:
                q = st.session_state.queues[n]
                st.markdown(f"""
                <div class="query-result">
                  {n} · {INTERSECTION_NAMES[nodes.index(n)]}<br>
                  Queue: {q:.0f} &nbsp;|&nbsp; {level_label(q)}
                </div>""", unsafe_allow_html=True)

        elif any(k in ql for k in ["most","heavy","worst","jam","congested"]):
            st.markdown("**🔴 Most Congested**")
            for n in sorted(nodes, key=lambda n: st.session_state.queues[n], reverse=True)[:4]:
                q = st.session_state.queues[n]
                st.markdown(f"""
                <div class="query-result" style="border-color:#7f1d1d;color:#fca5a5">
                  {n} · {INTERSECTION_NAMES[nodes.index(n)]}<br>
                  Queue: {q:.0f} &nbsp;|&nbsp; {level_label(q)}
                </div>""", unsafe_allow_html=True)

        elif any(k in ql for k in ["wors","rising","increas","surge"]):
            st.markdown("**⚠️ Worsening Nodes**")
            found = [(n, st.session_state.queues[n], predictions[n])
                     for n in nodes if predictions[n] > st.session_state.queues[n] + 5]
            if found:
                for n, q, p in sorted(found, key=lambda x: x[2]-x[1], reverse=True):
                    st.markdown(f"""
                    <div class="query-result" style="border-color:#92400e;color:#fdba74">
                      {n} · {INTERSECTION_NAMES[nodes.index(n)]}<br>
                      {q:.0f} → <b>{p:.0f}</b> (+{p-q:.0f})
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No worsening nodes detected.")

        else:
            if FAISS_AVAILABLE and model:
                st.markdown("**🧠 Semantic Matches (FAISS)**")
                qvec         = model.encode([query]).astype("float32")
                faiss_idx, _ = build_faiss_index(tuple(descs))
                D, I         = faiss_idx.search(qvec, 4)
                for rank, (dist, i) in enumerate(zip(D[0], I[0])):
                    n   = nodes[i]
                    q   = st.session_state.queues[n]
                    sim = max(0, 1 - dist / 10)
                    st.markdown(f"""
                    <div class="query-result">
                      #{rank+1} · {n} · {INTERSECTION_NAMES[i]}<br>
                      <span style="font-size:11px;color:#64748b">{descs[i]}</span><br>
                      Similarity: {sim:.2f} &nbsp;|&nbsp; Queue: {q:.0f}
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("FAISS not installed. Run: pip install faiss-cpu sentence-transformers")

    st.markdown('<div class="section-header" style="margin-top:16px">SAMPLE CYPHER</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="cypher-block">// High congestion intersections
MATCH (n:Intersection)
WHERE n.congestion_level = 'HIGH'
RETURN n.id, n.name, n.queue
ORDER BY n.queue DESC

// Shortest path
MATCH p = shortestPath(
  (a:Intersection {id:'N0'})
    -[:ROAD*]-
  (b:Intersection {id:'N7'})
)
RETURN [x IN nodes(p)|x.id], length(p)

// Zone avg congestion
MATCH (n:Intersection)
RETURN n.zone,
  avg(n.queue) AS avg_q,
  count(n) AS nodes
ORDER BY avg_q DESC</div>""", unsafe_allow_html=True)

# ── NEO4J LIVE EXPLORER ────────────────────────────────────────────────────────
with col_neo:
    st.markdown('<div class="section-header">🗄️ NEO4J LIVE EXPLORER</div>', unsafe_allow_html=True)

    if driver and st.session_state.neo4j_ready:
        try:
            n_nodes, n_edges, n_heavy = get_neo4j_stats()

            st.markdown(f"""
            <div class="connected-banner">
              <span class="status-dot" style="background:#4ade80"></span>
              <b>CONNECTED</b> · AuraDB<br>
              <span style="color:#86efac;font-size:11px">{NEO4J_URI}</span>
            </div>
            <div class="metric-card" style="--accent:#4ade80;margin-top:8px">
              <div class="metric-label">GRAPH STATS (LIVE FROM NEO4J)</div>
              <div style="font-family:'Space Mono',monospace;font-size:13px;margin-top:8px;line-height:2">
                Intersections : <b>{n_nodes}</b><br>
                Road edges &nbsp;&nbsp;: <b>{n_edges}</b><br>
                Heavy nodes &nbsp;: <b style="color:#ef4444">{n_heavy}</b><br>
                Database &nbsp;&nbsp;&nbsp;&nbsp;: <b>{NEO4J_DB}</b>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Zone Congestion (Neo4j query)**")
            for row in neo4j_zone_summary():
                avg_q = row["avg_q"]
                col_  = "#ef4444" if avg_q > 55 else "#f97316" if avg_q > 25 else "#4ade80"
                st.markdown(f"""
                <div style="font-family:'Space Mono',monospace;font-size:12px;
                            padding:8px 12px;background:#141820;border-radius:6px;
                            margin:4px 0;border-left:3px solid {col_}">
                  {row['zone']} &nbsp;·&nbsp; avg: <b>{avg_q:.1f}</b>
                  &nbsp;·&nbsp; {int(row['cnt'])} nodes
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:16px">SHORTEST PATH (Cypher)</div>',
                        unsafe_allow_html=True)
            sp_src = st.selectbox("From", nodes, key="sp_src")
            sp_dst = st.selectbox("To",   nodes, key="sp_dst", index=min(6, NUM_NODES-1))

            if st.button("⟳ FIND SHORTEST PATH", use_container_width=True):
                if sp_src == sp_dst:
                    st.warning("Source and destination are the same.")
                else:
                    path, hops = neo4j_shortest_path(sp_src, sp_dst)
                    if path:
                        st.markdown(f"""
                        <div class="query-result">
                          <b>{hops} hops</b><br>{' → '.join(path)}
                        </div>""", unsafe_allow_html=True)
                        for pn in path:
                            pq = st.session_state.queues.get(pn, 0)
                            st.markdown(f"""
                            <div style="font-family:'Space Mono',monospace;font-size:11px;
                                        color:#64748b;padding:2px 8px">
                              {pn} · {INTERSECTION_NAMES[nodes.index(pn)]} · queue: {pq:.0f}
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("No path found between selected nodes.")

        except Exception as e:
            st.error(f"Neo4j query error: {e}")
    else:
        st.error("Neo4j not connected. Check package installation.")

# ════════════════════════════════════════════════════════════════════════════
#  REFRESH BUTTON
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
_, btn_col, _ = st.columns([3, 1, 3])
with btn_col:
    if st.button("⟳  NEXT TICK", use_container_width=True):
        st.rerun()
