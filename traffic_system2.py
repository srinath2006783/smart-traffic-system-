import streamlit as st
import numpy as np
import time
from neo4j import GraphDatabase
import faiss

# -----------------------------
# NEO4J CONFIG
# -----------------------------
NEO4J_URI = "neo4j+s://2ba57011.databases.neo4j.io"
NEO4J_USER = "2ba57011"
NEO4J_PASSWORD = "MPg5aMmkFJnam_F2zhCVr5WzphPcj0L7GsVFVuUDAUQ"
NEO4J_DB = "2ba57011"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---------------- FAISS ----------------
DIM = 4
index = faiss.IndexFlatL2(DIM)
vector_store = []

# ---------------- NODES ----------------
NODES = [
    "Guindy", "T Nagar", "Velachery", "Adyar",
    "OMR Junction", "Tambaram", "Airport",
    "Central", "Egmore", "Vadapalani"
]

# ---------------- INIT DB ----------------
def init_db():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for node in NODES:
            session.run("""
            CREATE (:Intersection {
                id: $id,
                queue: rand()*20,
                trend: 0,
                variance: 0
            })
            """, id=node)

        # Meaningful connections
        edges = [
            ("Guindy", "T Nagar"),
            ("Guindy", "Velachery"),
            ("Guindy", "Airport"),
            ("T Nagar", "Central"),
            ("Velachery", "Adyar"),
            ("Adyar", "OMR Junction"),
            ("Guindy", "Vadapalani"),
            ("Vadapalani", "Egmore"),
            ("Tambaram", "Airport")
        ]

        for a, b in edges:
            session.run("""
            MATCH (a:Intersection {id:$a}), (b:Intersection {id:$b})
            MERGE (a)-[:CONNECTED_TO]->(b)
            """, a=a, b=b)

# ---------------- SIMULATION ----------------
def update_traffic():
    with driver.session() as session:
        result = session.run("MATCH (n:Intersection) RETURN n.id AS id, n.queue AS q")

        for record in result:
            node = record["id"]
            q = record["q"]

            inflow = np.random.randint(0, 10)
            outflow = np.random.randint(0, 8)

            new_q = max(0, q + inflow - outflow)

            session.run("""
            MATCH (n:Intersection {id:$id})
            SET n.queue=$q,
                n.trend=$trend,
                n.variance=$var
            """, id=node, q=new_q,
                 trend=inflow-outflow,
                 var=np.random.random())

# ---------------- FAISS LOGIC ----------------
def update_faiss():
    global vector_store

    with driver.session() as session:
        result = session.run("""
        MATCH (n:Intersection)
        RETURN n.queue AS q, n.trend AS t, n.variance AS v
        """)

        for r in result:
            vec = np.array([r["q"], r["t"], r["v"], 1], dtype="float32")
            vector_store.append(vec)

            if len(vector_store) > 100:
                vector_store.pop(0)

    if len(vector_store) > 10:
        index.reset()
        index.add(np.array(vector_store))

# ---------------- BOTTLENECK DETECTION ----------------
def detect_bottlenecks():
    alerts = []

    with driver.session() as session:
        result = session.run("""
        MATCH (a:Intersection)-[:CONNECTED_TO]->(b:Intersection)
        WHERE a.queue > b.queue + 5
        RETURN a.id AS from, b.id AS to, a.queue AS qa, b.queue AS qb
        """)

        for r in result:
            alerts.append(
                f"🚨 Heavy traffic at {r['from']} → affecting {r['to']} "
                f"(Queue: {int(r['qa'])} → {int(r['qb'])})"
            )

    return alerts

# ---------------- CONTROL LOGIC ----------------
def suggest_control():
    suggestions = []

    with driver.session() as session:
        result = session.run("""
        MATCH (n:Intersection)
        WHERE n.queue > 30
        RETURN n.id AS id, n.queue AS q
        """)

        for r in result:
            suggestions.append(
                f"🚦 Increase GREEN signal time at {r['id']} (Queue={int(r['q'])})"
            )

    return suggestions

# ---------------- STREAMLIT ----------------
st.title("🚦 Smart Traffic System (Neo4j + FAISS)")

if st.button("Initialize System"):
    init_db()
    st.success("System Initialized")

update_traffic()
update_faiss()

alerts = detect_bottlenecks()
controls = suggest_control()

st.subheader("🚨 Live Traffic Alerts")
for a in alerts:
    st.error(a)

st.subheader("🧠 Signal Control Suggestions")
for c in controls:
    st.warning(c)

st.subheader("🔄 Auto Refresh")
time.sleep(2)
st.experimental_rerun()
