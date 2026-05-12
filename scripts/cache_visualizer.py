"""Real-time bipartite cache visualizer for the ReuseLedger.

A lightweight Flask app that reads the ReuseLedger SQLite DB and serves
a dashboard with:
  - Bipartite graph (config prefixes <-> question IDs) via D3.js
  - Stats panel (total entries, unique prefixes, unique questions)
  - Prefix-depth heatmap (cache density at each pipeline depth)

Auto-refreshes every 2 seconds via JS polling.

Usage:
    pip install flask   # if not already installed
    python scripts/cache_visualizer.py --db /path/to/reuse_ledger.db --port 5050

The Flask app reads the DB in read-only mode using SQLite WAL, so it
can run concurrently with the UCT search writing to the same DB.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    print("Flask not installed. Run: pip install flask", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
DB_PATH: str = ":memory:"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _query_all_entries() -> list[tuple]:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT prefix_key, question_id, execution_id, checkpoint_step, created_at "
            "FROM materialized_entries ORDER BY created_at"
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stats")
def api_stats():
    rows = _query_all_entries()
    prefixes = set()
    questions = set()
    by_minute: dict[str, int] = defaultdict(int)
    for prefix_key, qid, _, _, created_at in rows:
        prefixes.add(prefix_key)
        questions.add(qid)
        minute = created_at[:16] if created_at else "unknown"
        by_minute[minute] += 1

    return jsonify({
        "total_entries": len(rows),
        "unique_prefixes": len(prefixes),
        "unique_questions": len(questions),
        "entries_by_minute": dict(sorted(by_minute.items())),
    })


@app.route("/api/graph")
def api_graph():
    rows = _query_all_entries()
    prefix_set = set()
    question_set = set()
    edges = []

    for prefix_key, qid, _, step, _ in rows:
        try:
            prefix_tuple = json.loads(prefix_key)
            prefix_label = "/".join(prefix_tuple) if isinstance(prefix_tuple, list) else str(prefix_tuple)
        except (json.JSONDecodeError, TypeError):
            prefix_label = str(prefix_key)

        pid = f"p:{prefix_label}"
        qnode = f"q:{qid}"
        prefix_set.add((pid, prefix_label, len(prefix_tuple) if isinstance(prefix_tuple, list) else 0))
        question_set.add(qnode)
        edges.append({"source": pid, "target": qnode, "step": step})

    nodes = []
    for pid, label, depth in prefix_set:
        nodes.append({"id": pid, "label": label, "type": "prefix", "depth": depth})
    for qnode in question_set:
        nodes.append({"id": qnode, "label": qnode[2:], "type": "question", "depth": 0})

    MAX_NODES = 200
    if len(nodes) > MAX_NODES:
        prefix_nodes = [n for n in nodes if n["type"] == "prefix"][:MAX_NODES // 2]
        q_nodes = [n for n in nodes if n["type"] == "question"][:MAX_NODES // 2]
        node_ids = {n["id"] for n in prefix_nodes + q_nodes}
        nodes = prefix_nodes + q_nodes
        edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/heatmap")
def api_heatmap():
    rows = _query_all_entries()
    depth_questions: dict[int, set] = defaultdict(set)

    for prefix_key, qid, _, _, _ in rows:
        try:
            prefix_tuple = json.loads(prefix_key)
            depth = len(prefix_tuple) if isinstance(prefix_tuple, list) else 0
        except (json.JSONDecodeError, TypeError):
            depth = 0
        depth_questions[depth].add(qid)

    heatmap = {}
    for depth in range(1, 6):
        heatmap[f"depth_{depth}"] = len(depth_questions.get(depth, set()))

    all_questions = set()
    for qs in depth_questions.values():
        all_questions |= qs

    return jsonify({
        "depths": heatmap,
        "total_unique_questions": len(all_questions),
    })


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OminiRAG Cache Visualizer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1d29; padding: 16px 24px; border-bottom: 1px solid #2a2d3a; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .status { font-size: 13px; color: #8b8fa3; }
  .header .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4caf50; margin-right: 6px; }
  .dashboard { display: grid; grid-template-columns: 280px 1fr; grid-template-rows: auto 1fr; gap: 16px; padding: 16px; height: calc(100vh - 60px); }
  .stats-panel { grid-row: 1 / 3; background: #1a1d29; border-radius: 8px; padding: 20px; border: 1px solid #2a2d3a; }
  .stat-card { margin-bottom: 20px; }
  .stat-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #8b8fa3; margin-bottom: 4px; }
  .stat-card .value { font-size: 28px; font-weight: 700; color: #fff; }
  .heatmap-section { margin-top: 24px; }
  .heatmap-section h3 { font-size: 13px; color: #8b8fa3; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .heatbar { display: flex; align-items: center; margin-bottom: 6px; font-size: 12px; }
  .heatbar .depth-label { width: 60px; color: #8b8fa3; }
  .heatbar .bar { height: 18px; border-radius: 3px; min-width: 2px; transition: width 0.5s ease; }
  .heatbar .bar-val { margin-left: 8px; font-size: 11px; color: #8b8fa3; }
  .d1 { background: #1e88e5; } .d2 { background: #43a047; } .d3 { background: #fb8c00; } .d4 { background: #e53935; } .d5 { background: #8e24aa; }
  .graph-panel { background: #1a1d29; border-radius: 8px; border: 1px solid #2a2d3a; overflow: hidden; position: relative; }
  .graph-panel h3 { position: absolute; top: 12px; left: 16px; font-size: 13px; color: #8b8fa3; z-index: 1; }
  #graph-svg { width: 100%; height: 100%; }
  .node-prefix { fill: #42a5f5; }
  .node-question { fill: #66bb6a; }
  .edge { stroke: #3a3d4a; stroke-width: 0.5; opacity: 0.4; }
  .timeline-panel { background: #1a1d29; border-radius: 8px; padding: 16px; border: 1px solid #2a2d3a; }
  .timeline-panel h3 { font-size: 13px; color: #8b8fa3; margin-bottom: 10px; }
  #timeline-canvas { width: 100%; height: 100px; }
  .legend { display: flex; gap: 16px; margin-top: 10px; font-size: 11px; }
  .legend span { display: flex; align-items: center; gap: 4px; }
  .legend .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
</style>
</head>
<body>
<div class="header">
  <h1>OminiRAG Bipartite Cache Visualizer</h1>
  <div class="status"><span class="dot"></span>Auto-refresh: <span id="last-update">--</span></div>
</div>
<div class="dashboard">
  <div class="stats-panel">
    <div class="stat-card"><div class="label">Total Entries</div><div class="value" id="stat-total">--</div></div>
    <div class="stat-card"><div class="label">Config Prefixes</div><div class="value" id="stat-prefixes">--</div></div>
    <div class="stat-card"><div class="label">Questions</div><div class="value" id="stat-questions">--</div></div>
    <div class="heatmap-section">
      <h3>Prefix Depth Coverage</h3>
      <div id="heatmap-bars"></div>
    </div>
    <div class="legend" style="margin-top:24px;">
      <span><span class="swatch" style="background:#42a5f5"></span> Config prefix</span>
      <span><span class="swatch" style="background:#66bb6a"></span> Question</span>
    </div>
  </div>
  <div class="graph-panel">
    <h3>Bipartite Reuse Graph</h3>
    <svg id="graph-svg"></svg>
  </div>
</div>

<script>
const POLL_MS = 2000;
const depthColors = {1:'#1e88e5',2:'#43a047',3:'#fb8c00',4:'#e53935',5:'#8e24aa'};
const depthNames = {1:'Chunking',2:'Query',3:'Retrieval',4:'PostRetr',5:'Generation'};

async function fetchStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('stat-total').textContent = d.total_entries;
    document.getElementById('stat-prefixes').textContent = d.unique_prefixes;
    document.getElementById('stat-questions').textContent = d.unique_questions;
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
  } catch(e) {}
}

async function fetchHeatmap() {
  try {
    const r = await fetch('/api/heatmap');
    const d = await r.json();
    const maxVal = Math.max(1, ...Object.values(d.depths));
    let html = '';
    for (let i = 1; i <= 5; i++) {
      const k = 'depth_' + i;
      const v = d.depths[k] || 0;
      const pct = Math.round(v / maxVal * 100);
      html += `<div class="heatbar"><span class="depth-label">D${i} ${depthNames[i]||''}</span><div class="bar d${i}" style="width:${Math.max(pct,2)}%"></div><span class="bar-val">${v}</span></div>`;
    }
    document.getElementById('heatmap-bars').innerHTML = html;
  } catch(e) {}
}

let simulation = null;
async function fetchGraph() {
  try {
    const r = await fetch('/api/graph');
    const data = await r.json();
    renderGraph(data);
  } catch(e) {}
}

function renderGraph(data) {
  const svg = document.getElementById('graph-svg');
  const rect = svg.getBoundingClientRect();
  const w = rect.width || 800, h = rect.height || 600;
  svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

  if (!data.nodes.length) {
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', w/2); t.setAttribute('y', h/2);
    t.setAttribute('text-anchor','middle'); t.setAttribute('fill','#8b8fa3');
    t.textContent = 'No cache entries yet...';
    svg.appendChild(t);
    return;
  }

  const nodeMap = {};
  data.nodes.forEach((n,i) => {
    n.x = n.type === 'prefix' ? w * 0.25 : w * 0.75;
    n.y = h * (i + 1) / (data.nodes.length + 1);
    nodeMap[n.id] = n;
  });

  const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg','g');
  data.edges.forEach(e => {
    const src = nodeMap[e.source], tgt = nodeMap[e.target];
    if (!src || !tgt) return;
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1', src.x); line.setAttribute('y1', src.y);
    line.setAttribute('x2', tgt.x); line.setAttribute('y2', tgt.y);
    line.setAttribute('class', 'edge');
    edgeGroup.appendChild(line);
  });
  svg.appendChild(edgeGroup);

  const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg','g');
  data.nodes.forEach(n => {
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', n.x); c.setAttribute('cy', n.y); c.setAttribute('r', 4);
    c.setAttribute('class', n.type === 'prefix' ? 'node-prefix' : 'node-question');
    const title = document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent = n.label;
    c.appendChild(title);
    nodeGroup.appendChild(c);
  });
  svg.appendChild(nodeGroup);
}

function poll() {
  fetchStats();
  fetchHeatmap();
  fetchGraph();
}
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>"""


def main():
    global DB_PATH
    parser = argparse.ArgumentParser(description="ReuseLedger cache visualizer")
    parser.add_argument("--db", type=str, required=True,
                        help="Path to reuse_ledger.db SQLite file")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    DB_PATH = args.db
    if not Path(DB_PATH).exists():
        print(f"WARNING: DB file {DB_PATH} does not exist yet. "
              "Will show empty dashboard until UCT search creates it.", flush=True)

    print(f"Starting cache visualizer on http://{args.host}:{args.port}", flush=True)
    print(f"  DB: {DB_PATH}", flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
