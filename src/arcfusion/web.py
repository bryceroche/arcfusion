"""
Web UI for ArcFusion - Explore components, engines, and dream architectures.

Usage:
    pip install arcfusion[web]
    arcfusion web  # or: uvicorn arcfusion.web:app --reload

Provides:
    - REST API for components, engines, relationships
    - Interactive component browser
    - Graph visualization of relationships
    - Dream interface for creating new architectures
"""

from pathlib import Path
from typing import Optional

# Check for FastAPI availability
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None

from .db import ArcFusionDB
from .composer import EngineComposer, get_component_category

# Default database path
DEFAULT_DB_PATH = "arcfusion.db"


def create_app(db_path: str = DEFAULT_DB_PATH) -> "FastAPI":
    """Create FastAPI application with ArcFusion endpoints."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Install with: pip install 'arcfusion[web]'")

    app = FastAPI(
        title="ArcFusion",
        description="ML Architecture Component Database & Composer",
        version="0.2.0",
    )

    # Database connection
    db = ArcFusionDB(db_path)

    # =========================================================================
    # API ENDPOINTS
    # =========================================================================

    @app.get("/api/stats")
    def get_stats():
        """Get database statistics."""
        return db.stats()

    @app.get("/api/components")
    def list_components(
        search: Optional[str] = Query(None, description="Search term"),
        category: Optional[str] = Query(None, description="Filter by category"),
        limit: int = Query(100, le=500),
    ):
        """List all components with optional filtering."""
        components = db.find_components(search) if search else db.find_components()

        results = []
        for comp in components[:limit]:
            cat = get_component_category(comp)
            if category and cat != category:
                continue
            results.append({
                "id": comp.component_id,
                "name": comp.name,
                "description": comp.description[:200] if comp.description else "",
                "category": cat,
                "score": comp.usefulness_score,
                "has_code": bool(comp.code and comp.code.strip()),
                "source_paper": comp.source_paper_id,
                "year": comp.introduced_year,
            })

        return {"components": results, "total": len(results)}

    @app.get("/api/components/{component_id}")
    def get_component(component_id: str):
        """Get detailed component information."""
        comp = db.get_component(component_id)
        if not comp:
            raise HTTPException(status_code=404, detail="Component not found")

        # Get compatible components
        compatible = db.get_compatible_components(component_id, min_score=0.5)

        return {
            "id": comp.component_id,
            "name": comp.name,
            "description": comp.description,
            "category": get_component_category(comp),
            "score": comp.usefulness_score,
            "code": comp.code,
            "interface_in": comp.interface_in,
            "interface_out": comp.interface_out,
            "hyperparameters": comp.hyperparameters,
            "time_complexity": comp.time_complexity,
            "space_complexity": comp.space_complexity,
            "source_paper": comp.source_paper_id,
            "year": comp.introduced_year,
            "is_parallelizable": comp.is_parallelizable,
            "is_causal": comp.is_causal,
            "compatible": [
                {"id": cid, "score": score}
                for cid, score in compatible[:10]
            ],
        }

    @app.get("/api/engines")
    def list_engines(limit: int = Query(50, le=200)):
        """List all engines/architectures."""
        engines = db.list_engines()

        results = []
        for engine in engines[:limit]:
            results.append({
                "id": engine.engine_id,
                "name": engine.name,
                "description": engine.description[:200] if engine.description else "",
                "score": engine.engine_score,
                "component_count": len(engine.component_ids),
                "paper_url": engine.paper_url,
            })

        return {"engines": results, "total": len(results)}

    @app.get("/api/engines/{engine_id}")
    def get_engine(engine_id: str):
        """Get detailed engine information."""
        engine = db.get_engine(engine_id)
        if not engine:
            raise HTTPException(status_code=404, detail="Engine not found")

        # Get component details
        components = []
        for cid in engine.component_ids:
            comp = db.get_component(cid)
            if comp:
                components.append({
                    "id": comp.component_id,
                    "name": comp.name,
                    "category": get_component_category(comp),
                })

        return {
            "id": engine.engine_id,
            "name": engine.name,
            "description": engine.description,
            "score": engine.engine_score,
            "paper_url": engine.paper_url,
            "components": components,
        }

    @app.get("/api/relationships")
    def get_relationships(min_score: float = Query(0.7, ge=0, le=1)):
        """Get component relationships for graph visualization."""
        # Get all components first
        components = db.find_components()
        comp_map = {c.component_id: c.name for c in components}

        # Build edges from relationships
        edges = []
        seen = set()

        for comp in components:
            compatible = db.get_compatible_components(comp.component_id, min_score=min_score)
            for cid, score in compatible:
                # Avoid duplicates
                edge_key = tuple(sorted([comp.component_id, cid]))
                if edge_key in seen:
                    continue
                seen.add(edge_key)

                if cid in comp_map:
                    edges.append({
                        "source": comp.component_id,
                        "target": cid,
                        "score": score,
                    })

        # Build nodes
        nodes = [
            {
                "id": c.component_id,
                "name": c.name,
                "category": get_component_category(c),
                "score": c.usefulness_score,
            }
            for c in components
        ]

        return {"nodes": nodes, "edges": edges}

    @app.post("/api/dream")
    def dream_architecture(
        strategy: str = Query("greedy", description="Dream strategy"),
        name: str = Query("DreamedArch", description="Architecture name"),
        temperature: float = Query(1.0, ge=0.1, le=3.0),
        top_k: int = Query(5, ge=1, le=20),
    ):
        """Dream a new architecture using the composer."""
        composer = EngineComposer(db)

        try:
            if strategy == "greedy":
                components, score = composer.dream("greedy", temperature=temperature, top_k=top_k)
            elif strategy == "random":
                components, score = composer.dream("random", steps=8, temperature=temperature)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

            return {
                "name": name,
                "strategy": strategy,
                "score": score,
                "components": [
                    {
                        "id": c.component_id,
                        "name": c.name,
                        "category": get_component_category(c),
                    }
                    for c in components
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # HTML FRONTEND
    # =========================================================================

    @app.get("/", response_class=HTMLResponse)
    def home():
        """Serve the main web interface."""
        return HTML_TEMPLATE

    return app


# Standalone HTML template with embedded CSS and JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArcFusion - ML Architecture Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/vis-network@9/dist/vis-network.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            line-height: 1.5;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #30363d;
            margin-bottom: 20px;
        }
        h1 { color: #58a6ff; font-size: 1.5rem; }
        .stats { display: flex; gap: 20px; }
        .stat { text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #58a6ff; }
        .stat-label { font-size: 0.75rem; color: #8b949e; }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            cursor: pointer;
            color: #c9d1d9;
        }
        .tab:hover { background: #30363d; }
        .tab.active { background: #238636; border-color: #238636; }

        .panel { display: none; }
        .panel.active { display: block; }

        .search-box {
            width: 100%;
            padding: 12px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 1rem;
            margin-bottom: 20px;
        }
        .search-box:focus { outline: none; border-color: #58a6ff; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .card:hover { border-color: #58a6ff; }
        .card-title { font-weight: 600; color: #58a6ff; margin-bottom: 5px; }
        .card-desc { font-size: 0.85rem; color: #8b949e; margin-bottom: 10px; }
        .card-meta { display: flex; gap: 10px; font-size: 0.75rem; }
        .badge {
            padding: 2px 8px;
            background: #30363d;
            border-radius: 12px;
            font-size: 0.7rem;
        }
        .badge.attention { background: #1f6feb; }
        .badge.layer { background: #238636; }
        .badge.embedding { background: #a371f7; }
        .badge.structure { background: #f85149; }
        .badge.efficiency { background: #f0883e; }

        #graph {
            height: 600px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
        }

        .dream-controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .dream-controls select, .dream-controls input, .dream-controls button {
            padding: 10px 15px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
        }
        .dream-controls button {
            background: #238636;
            cursor: pointer;
        }
        .dream-controls button:hover { background: #2ea043; }

        .dream-result {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .dream-result h3 { color: #58a6ff; margin-bottom: 15px; }
        .component-flow {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }
        .component-flow .comp {
            padding: 8px 15px;
            background: #21262d;
            border-radius: 6px;
        }
        .component-flow .arrow { color: #8b949e; }

        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            padding: 50px;
            overflow: auto;
        }
        .modal.active { display: block; }
        .modal-content {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            max-width: 800px;
            margin: 0 auto;
            padding: 30px;
        }
        .modal-close {
            float: right;
            cursor: pointer;
            font-size: 1.5rem;
            color: #8b949e;
        }
        .code-block {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
            font-family: monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            white-space: pre;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 ArcFusion</h1>
            <div class="stats" id="stats"></div>
        </header>

        <div class="tabs">
            <div class="tab active" onclick="showPanel('components')">Components</div>
            <div class="tab" onclick="showPanel('engines')">Engines</div>
            <div class="tab" onclick="showPanel('graph')">Graph</div>
            <div class="tab" onclick="showPanel('dream')">Dream</div>
        </div>

        <div id="components" class="panel active">
            <input type="text" class="search-box" placeholder="Search components..." oninput="searchComponents(this.value)">
            <div class="grid" id="components-grid"></div>
        </div>

        <div id="engines" class="panel">
            <div class="grid" id="engines-grid"></div>
        </div>

        <div id="graph" class="panel">
            <div id="network"></div>
        </div>

        <div id="dream" class="panel">
            <div class="dream-controls">
                <select id="strategy">
                    <option value="greedy">Greedy (best components)</option>
                    <option value="random">Random Walk</option>
                </select>
                <input type="number" id="temperature" value="1.0" min="0.1" max="3" step="0.1" style="width:100px" title="Temperature">
                <button onclick="dreamArchitecture()">🌟 Dream New Architecture</button>
            </div>
            <div id="dream-result"></div>
        </div>
    </div>

    <div class="modal" id="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        let allComponents = [];
        let network = null;

        async function loadStats() {
            const resp = await fetch('/api/stats');
            const data = await resp.json();
            document.getElementById('stats').innerHTML = `
                <div class="stat"><div class="stat-value">${data.components}</div><div class="stat-label">Components</div></div>
                <div class="stat"><div class="stat-value">${data.engines}</div><div class="stat-label">Engines</div></div>
                <div class="stat"><div class="stat-value">${data.relationships}</div><div class="stat-label">Relationships</div></div>
            `;
        }

        async function loadComponents() {
            const resp = await fetch('/api/components?limit=200');
            const data = await resp.json();
            allComponents = data.components;
            renderComponents(allComponents);
        }

        function renderComponents(components) {
            document.getElementById('components-grid').innerHTML = components.map(c => `
                <div class="card" onclick="showComponent('${c.id}')">
                    <div class="card-title">${c.name}</div>
                    <div class="card-desc">${c.description || 'No description'}</div>
                    <div class="card-meta">
                        <span class="badge ${c.category}">${c.category}</span>
                        <span>Score: ${c.score.toFixed(2)}</span>
                        ${c.has_code ? '<span>✓ Code</span>' : ''}
                    </div>
                </div>
            `).join('');
        }

        function searchComponents(query) {
            const filtered = allComponents.filter(c =>
                c.name.toLowerCase().includes(query.toLowerCase()) ||
                (c.description && c.description.toLowerCase().includes(query.toLowerCase()))
            );
            renderComponents(filtered);
        }

        async function loadEngines() {
            const resp = await fetch('/api/engines');
            const data = await resp.json();
            document.getElementById('engines-grid').innerHTML = data.engines.map(e => `
                <div class="card" onclick="showEngine('${e.id}')">
                    <div class="card-title">${e.name}</div>
                    <div class="card-desc">${e.description || 'No description'}</div>
                    <div class="card-meta">
                        <span>Score: ${e.score.toFixed(2)}</span>
                        <span>${e.component_count} components</span>
                    </div>
                </div>
            `).join('');
        }

        async function loadGraph() {
            const resp = await fetch('/api/relationships?min_score=0.8');
            const data = await resp.json();

            const categoryColors = {
                attention: '#1f6feb',
                layer: '#238636',
                embedding: '#a371f7',
                structure: '#f85149',
                efficiency: '#f0883e',
                position: '#db61a2',
                output: '#8b949e',
                training: '#f0883e',
            };

            const nodes = new vis.DataSet(data.nodes.map(n => ({
                id: n.id,
                label: n.name,
                color: categoryColors[n.category] || '#8b949e',
                title: `${n.name} (${n.category})`,
            })));

            const edges = new vis.DataSet(data.edges.map(e => ({
                from: e.source,
                to: e.target,
                value: e.score,
                title: `Score: ${e.score.toFixed(2)}`,
            })));

            const container = document.getElementById('network');
            container.style.height = '600px';

            network = new vis.Network(container, { nodes, edges }, {
                nodes: { shape: 'dot', size: 16, font: { color: '#c9d1d9' } },
                edges: { color: { color: '#30363d', highlight: '#58a6ff' }, smooth: true },
                physics: { stabilization: { iterations: 100 } },
            });
        }

        async function showComponent(id) {
            const resp = await fetch(`/api/components/${id}`);
            const c = await resp.json();

            document.getElementById('modal-body').innerHTML = `
                <h2>${c.name}</h2>
                <p><span class="badge ${c.category}">${c.category}</span> Score: ${c.score.toFixed(2)}</p>
                <p>${c.description || ''}</p>
                ${c.code ? `<h4>Code</h4><div class="code-block">${escapeHtml(c.code)}</div>` : ''}
                <h4>Interface</h4>
                <p>Input: ${JSON.stringify(c.interface_in)}</p>
                <p>Output: ${JSON.stringify(c.interface_out)}</p>
                ${c.compatible.length ? `<h4>Compatible With</h4><p>${c.compatible.map(x => x.id.slice(0,8)).join(', ')}</p>` : ''}
            `;
            document.getElementById('modal').classList.add('active');
        }

        async function showEngine(id) {
            const resp = await fetch(`/api/engines/${id}`);
            const e = await resp.json();

            document.getElementById('modal-body').innerHTML = `
                <h2>${e.name}</h2>
                <p>Score: ${e.score.toFixed(2)}</p>
                <p>${e.description || ''}</p>
                ${e.paper_url ? `<p><a href="${e.paper_url}" target="_blank" style="color:#58a6ff">Paper</a></p>` : ''}
                <h4>Components (${e.components.length})</h4>
                <div class="component-flow">
                    ${e.components.map((c, i) =>
                        `<span class="comp badge ${c.category}">${c.name}</span>${i < e.components.length - 1 ? '<span class="arrow">→</span>' : ''}`
                    ).join('')}
                </div>
            `;
            document.getElementById('modal').classList.add('active');
        }

        async function dreamArchitecture() {
            const strategy = document.getElementById('strategy').value;
            const temperature = document.getElementById('temperature').value;

            const resp = await fetch(`/api/dream?strategy=${strategy}&temperature=${temperature}`, { method: 'POST' });
            const result = await resp.json();

            if (result.error) {
                document.getElementById('dream-result').innerHTML = `<p style="color:#f85149">Error: ${result.detail}</p>`;
                return;
            }

            document.getElementById('dream-result').innerHTML = `
                <div class="dream-result">
                    <h3>🌟 Dreamed Architecture (Score: ${result.score.toFixed(2)})</h3>
                    <p>Strategy: ${result.strategy}</p>
                    <div class="component-flow">
                        ${result.components.map((c, i) =>
                            `<span class="comp badge ${c.category}">${c.name}</span>${i < result.components.length - 1 ? '<span class="arrow">→</span>' : ''}`
                        ).join('')}
                    </div>
                </div>
            `;
        }

        function showPanel(name) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(name).classList.add('active');
            event.target.classList.add('active');

            if (name === 'graph' && !network) loadGraph();
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize
        loadStats();
        loadComponents();
        loadEngines();
    </script>
</body>
</html>
"""


# Create default app instance
app = create_app() if HAS_FASTAPI else None


def run_server(host: str = "127.0.0.1", port: int = 8000, db_path: str = DEFAULT_DB_PATH):
    """Run the web server."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Install with: pip install 'arcfusion[web]'")

    import uvicorn
    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port)
