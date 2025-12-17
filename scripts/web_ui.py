#!/usr/bin/env python3
"""ArcFusion Web UI - Database Visualization Dashboard."""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from arcfusion.db import ArcFusionDB

# Page config
st.set_page_config(
    page_title="ArcFusion Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection
@st.cache_resource
def get_db():
    db_path = Path(__file__).parent.parent / "arcfusion.db"
    return ArcFusionDB(str(db_path), check_same_thread=False)

db = get_db()

# Sidebar navigation
st.sidebar.title("🧬 ArcFusion")
st.sidebar.markdown("ML Architecture Component Database")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🧩 Components", "🏗️ Engines", "🏆 Leaderboard", "💭 Dream Candidates", "📈 Surrogate Model", "📚 Findings", "📋 Issues"]
)

# Helper functions
def get_component_category(name: str) -> str:
    """Infer category from component name."""
    name_lower = name.lower()
    if any(x in name_lower for x in ['attention', 'query', 'key', 'value']):
        return 'attention'
    if any(x in name_lower for x in ['mamba', 'ssm', 's4', 'state space', 'selective']):
        return 'ssm'
    if any(x in name_lower for x in ['position', 'rotary', 'rope', 'sinusoidal', 'alibi']):
        return 'position'
    if any(x in name_lower for x in ['norm', 'layer norm', 'rms']):
        return 'normalization'
    if any(x in name_lower for x in ['ffn', 'feed forward', 'mlp', 'swiglu']):
        return 'ffn'
    if any(x in name_lower for x in ['embed', 'token']):
        return 'embedding'
    if any(x in name_lower for x in ['flash', 'efficient', 'sparse', 'linear']):
        return 'efficiency'
    if any(x in name_lower for x in ['moe', 'mixture', 'expert', 'router']):
        return 'moe'
    if any(x in name_lower for x in ['hyena', 'xlstm', 'lstm', 'gru', 'conv', 'rwkv']):
        return 'alternative'
    if any(x in name_lower for x in ['gelu', 'relu', 'silu', 'activation']):
        return 'activation'
    if any(x in name_lower for x in ['dropout', 'residual']):
        return 'regularization'
    return 'other'


# =============================================================================
# Overview Page
# =============================================================================
if page == "📊 Overview":
    st.title("📊 ArcFusion Overview")

    # Get stats
    stats = db.stats()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Components", stats.get('components', 0))
    with col2:
        st.metric("Engines", stats.get('engines', 0))
    with col3:
        st.metric("Training Runs", stats.get('training_runs', 0))
    with col4:
        st.metric("Dream Candidates", stats.get('dream_candidates', 0))

    st.divider()

    # Component category distribution
    st.subheader("Component Categories")
    components = db.find_components()
    if components:
        categories = [get_component_category(c.name) for c in components]
        cat_counts = pd.Series(categories).value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                     title="Component Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)

    # Training runs over time
    st.subheader("Recent Training Results")
    runs = db.get_training_leaderboard(limit=20)
    if runs:
        run_data = [{
            'name': r.model_name,
            'ppl': r.perplexity,
            'time': r.time_seconds,
            'date': r.created_at[:10] if r.created_at else 'Unknown'
        } for r in runs]
        df = pd.DataFrame(run_data)
        fig = px.scatter(df, x='time', y='ppl', hover_name='name',
                        title="PPL vs Training Time",
                        labels={'time': 'Time (seconds)', 'ppl': 'Perplexity'})
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Components Page
# =============================================================================
elif page == "🧩 Components":
    st.title("🧩 Component Browser")

    # Search and filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search components", placeholder="e.g., attention, mamba, ffn...")
    with col2:
        min_score = st.slider("Min usefulness score", 0.0, 1.0, 0.0, 0.1)

    # Get components
    components = db.find_components(name_pattern=search if search else None, min_score=min_score)

    # Add category info
    comp_data = []
    for c in components:
        comp_data.append({
            'Name': c.name,
            'Category': get_component_category(c.name),
            'Score': c.usefulness_score or 0,
            'Description': (c.description or '')[:100] + '...' if c.description and len(c.description) > 100 else c.description
        })

    df = pd.DataFrame(comp_data)

    # Category filter
    if not df.empty:
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_cat = st.selectbox("Filter by category", categories)

        if selected_cat != 'All':
            df = df[df['Category'] == selected_cat]

        st.write(f"Found {len(df)} components")

        # Display table
        st.dataframe(
            df,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                )
            },
            use_container_width=True
        )

        # Category breakdown chart
        cat_counts = df['Category'].value_counts()
        fig = px.bar(x=cat_counts.index, y=cat_counts.values,
                    title="Components by Category",
                    labels={'x': 'Category', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Engines Page
# =============================================================================
elif page == "🏗️ Engines":
    st.title("🏗️ Engine/Architecture Browser")

    # Get all engines
    engines = db.list_engines()

    if engines:
        st.write(f"Found {len(engines)} engines")

        for engine in engines:
            with st.expander(f"🏗️ {engine.name} (Score: {engine.engine_score or 'N/A'})"):
                st.write(f"**Description:** {engine.description or 'No description'}")
                st.write(f"**Components:** {len(engine.component_ids)} components")

                # List component names
                comp_names = []
                for cid in engine.component_ids[:10]:  # Show first 10
                    comp = db.get_component(cid)
                    if comp:
                        comp_names.append(comp.name)
                if comp_names:
                    st.write("**Component list:**")
                    st.write(", ".join(comp_names))
                    if len(engine.component_ids) > 10:
                        st.write(f"... and {len(engine.component_ids) - 10} more")
    else:
        st.info("No engines found in database")


# =============================================================================
# Leaderboard Page
# =============================================================================
elif page == "🏆 Leaderboard":
    st.title("🏆 Training Leaderboard")

    tab1, tab2 = st.tabs(["⚡ By Efficiency", "📉 By Perplexity"])

    with tab1:
        st.subheader("Top Models by Efficiency (lower is better)")
        st.caption("Efficiency = PPL × √(time/300s) — balances quality and speed")

        eff_runs = db.get_efficiency_leaderboard(limit=20)
        if eff_runs:
            data = [{
                'Rank': i+1,
                'Model': r.model_name,
                'Efficiency': f"{eff:.1f}",
                'PPL': f"{r.perplexity:.1f}",
                'Time (s)': f"{r.time_seconds:.1f}",
            } for i, (r, eff) in enumerate(eff_runs)]
            st.dataframe(pd.DataFrame(data), use_container_width=True)

            # Scatter plot
            scatter_data = [{
                'Model': r.model_name,
                'PPL': r.perplexity,
                'Time': r.time_seconds,
                'Efficiency': eff
            } for r, eff in eff_runs]
            df = pd.DataFrame(scatter_data)
            fig = px.scatter(df, x='Time', y='PPL', size='Efficiency',
                           hover_name='Model', color='Efficiency',
                           title="PPL vs Time (bubble size = efficiency score)",
                           color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training runs found")

    with tab2:
        st.subheader("Top Models by Perplexity (lower is better)")
        runs = db.get_training_leaderboard(limit=20)
        if runs:
            data = [{
                'Rank': i+1,
                'Model': r.model_name,
                'PPL': f"{r.perplexity:.1f}",
                'Time (s)': f"{r.time_seconds:.1f}",
                'Date': r.created_at[:10] if r.created_at else 'Unknown'
            } for i, r in enumerate(runs)]
            st.dataframe(pd.DataFrame(data), use_container_width=True)

            # Chart
            chart_data = [{'Model': r.model_name[:30], 'PPL': r.perplexity} for r in runs[:15]]
            fig = px.bar(pd.DataFrame(chart_data), x='Model', y='PPL',
                        title="Top 15 Models by Perplexity")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training runs found")


# =============================================================================
# Dream Candidates Page
# =============================================================================
elif page == "💭 Dream Candidates":
    st.title("💭 Dream Candidates Pipeline")

    tab1, tab2 = st.tabs(["🆕 Untrained", "✅ Trained"])

    with tab1:
        st.subheader("Untrained Candidates (waiting for GPU)")
        candidates = db.list_dream_candidates(untrained_only=True, limit=50)
        if candidates:
            def parse_components(json_str):
                try:
                    comps = json.loads(json_str) if json_str else []
                    return ', '.join(comps[:5]) + ('...' if len(comps) > 5 else '')
                except:
                    return json_str or ''

            data = [{
                'Strategy': c.strategy,
                'Temp': f"{c.temperature:.2f}",
                'Predicted PPL': f"{c.predicted_ppl:.1f}" if c.predicted_ppl else 'N/A',
                'Arch Type': c.arch_type,
                'Layers': c.n_layers,
                'Components': parse_components(c.components_json),
            } for c in candidates]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.write(f"Total: {len(candidates)} untrained candidates")
        else:
            st.success("All candidates have been trained!")

    with tab2:
        st.subheader("Trained Candidates (with results)")
        trained = db.list_dream_candidates(trained_only=True, limit=50)
        if trained:
            def parse_components_trained(json_str):
                try:
                    comps = json.loads(json_str) if json_str else []
                    return ', '.join(comps[:5]) + ('...' if len(comps) > 5 else '')
                except:
                    return json_str or ''

            data = [{
                'Strategy': c.strategy,
                'Predicted PPL': f"{c.predicted_ppl:.1f}" if c.predicted_ppl else 'N/A',
                'Actual PPL': f"{c.actual_ppl:.1f}" if c.actual_ppl else 'N/A',
                'Error': f"{abs(c.predicted_ppl - c.actual_ppl):.1f}" if c.predicted_ppl and c.actual_ppl else 'N/A',
                'Actual Time': f"{c.actual_time:.1f}s" if c.actual_time else 'N/A',
                'Components': parse_components_trained(c.components_json),
            } for c in trained]
            st.dataframe(pd.DataFrame(data), use_container_width=True)

            # Prediction accuracy chart
            pred_actual = [(c.predicted_ppl, c.actual_ppl) for c in trained
                          if c.predicted_ppl and c.actual_ppl]
            if pred_actual:
                df = pd.DataFrame(pred_actual, columns=['Predicted', 'Actual'])
                fig = px.scatter(df, x='Predicted', y='Actual',
                               title="Surrogate Model: Predicted vs Actual PPL")
                # Add diagonal line
                max_val = max(df['Predicted'].max(), df['Actual'].max())
                fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                            line=dict(dash='dash', color='gray'))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trained candidates yet")


# =============================================================================
# Surrogate Model Page
# =============================================================================
elif page == "📈 Surrogate Model":
    st.title("📈 Surrogate Model Statistics")

    stats = db.get_surrogate_accuracy_stats()

    if stats.get('insufficient_data'):
        st.warning(f"Insufficient data for statistics. Need at least 2 trained candidates (have {stats.get('n_samples', 0)})")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", stats.get('n_samples', 0))
        with col2:
            st.metric("PPL MAE", f"{stats.get('ppl_mae', 0):.2f}")
        with col3:
            st.metric("PPL MAPE", f"{stats.get('ppl_mape', 0):.1f}%")

        col4, col5, col6 = st.columns(3)
        with col4:
            corr = stats.get('ppl_correlation')
            st.metric("PPL Correlation", f"{corr:.3f}" if corr else "N/A")
        with col5:
            st.metric("Time MAE", f"{stats.get('time_mae', 0):.1f}s" if stats.get('time_mae') else "N/A")
        with col6:
            st.metric("Time MAPE", f"{stats.get('time_mape', 0):.1f}%" if stats.get('time_mape') else "N/A")

        st.divider()

        # Interpretation
        st.subheader("Model Quality Assessment")
        ppl_mape = stats.get('ppl_mape', 100)
        if ppl_mape < 5:
            st.success("🎯 Excellent! PPL predictions are within 5% of actual values.")
        elif ppl_mape < 10:
            st.success("✅ Good! PPL predictions are within 10% of actual values.")
        elif ppl_mape < 20:
            st.warning("⚠️ Moderate. PPL predictions have ~20% error. Consider more training data.")
        else:
            st.error("❌ High error. Surrogate model needs more training data.")

        # Show trained candidates prediction accuracy
        trained = db.list_dream_candidates(trained_only=True, limit=100)
        if trained:
            pred_actual = [(c.predicted_ppl, c.actual_ppl, c.strategy) for c in trained
                          if c.predicted_ppl and c.actual_ppl]
            if pred_actual:
                df = pd.DataFrame(pred_actual, columns=['Predicted', 'Actual', 'Strategy'])
                fig = px.scatter(df, x='Predicted', y='Actual', color='Strategy',
                               title="Prediction Accuracy by Strategy")
                max_val = max(df['Predicted'].max(), df['Actual'].max())
                fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                            line=dict(dash='dash', color='gray'))
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Findings Page
# =============================================================================
elif page == "📚 Findings":
    st.title("📚 Research Findings")
    st.caption("Accumulated knowledge from architecture experiments")

    # Get all findings
    findings = db.list_findings(limit=100)

    if not findings:
        st.info("No findings recorded yet. Run experiments and add findings to build knowledge base.")
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        high_conf = sum(1 for f in findings if f.confidence == 'high')
        with_delta = [f for f in findings if f.delta_vs_baseline != 0]
        avg_delta = sum(f.delta_vs_baseline for f in with_delta) / len(with_delta) if with_delta else 0

        with col1:
            st.metric("Total Findings", len(findings))
        with col2:
            st.metric("High Confidence", high_conf)
        with col3:
            st.metric("Avg PPL Delta", f"{avg_delta:.1f}%")

        st.divider()

        # Get all unique tags
        all_tags = set()
        for f in findings:
            if isinstance(f.tags, list):
                all_tags.update(f.tags)

        # Filter controls
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_tags = st.multiselect("Filter by tags", sorted(all_tags), default=[])
        with col2:
            sort_by = st.selectbox("Sort by", ["Date (newest)", "Confidence", "PPL Delta"])

        # Filter findings
        filtered = findings
        if selected_tags:
            filtered = [f for f in findings
                       if isinstance(f.tags, list) and any(t in f.tags for t in selected_tags)]

        # Sort findings
        if sort_by == "Confidence":
            conf_order = {'high': 0, 'medium': 1, 'low': 2}
            filtered = sorted(filtered, key=lambda f: conf_order.get(f.confidence, 3))
        elif sort_by == "PPL Delta":
            filtered = sorted(filtered, key=lambda f: f.delta_vs_baseline)

        st.write(f"Showing {len(filtered)} findings")

        # Display findings
        for f in filtered:
            confidence_color = "green" if f.confidence == 'high' else "orange" if f.confidence == 'medium' else "red"
            delta_str = f"{f.delta_vs_baseline:+.1f}%" if f.delta_vs_baseline != 0 else ""
            delta_color = "green" if f.delta_vs_baseline < 0 else "red" if f.delta_vs_baseline > 0 else ""

            with st.expander(f"**{f.title}**"):
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    st.markdown(f"**Confidence:** :{confidence_color}[{f.confidence}]")
                with cols[1]:
                    if delta_str:
                        st.markdown(f"**PPL Delta:** :{delta_color}[{delta_str}]")
                with cols[2]:
                    if isinstance(f.tags, list) and f.tags:
                        st.markdown(f"**Tags:** {', '.join(f.tags)}")

                if f.description:
                    st.markdown(f.description)

                st.caption(f"Created: {f.created_at[:10] if f.created_at else 'Unknown'}")

        # Architecture patterns section
        st.divider()
        st.subheader("🎯 Active Architecture Patterns")
        st.caption("Recommendations derived from findings for architecture search")

        patterns = db.get_architecture_patterns()
        if patterns:
            for p in patterns:
                rec_display = p['recommendation'].replace('_', ' ').title()
                st.markdown(f"- **{rec_display}**: {p['pattern'][:80]}...")
        else:
            st.info("No actionable patterns derived yet")

        # Efficiency constraints
        st.subheader("⚡ Efficiency Constraints")
        constraints = db.get_efficiency_constraints()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Slowdown", f"{constraints['max_slowdown']}x")
            st.metric("Baseline Time", f"{constraints['baseline_time']:.0f}s")
        with col2:
            st.metric("Min PPL Gain (for slow)", f"{constraints['min_ppl_gain_for_slow']*100:.0f}%")
            st.metric("Baseline PPL", f"{constraints['baseline_ppl']:.0f}")


# =============================================================================
# Issues Page
# =============================================================================
elif page == "📋 Issues":
    st.title("📋 Project Issues (Beads)")

    # Load issues from .beads/issues.jsonl
    issues_path = Path(__file__).parent.parent / ".beads" / "issues.jsonl"
    issues = []
    if issues_path.exists():
        with open(issues_path) as f:
            for line in f:
                if line.strip():
                    issues.append(json.loads(line))

    if not issues:
        st.info("No issues found in .beads/issues.jsonl")
    else:
        # Separate by status
        open_issues = [i for i in issues if i.get('status') == 'open']
        in_progress = [i for i in issues if i.get('status') == 'in_progress']
        closed_issues = [i for i in issues if i.get('status') == 'closed']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Open", len(open_issues))
        with col2:
            st.metric("In Progress", len(in_progress))
        with col3:
            st.metric("Closed", len(closed_issues))
        with col4:
            st.metric("Total", len(issues))

        st.divider()

        # Tabs for different statuses
        tab1, tab2, tab3 = st.tabs(["🔴 Open", "🟡 In Progress", "✅ Closed"])

        def render_issue(issue):
            priority = issue.get('priority', 3)
            priority_badge = "🔥" if priority == 1 else "⚡" if priority == 2 else ""
            issue_type = issue.get('issue_type', 'task')
            type_color = "blue" if issue_type == 'feature' else "green" if issue_type == 'task' else "red"

            with st.expander(f"{priority_badge} **{issue['id']}**: {issue['title']}"):
                st.caption(f"Type: :{type_color}[{issue_type}] | Priority: P{priority}")
                if issue.get('description'):
                    st.markdown(issue['description'][:500] + ('...' if len(issue.get('description', '')) > 500 else ''))
                st.caption(f"Created: {issue.get('created_at', 'Unknown')[:10]}")

        with tab1:
            if open_issues:
                for issue in sorted(open_issues, key=lambda x: x.get('priority', 3)):
                    render_issue(issue)
            else:
                st.success("No open issues!")

        with tab2:
            if in_progress:
                for issue in in_progress:
                    render_issue(issue)
            else:
                st.info("No issues in progress")

        with tab3:
            st.caption(f"Showing last 20 of {len(closed_issues)} closed issues")
            for issue in sorted(closed_issues, key=lambda x: x.get('closed_at', ''), reverse=True)[:20]:
                render_issue(issue)


# Footer
st.sidebar.divider()
st.sidebar.caption("ArcFusion v0.1 - ML Architecture Research")
