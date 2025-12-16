#!/usr/bin/env python3
"""ArcFusion Web UI - Database Visualization Dashboard."""

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
    ["📊 Overview", "🧩 Components", "🏗️ Engines", "🏆 Leaderboard", "💭 Dream Candidates", "📈 Surrogate Model"]
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

    tab1, tab2 = st.tabs(["📉 By Perplexity", "⚡ By Efficiency"])

    with tab1:
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

    with tab2:
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
            data = [{
                'Strategy': c.strategy,
                'Temp': f"{c.temperature:.2f}",
                'Predicted PPL': f"{c.predicted_ppl:.1f}" if c.predicted_ppl else 'N/A',
                'Arch Type': c.arch_type,
                'Layers': c.n_layers,
                'Has Mamba': '✅' if c.has_mamba else '❌',
                'Is Hybrid': '✅' if c.is_hybrid else '❌',
            } for c in candidates]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.write(f"Total: {len(candidates)} untrained candidates")
        else:
            st.success("All candidates have been trained!")

    with tab2:
        st.subheader("Trained Candidates (with results)")
        trained = db.list_dream_candidates(trained_only=True, limit=50)
        if trained:
            data = [{
                'Strategy': c.strategy,
                'Predicted PPL': f"{c.predicted_ppl:.1f}" if c.predicted_ppl else 'N/A',
                'Actual PPL': f"{c.actual_ppl:.1f}" if c.actual_ppl else 'N/A',
                'Error': f"{abs(c.predicted_ppl - c.actual_ppl):.1f}" if c.predicted_ppl and c.actual_ppl else 'N/A',
                'Actual Time': f"{c.actual_time:.1f}s" if c.actual_time else 'N/A',
                'Arch Type': c.arch_type,
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


# Footer
st.sidebar.divider()
st.sidebar.caption("ArcFusion v0.1 - ML Architecture Research")
