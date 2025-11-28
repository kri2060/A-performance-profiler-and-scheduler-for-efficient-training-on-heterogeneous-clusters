"""
Professional Monitoring Dashboard
Real-time visualization of heterogeneous cluster training
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Professional styling
st.set_page_config(
    page_title="Heterogeneous Cluster Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    h3 {
        color: #4b5563;
        font-weight: 500;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-active {
        background-color: #10b981;
        color: white;
    }
    .status-inactive {
        background-color: #ef4444;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme
COLORS = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'dark': '#1f2937',
    'light': '#f3f4f6'
}


def load_metrics(metrics_dir: str = "experiments/logs"):
    """Load metrics from JSON files"""
    metrics_path = Path(metrics_dir)

    if not metrics_path.exists():
        return {}

    all_metrics = {}

    # Load metrics for each rank
    for metrics_file in metrics_path.glob("rank_*_metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                rank = int(metrics_file.stem.split('_')[1])
                all_metrics[rank] = data
        except Exception as e:
            st.error(f"Error loading {metrics_file}: {e}")

    return all_metrics


def load_gpu_profiles(config_dir: str = "experiments/configs"):
    """Load GPU profiles"""
    profile_path = Path(config_dir) / "gpu_profiles.json"

    if not profile_path.exists():
        return []

    try:
        with open(profile_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GPU profiles: {e}")
        return []


def plot_gpu_utilization(metrics_dict):
    """Plot GPU utilization over time with professional styling"""
    fig = go.Figure()

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]

    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['gpu_utilization'],
                mode='lines',
                name=f'Worker {rank}',
                line=dict(width=3, color=colors[idx % len(colors)]),
                hovertemplate='<b>Worker %{fullData.name}</b><br>Iteration: %{x}<br>Utilization: %{y:.1f}%<extra></extra>'
            )
        )

    fig.update_layout(
        title=dict(text="GPU Utilization", font=dict(size=16, color=COLORS['dark'])),
        xaxis=dict(title="Iteration", showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(title="Utilization (%)", range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
        height=350,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_memory_usage(metrics_dict):
    """Plot GPU memory usage"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("GPU Memory Usage",)
    )

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['gpu_memory_percent'],
                mode='lines',
                name=f'Rank {rank}',
                line=dict(width=2)
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Memory Usage (%)", range=[0, 100])
    fig.update_layout(height=400, hovermode='x unified')

    return fig


def plot_throughput(metrics_dict):
    """Plot training throughput"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Training Throughput",)
    )

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['throughput'],
                mode='lines',
                name=f'Rank {rank}',
                line=dict(width=2)
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Throughput (samples/s)")
    fig.update_layout(height=400, hovermode='x unified')

    return fig


def plot_loss(metrics_dict):
    """Plot training loss with professional styling"""
    fig = go.Figure()

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]

    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['loss'],
                mode='lines',
                name=f'Worker {rank}',
                line=dict(width=3, color=colors[idx % len(colors)]),
                hovertemplate='<b>Worker %{fullData.name}</b><br>Iteration: %{x}<br>Loss: %{y:.4f}<extra></extra>'
            )
        )

    fig.update_layout(
        title=dict(text="Training Loss", font=dict(size=16, color=COLORS['dark'])),
        xaxis=dict(title="Iteration", showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(title="Loss", showgrid=True, gridcolor='#e5e7eb'),
        height=350,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_iteration_time_breakdown(metrics_dict):
    """Plot iteration time breakdown"""
    # Calculate averages for latest N iterations
    n = 50
    breakdown_data = []

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)
        if len(df) == 0:
            continue

        recent = df.tail(n)

        breakdown_data.append({
            'Rank': rank,
            'Data Loading': recent['data_loading_time'].mean(),
            'Forward': recent['forward_time'].mean(),
            'Backward': recent['backward_time'].mean(),
            'Optimizer': recent['optimizer_time'].mean(),
        })

    if not breakdown_data:
        return go.Figure()

    df_breakdown = pd.DataFrame(breakdown_data)

    fig = go.Figure()

    for col in ['Data Loading', 'Forward', 'Backward', 'Optimizer']:
        fig.add_trace(go.Bar(
            name=col,
            x=df_breakdown['Rank'],
            y=df_breakdown[col],
        ))

    fig.update_layout(
        barmode='stack',
        title="Iteration Time Breakdown (avg last 50 iters)",
        xaxis_title="Rank",
        yaxis_title="Time (s)",
        height=400
    )

    return fig


def plot_gpu_comparison(gpu_profiles):
    """Plot GPU comparison"""
    if not gpu_profiles:
        return go.Figure()

    df = pd.DataFrame(gpu_profiles)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Compute Score", "Memory (MB)")
    )

    # Compute scores
    fig.add_trace(
        go.Bar(
            x=df['device_id'],
            y=df['compute_score'],
            name='Compute Score',
            marker_color='lightblue'
        ),
        row=1, col=1
    )

    # Memory
    fig.add_trace(
        go.Bar(
            x=df['device_id'],
            y=df['total_memory_mb'],
            name='Memory (MB)',
            marker_color='lightgreen'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="GPU ID", row=1, col=1)
    fig.update_xaxes(title_text="GPU ID", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)

    return fig


def display_current_stats(metrics_dict):
    """Display current statistics"""
    cols = st.columns(len(metrics_dict) if metrics_dict else 1)

    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue

        with cols[idx]:
            latest = metrics[-1] if metrics else {}

            st.metric(
                label=f"Rank {rank} - GPU Util",
                value=f"{latest.get('gpu_utilization', 0):.1f}%"
            )
            st.metric(
                label=f"Rank {rank} - Memory",
                value=f"{latest.get('gpu_memory_percent', 0):.1f}%"
            )
            st.metric(
                label=f"Rank {rank} - Throughput",
                value=f"{latest.get('throughput', 0):.1f} samples/s"
            )


def main():
    """Main dashboard function"""

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚡ Heterogeneous Cluster Monitor")
        st.caption("Real-time distributed training visualization")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Experiment selection (compact)
    experiments_base = Path("experiments")
    available_experiments = []
    if experiments_base.exists():
        available_experiments = [d.name for d in experiments_base.iterdir()
                               if d.is_dir() and (d / "logs").exists()]

    if available_experiments:
        experiment_name = st.selectbox(
            "📁 Experiment",
            options=available_experiments,
            index=len(available_experiments) - 1,  # Latest experiment
            label_visibility="collapsed"
        )
        metrics_dir = f"experiments/{experiment_name}/logs"
        config_dir = f"experiments/{experiment_name}/configs"
    else:
        st.warning("⚠️ No experiments found. Start training to see metrics.")
        metrics_dir = "experiments/logs"
        config_dir = "experiments/configs"

    # Load data
    metrics_dict = load_metrics(metrics_dir)
    gpu_profiles = load_gpu_profiles(config_dir)

    if not metrics_dict:
        st.info("📊 Waiting for training data...")
        st.caption(f"Looking in: `{metrics_dir}`")
        return

    # === CLUSTER STATUS ===
    st.subheader("🖥️ Cluster Status")
    status_cols = st.columns(len(metrics_dict) if metrics_dict else 1)

    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        with status_cols[idx]:
            latest = metrics[-1] if metrics else {}
            gpu_util = latest.get('gpu_utilization', 0)

            # Status indicator
            status_color = COLORS['success'] if gpu_util > 10 else COLORS['danger']
            st.markdown(f'<span class="status-badge" style="background-color: {status_color};">Worker {rank}</span>',
                       unsafe_allow_html=True)

            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("GPU", f"{gpu_util:.0f}%", delta=None)
                st.metric("Memory", f"{latest.get('gpu_memory_percent', 0):.0f}%")
            with col2:
                st.metric("Throughput", f"{latest.get('throughput', 0):.0f}")
                st.metric("Loss", f"{latest.get('loss', 0):.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # === TRAINING METRICS ===
    st.subheader("📈 Training Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_loss(metrics_dict), use_container_width=True)
    with col2:
        st.plotly_chart(plot_throughput(metrics_dict), use_container_width=True)

    # === GPU METRICS ===
    st.subheader("⚙️ GPU Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_gpu_utilization(metrics_dict), use_container_width=True)
    with col2:
        st.plotly_chart(plot_memory_usage(metrics_dict), use_container_width=True)

    # === PERFORMANCE BREAKDOWN ===
    st.subheader("🔍 Performance Breakdown")
    st.plotly_chart(plot_iteration_time_breakdown(metrics_dict), use_container_width=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
