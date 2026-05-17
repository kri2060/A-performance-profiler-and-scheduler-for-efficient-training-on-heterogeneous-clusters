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



# ---------------------------------------------------------------------------
# DBS: Adaptive balancing visualisations
# ---------------------------------------------------------------------------

def plot_partition_distribution(metrics_dict):
    """Bar chart showing per-worker data partition share over iterations.

    Reads the ``batch_size`` field (proxy for partition) from each worker's
    metric stream and computes the relative share so charts are comparable
    across different total batch sizes.
    """
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]

    all_sizes = {}
    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue
        df = pd.DataFrame(metrics)
        if 'batch_size' not in df.columns:
            continue
        all_sizes[rank] = df

    if not all_sizes:
        fig.update_layout(
            title="DBS Partition Distribution",
            annotations=[dict(text="No batch_size data", showarrow=False, x=0.5, y=0.5)]
        )
        return fig

    # Compute relative share per iteration across all active workers
    for idx, (rank, df) in enumerate(all_sizes.items()):
        fig.add_trace(go.Scatter(
            x=df.get('iteration', list(range(len(df)))),
            y=df['batch_size'],
            mode='lines',
            name=f'Worker {rank}',
            line=dict(width=2, color=colors[idx % len(colors)]),
            fill='tozeroy',
            fillcolor=colors[idx % len(colors)].replace(')', ', 0.08)').replace('rgb', 'rgba'),
            hovertemplate='<b>Worker %{fullData.name}</b><br>Iter: %{x}<br>Batch: %{y}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(text="Dynamic Batch Redistribution (Partition Size vs Time)", font=dict(size=14)),
        xaxis=dict(title="Iteration", showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(title="Batch Size", showgrid=True, gridcolor='#e5e7eb'),
        height=350,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_sync_overhead(metrics_dict):
    """Stacked area chart comparing compute vs. sync time per worker.

    Reads ``forward_time + backward_time + optimizer_time`` (compute)
    and derives sync_time as the remainder of ``iteration_time``.
    """
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['danger'], COLORS['success'], COLORS['warning']]

    has_data = False
    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue
        df = pd.DataFrame(metrics)
        required = {'iteration_time', 'forward_time', 'backward_time', 'optimizer_time'}
        if not required.issubset(df.columns):
            continue

        has_data = True
        compute = df['forward_time'] + df['backward_time'] + df['optimizer_time']
        sync = (df['iteration_time'] - compute).clip(lower=0)
        iters = df.get('iteration', pd.Series(range(len(df))))

        fig.add_trace(go.Scatter(
            x=iters, y=compute,
            name=f'W{rank} compute',
            mode='lines',
            line=dict(width=2, color=colors[idx % len(colors)]),
            hovertemplate='Compute: %{y:.3f}s<extra>Worker %{fullData.name}</extra>',
        ))
        fig.add_trace(go.Scatter(
            x=iters, y=sync,
            name=f'W{rank} sync',
            mode='lines',
            line=dict(width=2, dash='dot', color=colors[idx % len(colors)]),
            hovertemplate='Sync: %{y:.3f}s<extra>Worker %{fullData.name}</extra>',
        ))

    if not has_data:
        fig.update_layout(
            title="Sync Overhead vs Compute",
            annotations=[dict(text="No timing data", showarrow=False, x=0.5, y=0.5)]
        )
        return fig

    fig.update_layout(
        title=dict(text="Straggler Detection: Sync Overhead vs Compute Time", font=dict(size=14)),
        xaxis=dict(title="Iteration", showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(title="Time (s)", showgrid=True, gridcolor='#e5e7eb'),
        height=350,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_straggler_timeline(metrics_dict):
    """Scatter plot showing straggler events per worker over time.

    Reads the ``is_straggler`` flag (1/0) written by the demo script.
    Falls back to flagging any iteration whose iteration_time is >1.8×
    the median for that worker.
    """
    import statistics
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['danger']]

    has_data = False
    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue
        df = pd.DataFrame(metrics)
        iters = df.get('iteration', pd.Series(range(len(df))))

        # Use explicit flag if available, else derive from timing
        if 'is_straggler' in df.columns:
            straggler_mask = df['is_straggler'] == 1
        elif 'iteration_time' in df.columns:
            med = df['iteration_time'].median()
            straggler_mask = df['iteration_time'] > (med * 1.8)
        else:
            continue

        has_data = True
        straggler_iters = iters[straggler_mask]
        normal_iters    = iters[~straggler_mask]

        fig.add_trace(go.Scatter(
            x=normal_iters, y=[rank] * len(normal_iters),
            mode='markers', name=f'W{rank} normal',
            marker=dict(color=colors[idx % len(colors)], size=6, opacity=0.4),
            hovertemplate=f'Worker {rank}<br>Iter: %{{x}}<br>Status: Normal<extra></extra>',
        ))
        if len(straggler_iters):
            fig.add_trace(go.Scatter(
                x=straggler_iters, y=[rank] * len(straggler_iters),
                mode='markers', name=f'W{rank} STRAGGLER',
                marker=dict(color=COLORS['danger'], size=10, symbol='x',
                            line=dict(width=2)),
                hovertemplate=f'<b>Worker {rank} STRAGGLER</b><br>Iter: %{{x}}<extra></extra>',
            ))

    if not has_data:
        fig.update_layout(
            title='Straggler Detection Timeline',
            annotations=[dict(text='No straggler data', showarrow=False, x=0.5, y=0.5)]
        )
        return fig

    fig.update_layout(
        title=dict(text='Straggler Detection Timeline (✕ = fault event)', font=dict(size=14)),
        xaxis=dict(title='Iteration', showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(title='Worker Rank', tickvals=list(range(len(metrics_dict))),
                   ticktext=[f'Worker {r}' for r in metrics_dict.keys()],
                   showgrid=False),
        height=300,
        hovermode='closest',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter, sans-serif'),
        margin=dict(l=80, r=40, t=60, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def plot_partition_ratio(metrics_dict):
    """Stacked area chart of each worker's fraction of the global batch.

    Uses ``partition_ratio`` if present, otherwise normalises ``batch_size``.
    This is the primary visual proof that DBS is actively redistributing work.
    """
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]

    all_dfs = {}
    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue
        df = pd.DataFrame(metrics)
        if 'partition_ratio' in df.columns:
            all_dfs[rank] = df
        elif 'batch_size' in df.columns:
            # Normalise batch_size across ranks at each iteration
            all_dfs[rank] = df

    if not all_dfs:
        fig.update_layout(
            title='Adaptive Partition Ratio',
            annotations=[dict(text='No partition data', showarrow=False, x=0.5, y=0.5)]
        )
        return fig

    # If batch_size only, compute fraction per iteration step
    if 'partition_ratio' not in pd.DataFrame(next(iter(all_dfs.values()))).columns:
        min_len = min(len(df) for df in all_dfs.values())
        totals = np.zeros(min_len)
        for df in all_dfs.values():
            totals += df['batch_size'].values[:min_len]
        for rank, df in all_dfs.items():
            df = df.copy()
            df['partition_ratio'] = df['batch_size'].values[:min_len] / np.maximum(totals, 1)
            all_dfs[rank] = df

    for idx, (rank, df) in enumerate(all_dfs.items()):
        iters = df.get('iteration', pd.Series(range(len(df))))
        ratios = df['partition_ratio'] * 100  # as percentage
        fig.add_trace(go.Scatter(
            x=iters, y=ratios,
            mode='lines',
            name=f'Worker {rank}',
            line=dict(width=2, color=colors[idx % len(colors)]),
            fill='tonexty' if idx > 0 else 'tozeroy',
            fillcolor=colors[idx % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.15)'),
            hovertemplate=f'<b>Worker {rank}</b><br>Iter: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>',
        ))

    fig.update_layout(
        title=dict(text='Adaptive Partition Ratio — DBS Load Redistribution (%)', font=dict(size=14)),
        xaxis=dict(title='Iteration', showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(title='Batch Share (%)', range=[0, 105], showgrid=True, gridcolor='#e5e7eb'),
        height=350,
        hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter, sans-serif'),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def main():
    """Main dashboard function"""

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚡ Adaptive Load Balancer Dashboard")
        st.caption("Real-time distributed training & heterogeneous worker monitoring")
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

    # ==========================================
    # DEMO PLAYBACK CONTROLS
    # ==========================================
    st.sidebar.header("🎬 Demo Controls")
    simulate_live = st.sidebar.checkbox("Simulate Live Run", value=False)
    
    if simulate_live and metrics_dict:
        # Find the maximum iterations available in the loaded logs
        max_iters = max([len(m) for m in metrics_dict.values() if m] or [0])
        
        # Initialize session state for playback
        if 'playback_step' not in st.session_state:
            st.session_state.playback_step = 5  # Start with a few data points
            
        playback_speed = st.sidebar.slider("Playback Speed (iters/tick)", 1, 20, 5)
        
        if st.sidebar.button("Restart Simulation"):
            st.session_state.playback_step = 5
            st.rerun()
            
        # Slice the metrics array to simulate data arriving over time
        sliced_metrics = {}
        for rank, m in metrics_dict.items():
            sliced_metrics[rank] = m[:st.session_state.playback_step]
        metrics_dict = sliced_metrics
        
        # Display progress
        st.sidebar.progress(min(st.session_state.playback_step / max(max_iters, 1), 1.0))
        st.sidebar.caption(f"Iteration: {min(st.session_state.playback_step, max_iters)} / {max_iters}")
        
        # Auto-increment the playback step
        if st.session_state.playback_step < max_iters:
            st.session_state.playback_step += playback_speed
            time.sleep(1) # Visual delay for realism
            st.rerun()
        else:
            st.sidebar.success("Training Complete!")
            
        # Override auto-refresh when simulating
        auto_refresh = False

    if not metrics_dict:
        st.info("📊 Waiting for training data...")
        st.caption(f"Looking in: `{metrics_dir}`")
        return

    # === CLUSTER STATUS ===
    st.subheader("🖥️ Heterogeneous Worker Status")
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
        st.plotly_chart(plot_loss(metrics_dict), use_container_width=True, key="chart_loss")
    with col2:
        st.plotly_chart(plot_throughput(metrics_dict), use_container_width=True, key="chart_throughput")

    # === GPU METRICS ===
    st.subheader("⚙️ GPU Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_gpu_utilization(metrics_dict), use_container_width=True, key="chart_gpu_util")
    with col2:
        st.plotly_chart(plot_memory_usage(metrics_dict), use_container_width=True, key="chart_gpu_mem")

    # === PERFORMANCE BREAKDOWN ===
    st.subheader("🔍 Performance Breakdown")
    st.plotly_chart(plot_iteration_time_breakdown(metrics_dict), use_container_width=True, key="chart_time_breakdown")

    # === DBS ADAPTIVE BALANCING ===
    st.subheader("⚖️ Dynamic Batch Redistribution & Straggler Detection")

    # Row 1: Straggler timeline (full width — most dramatic for reviewers)
    st.plotly_chart(plot_straggler_timeline(metrics_dict), use_container_width=True, key="chart_straggler_timeline")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_partition_ratio(metrics_dict), use_container_width=True, key="chart_partition_ratio")
    with col2:
        st.plotly_chart(plot_sync_overhead(metrics_dict), use_container_width=True, key="chart_sync_overhead")

    st.plotly_chart(plot_partition_distribution(metrics_dict), use_container_width=True, key="chart_partition_dist")

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
